package com.ndl.mainapp.ocr

import android.util.Log
import com.ndl.mainapp.ocr.model.OcrLine
import com.ndl.mainapp.ocr.model.OcrSnapshot
import com.ndl.mainapp.ocr.model.PipelineStatus
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

class OcrOrchestrator(
    private val frameStore: LatestFrameStore,
    private val roiState: RoiState,
    private val detectorEngine: DetectorEngine,
    private val recognizerEngine: RecognizerEngine,
    private val metrics: MetricsCollector,
) {
    companion object {
        private const val TAG = "OcrOrchestrator"
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val detectorRunning = AtomicBoolean(false)
    private val recognizerRunning = AtomicBoolean(false)

    private var detectorIntervalMs: Long = 1000L
    private var recognizerIntervalMs: Long = 500L
    private var detectorTimeoutMs: Long = 12000L
    private var recognizerTimeoutMs: Long = 12000L

    private var detectorJob: Job? = null
    private var recognizerJob: Job? = null
    private var latestLines = emptyList<OcrLine>()

    private val _snapshot = MutableStateFlow<OcrSnapshot?>(null)
    val snapshot: StateFlow<OcrSnapshot?> = _snapshot.asStateFlow()
    private val _status = MutableStateFlow(
        PipelineStatus(
            detectorRunning = false,
            recognizerRunning = false,
            lastDetectorRoiCount = 0,
            lastRecognizerLineCount = 0,
            lastError = null,
            updatedAtMs = System.currentTimeMillis(),
        )
    )
    val status: StateFlow<PipelineStatus> = _status.asStateFlow()

    fun setDetectorIntervalMs(ms: Long) {
        detectorIntervalMs = ms.coerceAtLeast(200L)
    }

    fun setRecognizerIntervalMs(ms: Long) {
        recognizerIntervalMs = ms.coerceAtLeast(100L)
    }

    fun setDetectorTimeoutMs(ms: Long) {
        detectorTimeoutMs = ms.coerceAtLeast(1000L)
    }

    fun setRecognizerTimeoutMs(ms: Long) {
        recognizerTimeoutMs = ms.coerceAtLeast(1000L)
    }

    fun start() {
        if (detectorJob?.isActive == true || recognizerJob?.isActive == true) {
            return
        }
        Log.i(TAG, "start detectorInterval=${detectorIntervalMs}ms recognizerInterval=${recognizerIntervalMs}ms")

        detectorJob = scope.launch {
            while (isActive) {
                val didRun = try {
                    runDetectorOnce()
                } catch (e: Exception) {
                    Log.e(TAG, "detector loop failed", e)
                    publishError("detector error: ${e.message}")
                    false
                }
                try {
                    delay(if (didRun) detectorIntervalMs else 250L)
                } catch (_: Exception) {
                    // no-op
                }
            }
        }

        recognizerJob = scope.launch {
            while (isActive) {
                val didRun = try {
                    runRecognizerOnce()
                } catch (e: Exception) {
                    Log.e(TAG, "recognizer loop failed", e)
                    publishError("recognizer error: ${e.message}")
                    false
                }
                try {
                    delay(if (didRun) recognizerIntervalMs else 250L)
                } catch (_: Exception) {
                    // no-op
                }
            }
        }
    }

    fun stop() {
        detectorJob?.cancel()
        recognizerJob?.cancel()
        detectorJob = null
        recognizerJob = null
    }

    private suspend fun runDetectorOnce(): Boolean {
        if (!detectorRunning.compareAndSet(false, true)) {
            return false
        }
        markRunning(detector = true, recognizer = recognizerRunning.get())
        try {
            val frame = frameStore.getLatest()
            if (frame == null) {
                Log.d(TAG, "det skipped: no frame")
                return false
            }
            val t0 = System.currentTimeMillis()
            Log.i(TAG, "det start frame=${frame.id} size=${frame.width}x${frame.height}")
            val rois = withTimeoutOrNull(detectorTimeoutMs) {
                detectorEngine.detect(frame)
            }
            if (rois == null) {
                publishError("detector timeout ${detectorTimeoutMs}ms")
                Log.w(TAG, "det timeout frame=${frame.id}")
                return false
            }
            val dt = System.currentTimeMillis() - t0
            metrics.updateDetector(dt)
            roiState.update(rois, frame.timestampMs)
            Log.d(TAG, "det frame=${frame.id} roi=${rois.size} detMs=$dt")

            _snapshot.value = OcrSnapshot(
                frameId = frame.id,
                frameWidth = frame.width,
                frameHeight = frame.height,
                timestampMs = System.currentTimeMillis(),
                rois = rois,
                lines = latestLines,
                metrics = metrics.snapshot(),
            )
            _status.value = _status.value.copy(
                detectorRunning = true,
                lastDetectorRoiCount = rois.size,
                lastError = null,
                updatedAtMs = System.currentTimeMillis(),
            )
            return true
        } finally {
            detectorRunning.set(false)
            markRunning(detector = false, recognizer = recognizerRunning.get())
        }
    }

    private suspend fun runRecognizerOnce(): Boolean {
        if (!recognizerRunning.compareAndSet(false, true)) {
            return false
        }
        markRunning(detector = detectorRunning.get(), recognizer = true)
        try {
            val frame = frameStore.getLatest()
            if (frame == null) {
                Log.d(TAG, "rec skipped: no frame")
                return false
            }
            val roiSnapshot = roiState.get()
            if (roiSnapshot.rois.isEmpty()) {
                Log.d(TAG, "rec frame=${frame.id} skipped: no roi")
                _status.value = _status.value.copy(
                    recognizerRunning = true,
                    lastRecognizerLineCount = 0,
                    lastError = null,
                    updatedAtMs = System.currentTimeMillis(),
                )
                return false
            }

            val t0 = System.currentTimeMillis()
            Log.i(TAG, "rec start frame=${frame.id} roi=${roiSnapshot.rois.size}")
            val lines = withTimeoutOrNull(recognizerTimeoutMs) {
                recognizerEngine.recognize(frame, roiSnapshot.rois)
            }
            if (lines == null) {
                publishError("recognizer timeout ${recognizerTimeoutMs}ms")
                Log.w(TAG, "rec timeout frame=${frame.id}")
                return false
            }
            val rt = System.currentTimeMillis() - t0
            metrics.updateRecognizer(rt)
            latestLines = lines
            Log.d(TAG, "rec frame=${frame.id} lines=${lines.size} recMs=$rt")

            _snapshot.value = OcrSnapshot(
                frameId = frame.id,
                frameWidth = frame.width,
                frameHeight = frame.height,
                timestampMs = System.currentTimeMillis(),
                rois = roiSnapshot.rois,
                lines = lines,
                metrics = metrics.snapshot(),
            )
            _status.value = _status.value.copy(
                recognizerRunning = true,
                lastRecognizerLineCount = lines.size,
                lastError = null,
                updatedAtMs = System.currentTimeMillis(),
            )
            return true
        } finally {
            recognizerRunning.set(false)
            markRunning(detector = detectorRunning.get(), recognizer = false)
        }
    }

    private fun markRunning(detector: Boolean, recognizer: Boolean) {
        _status.value = _status.value.copy(
            detectorRunning = detector,
            recognizerRunning = recognizer,
            updatedAtMs = System.currentTimeMillis(),
        )
    }

    private fun publishError(message: String) {
        _status.value = _status.value.copy(
            lastError = message,
            updatedAtMs = System.currentTimeMillis(),
        )
    }
}
