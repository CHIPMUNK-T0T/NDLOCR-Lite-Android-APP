package com.ndl.mainapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.TypedValue
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.ndl.mainapp.ocr.CameraPipeline
import com.ndl.mainapp.ocr.DetectorEngine
import com.ndl.mainapp.ocr.ImageUtils
import com.ndl.mainapp.ocr.LatestFrameStore
import com.ndl.mainapp.ocr.NoopDetectorEngine
import com.ndl.mainapp.ocr.NoopRecognizerEngine
import com.ndl.mainapp.ocr.RecognizerEngine
import com.ndl.mainapp.ocr.model.FramePacket
import com.ndl.mainapp.ocr.model.MetricsSnapshot
import com.ndl.mainapp.ocr.model.OcrSnapshot
import com.ndl.mainapp.ocr.ort.OrtDetectorEngine
import com.ndl.mainapp.ocr.ort.OrtRecognizerEngine
import com.ndl.mainapp.ocr.ui.OverlayView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "MainActivity"
    }

    private enum class UiMode {
        LIVE,
        PROCESSING,
        RESULT,
    }

    private lateinit var rootContainer: FrameLayout
    private lateinit var previewView: PreviewView
    private lateinit var frozenImageView: ImageView
    private lateinit var overlayView: OverlayView
    private lateinit var metricsText: TextView
    private lateinit var impatientText: TextView
    private lateinit var captureButton: Button

    private val frameStore = LatestFrameStore()
    private lateinit var detectorEngine: DetectorEngine
    private lateinit var recognizerEngine: RecognizerEngine
    private var cameraPipeline: CameraPipeline? = null

    private var uiMode: UiMode = UiMode.LIVE
    private var processingJob: Job? = null
    private var impatientHideJob: Job? = null
    private var frozenBitmap: Bitmap? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            startPipeline()
        } else {
            metricsText.text = "Camera permission is required"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        rootContainer = findViewById(R.id.rootContainer)
        previewView = findViewById(R.id.previewView)
        frozenImageView = findViewById(R.id.frozenImageView)
        overlayView = findViewById(R.id.overlayView)
        metricsText = findViewById(R.id.metricsText)
        impatientText = findViewById(R.id.impatientText)
        captureButton = findViewById(R.id.captureButton)

        impatientText.setTextSize(TypedValue.COMPLEX_UNIT_PX, OverlayView.OCR_TEXT_SIZE_PX * 2f)
        metricsText.text = "ライブ表示中。下のOCRボタンで実行"

        initEngines()
        bindUi()

        if (hasCameraPermission()) {
            startPipeline()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        processingJob?.cancel()
        impatientHideJob?.cancel()
        cameraPipeline?.stop()
        cameraPipeline = null
        (detectorEngine as? AutoCloseable)?.close()
        (recognizerEngine as? AutoCloseable)?.close()
        clearFrozenImage()
    }

    private fun bindUi() {
        captureButton.setOnClickListener {
            onCaptureClicked()
        }

        rootContainer.setOnClickListener {
            if (uiMode != UiMode.LIVE) {
                backToLiveView()
            }
        }
    }

    private fun onCaptureClicked() {
        if (processingJob?.isActive == true) {
            showImpatientMessage()
            return
        }
        runOneShotOcr()
    }

    private fun runOneShotOcr() {
        val displayBitmap = captureDisplayedBitmap()
        if (displayBitmap == null) {
            metricsText.text = "表示フレーム未取得。少し待って再実行してください"
            return
        }

        val inferBitmap = ImageUtils.resizeLongEdge(displayBitmap, 1000)
        val inferInput = if (inferBitmap === displayBitmap) {
            displayBitmap.copy(Bitmap.Config.ARGB_8888, false)
                ?: Bitmap.createScaledBitmap(displayBitmap, displayBitmap.width, displayBitmap.height, true)
        } else {
            inferBitmap
        }
        val frame = FramePacket(
            id = System.currentTimeMillis(),
            timestampMs = System.currentTimeMillis(),
            width = inferInput.width,
            height = inferInput.height,
            bitmap = inferInput,
        )

        enterOcrView(displayBitmap)
        metricsText.text = "OCR処理中..."
        impatientText.visibility = View.GONE

        processingJob = lifecycleScope.launch {
            try {
                val t0 = System.currentTimeMillis()
                val rois = withContext(Dispatchers.Default) {
                    detectorEngine.detect(frame)
                }
                val detMs = System.currentTimeMillis() - t0

                val t1 = System.currentTimeMillis()
                val lines = if (rois.isEmpty()) {
                    emptyList()
                } else {
                    withContext(Dispatchers.Default) {
                        recognizerEngine.recognize(frame, rois)
                    }
                }
                val recMs = System.currentTimeMillis() - t1
                if (!isActive) return@launch

                val snapshot = OcrSnapshot(
                    frameId = frame.id,
                    frameWidth = frame.width,
                    frameHeight = frame.height,
                    timestampMs = System.currentTimeMillis(),
                    rois = rois,
                    lines = lines,
                    metrics = MetricsSnapshot(
                        detectorMs = detMs,
                        recognizerMs = recMs,
                        droppedFrames = frameStore.droppedFrames(),
                    ),
                )
                overlayView.render(snapshot)
                metricsText.text = buildString {
                    append("OCR完了\n")
                    append("roi=${rois.size} lines=${lines.size}\n")
                    append("det=${detMs}ms rec=${recMs}ms")
                }
                uiMode = UiMode.RESULT
                Log.i(TAG, "oneshot done roi=${rois.size} lines=${lines.size} det=${detMs} rec=${recMs}")
            } catch (e: Exception) {
                Log.e(TAG, "oneshot OCR failed", e)
                metricsText.text = "OCR失敗: ${e.message}"
                uiMode = UiMode.RESULT
            } finally {
                if (!frame.bitmap.isRecycled) {
                    frame.bitmap.recycle()
                }
                processingJob = null
            }
        }
    }

    private fun captureDisplayedBitmap(): Bitmap? {
        val fromPreview = previewView.bitmap
        if (fromPreview != null) {
            return fromPreview.copy(Bitmap.Config.ARGB_8888, false)
        }
        val fallback = frameStore.snapshotCopy()
        if (fallback != null) {
            return fallback.bitmap
        }
        return null
    }

    private fun enterOcrView(bitmap: Bitmap) {
        uiMode = UiMode.PROCESSING
        overlayView.render(null)
        clearFrozenImage()
        frozenBitmap = bitmap
        frozenImageView.setImageBitmap(bitmap)
        frozenImageView.visibility = View.VISIBLE
    }

    private fun backToLiveView() {
        processingJob?.cancel()
        processingJob = null
        uiMode = UiMode.LIVE
        impatientText.visibility = View.GONE
        overlayView.render(null)
        clearFrozenImage()
        metricsText.text = "ライブ表示中。下のOCRボタンで実行"
    }

    private fun clearFrozenImage() {
        frozenImageView.setImageBitmap(null)
        frozenImageView.visibility = View.GONE
        frozenBitmap = null
    }

    private fun showImpatientMessage() {
        impatientHideJob?.cancel()
        impatientText.visibility = View.VISIBLE
        impatientHideJob = lifecycleScope.launch {
            delay(1600L)
            if (processingJob?.isActive != true) {
                impatientText.visibility = View.GONE
            }
        }
    }

    private fun startPipeline() {
        Log.i(TAG, "startPipeline")
        if (cameraPipeline == null) {
            cameraPipeline = CameraPipeline(
                context = this,
                lifecycleOwner = this,
                previewView = previewView,
                frameStore = frameStore,
            )
        }
        cameraPipeline?.start()
        if (uiMode == UiMode.LIVE) {
            metricsText.text = "ライブ表示中。下のOCRボタンで実行"
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA,
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun initEngines() {
        try {
            detectorEngine = OrtDetectorEngine(this)
            recognizerEngine = OrtRecognizerEngine(this)
            metricsText.text = "ORT engines loaded"
            Log.i(TAG, "ORT engines loaded")
        } catch (e: Exception) {
            detectorEngine = NoopDetectorEngine()
            recognizerEngine = NoopRecognizerEngine()
            metricsText.text = "ORT init failed, fallback noop: ${e.message}"
            Log.e(TAG, "ORT init failed, fallback noop", e)
        }
    }
}
