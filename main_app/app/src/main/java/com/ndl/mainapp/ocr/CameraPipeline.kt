package com.ndl.mainapp.ocr

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.ndl.mainapp.ocr.model.FramePacket
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

class CameraPipeline(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val frameStore: LatestFrameStore,
    private val analysisResolution: Size = Size(1280, 960),
    private val captureIntervalMs: Long = 250L,
    private val normalizedLongEdgePx: Int = 1000,
) {
    companion object {
        private const val TAG = "CameraPipeline"
    }

    private val frameIdGen = AtomicLong(0)
    private val lastAcceptedTs = AtomicLong(0L)
    private val analyzedCount = AtomicLong(0L)
    private val acceptedCount = AtomicLong(0L)
    private val analyzerExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    fun start(cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            cameraProvider = provider

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(analysisResolution)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analysis.setAnalyzer(analyzerExecutor) { imageProxy ->
                try {
                    val n = analyzedCount.incrementAndGet()
                    if (n <= 5 || n % 60L == 0L) {
                        Log.i(
                            TAG,
                            "analyze #$n format=${imageProxy.format} planes=${imageProxy.planes.size} " +
                                "size=${imageProxy.width}x${imageProxy.height}",
                        )
                    }
                    val now = System.currentTimeMillis()
                    val last = lastAcceptedTs.get()
                    if (now - last < captureIntervalMs) {
                        return@setAnalyzer
                    }

                    val bitmap = ImageUtils.imageProxyToBitmap(imageProxy)
                    if (bitmap != null) {
                        val normalized = ImageUtils.resizeLongEdge(bitmap, normalizedLongEdgePx)
                        if (normalized !== bitmap) {
                            bitmap.recycle()
                        }
                        lastAcceptedTs.set(now)
                        val accepted = acceptedCount.incrementAndGet()
                        val frame = FramePacket(
                            id = frameIdGen.incrementAndGet(),
                            timestampMs = now,
                            width = normalized.width,
                            height = normalized.height,
                            bitmap = normalized,
                        )
                        frameStore.update(frame)
                        if (accepted <= 5 || accepted % 20L == 0L) {
                            Log.i(
                                TAG,
                                "accepted #$accepted frameId=${frame.id} bitmap=${frame.width}x${frame.height} " +
                                    "longEdgeTarget=$normalizedLongEdgePx",
                            )
                        }
                    } else {
                        Log.w(TAG, "imageProxyToBitmap returned null")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "analysis failed", e)
                } finally {
                    imageProxy.close()
                }
            }

            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, analysis)
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        cameraProvider?.unbindAll()
        frameStore.clear()
        analyzerExecutor.shutdown()
    }
}
