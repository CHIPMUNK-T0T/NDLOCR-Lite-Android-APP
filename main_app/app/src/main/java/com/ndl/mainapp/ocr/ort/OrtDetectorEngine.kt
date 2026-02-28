package com.ndl.mainapp.ocr.ort

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.util.Log
import com.ndl.mainapp.ocr.DetectorEngine
import com.ndl.mainapp.ocr.model.FramePacket
import com.ndl.mainapp.ocr.model.RoiBox
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class OrtDetectorEngine(
    context: Context,
    modelAssetPath: String = "models/deim-s-1024x1024.onnx",
    private val confThreshold: Float = 0.20f,
) : DetectorEngine, AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = OrtAssets.readAssetBytes(context, modelAssetPath)
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
        }
        session = env.createSession(modelBytes, opts)
    }

    override suspend fun detect(frame: FramePacket): List<RoiBox> {
        val inputSize = 800
        val maxWh = max(frame.width, frame.height)

        val padded = Bitmap.createBitmap(maxWh, maxWh, Bitmap.Config.ARGB_8888)
        Canvas(padded).drawBitmap(frame.bitmap, 0f, 0f, null)
        val resized = Bitmap.createScaledBitmap(padded, inputSize, inputSize, true)

        val chw = bitmapToDetectorInput(resized)
        val sizeTensor = longArrayOf(inputSize.toLong(), inputSize.toLong())

        val rois = mutableListOf<RoiBox>()
        val scale = maxWh.toFloat() / inputSize.toFloat()

        try {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())).use { imageTensor ->
                OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(sizeTensor), longArrayOf(1, 2)).use { origSizeTensor ->
                    val inputs = mapOf(
                        "images" to imageTensor,
                        "orig_target_sizes" to origSizeTensor,
                    )
                    session.run(inputs).use { result ->
                        val labels = flattenLong(result[0].value)
                        val boxes = flattenBoxes(result[1].value)
                        val scores = flattenFloat(result[2].value)
                        val charCounts = if (result.size() >= 4) flattenLong(result[3].value) else LongArray(labels.size) { 100L }

                        val n = min(min(labels.size, scores.size), boxes.size)
                        val candidates = ArrayList<RoiBox>(n)
                        for (i in 0 until n) {
                            if (scores[i] < confThreshold) continue

                            val classIndex = labels[i].toInt() - 1
                            val b = boxes[i]
                            var x1 = (b[0] * scale).toInt()
                            var y1 = (b[1] * scale).toInt()
                            var x2 = (b[2] * scale).toInt()
                            var y2 = (b[3] * scale).toInt()

                            x1 = x1.coerceIn(0, frame.width - 1)
                            y1 = y1.coerceIn(0, frame.height - 1)
                            x2 = x2.coerceIn(0, frame.width - 1)
                            y2 = y2.coerceIn(0, frame.height - 1)
                            if (x2 <= x1 || y2 <= y1) continue

                            val predCharCnt = if (i < charCounts.size) charCounts[i].toFloat() else 100f
                            val roi = RoiBox(
                                x1 = x1,
                                y1 = y1,
                                x2 = x2,
                                y2 = y2,
                                score = scores[i],
                                predCharCnt = predCharCnt,
                            )

                            // Prefer line_main-compatible class ids.
                            if (classIndex == 0 || classIndex == 1) {
                                rois += roi
                            }
                            candidates += roi
                        }
                        if (rois.isEmpty()) {
                            rois.addAll(candidates.take(30))
                        }
                        Log.d("OrtDetectorEngine", "rois=${rois.size}, candidates=${candidates.size}, labels=${labels.size}")
                    }
                }
            }
        } finally {
            resized.recycle()
            padded.recycle()
        }

        // Japanese vertical layout heuristic: right-to-left columns.
        return rois.sortedWith(compareByDescending<RoiBox> { it.x1 }.thenBy { it.y1 })
    }

    override fun close() {
        session.close()
    }

    private fun bitmapToDetectorInput(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val out = FloatArray(3 * width * height)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val idx = y * width + x
                val p = pixels[idx]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f

                out[idx] = (r - mean[0]) / std[0]
                out[width * height + idx] = (g - mean[1]) / std[1]
                out[2 * width * height + idx] = (b - mean[2]) / std[2]
            }
        }
        return out
    }

    private fun flattenLong(value: Any?): LongArray {
        return when (value) {
            null -> LongArray(0)
            is LongArray -> value
            is Array<*> -> value.flatMap { flattenLong(it).asList() }.toLongArray()
            else -> LongArray(0)
        }
    }

    private fun flattenFloat(value: Any?): FloatArray {
        return when (value) {
            null -> FloatArray(0)
            is FloatArray -> value
            is Array<*> -> value.flatMap { flattenFloat(it).asList() }.toFloatArray()
            else -> FloatArray(0)
        }
    }

    private fun flattenBoxes(value: Any?): List<FloatArray> {
        val all = flattenFloat(value)
        if (all.isEmpty()) return emptyList()
        val n = all.size / 4
        val boxes = ArrayList<FloatArray>(n)
        for (i in 0 until n) {
            boxes += floatArrayOf(
                all[i * 4],
                all[i * 4 + 1],
                all[i * 4 + 2],
                all[i * 4 + 3],
            )
        }
        return boxes
    }
}
