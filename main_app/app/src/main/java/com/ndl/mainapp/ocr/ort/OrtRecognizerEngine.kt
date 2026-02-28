package com.ndl.mainapp.ocr.ort

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import com.ndl.mainapp.ocr.RecognizerEngine
import com.ndl.mainapp.ocr.model.FramePacket
import com.ndl.mainapp.ocr.model.OcrLine
import com.ndl.mainapp.ocr.model.RoiBox
import java.nio.FloatBuffer

class OrtRecognizerEngine(
    context: Context,
    modelAssetPath: String = "models/parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx",
    charsetAssetPath: String = "models/charset.txt",
    private val maxRois: Int = 20,
) : RecognizerEngine, AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val outputName: String
    private val inputHeight: Int
    private val inputWidth: Int
    private val outputSeqLen: Int
    private val outputVocab: Int
    private val charset: List<String>

    init {
        charset = OrtAssets.readCharset(context, charsetAssetPath)

        val modelBytes = OrtAssets.readAssetBytes(context, modelAssetPath)
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
        }
        session = env.createSession(modelBytes, opts)

        inputName = session.inputNames.first()
        outputName = session.outputNames.first()

        val inputShape = (session.inputInfo[inputName]?.info as TensorInfo).shape
        inputHeight = inputShape[2].toInt()
        inputWidth = inputShape[3].toInt()

        val outputShape = (session.outputInfo[outputName]?.info as TensorInfo).shape
        outputSeqLen = outputShape[1].toInt()
        outputVocab = outputShape[2].toInt()
    }

    override suspend fun recognize(frame: FramePacket, rois: List<RoiBox>): List<OcrLine> {
        val out = ArrayList<OcrLine>()
        for (roi in rois.take(maxRois)) {
            val crop = cropBitmap(frame.bitmap, roi) ?: continue
            try {
                val text = runSingle(crop)
                out += OcrLine(roi = roi, text = text)
            } catch (e: Exception) {
                Log.e("OrtRecognizerEngine", "recognize failed: ${e.message}", e)
            } finally {
                crop.recycle()
            }
        }
        Log.d("OrtRecognizerEngine", "recognized=${out.size}/${rois.take(maxRois).size}")
        return out
    }

    override fun close() {
        session.close()
    }

    private fun runSingle(crop: Bitmap): String {
        val input = preprocess(crop)
        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong()),
        ).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { result ->
                val logits = flattenFloat(result[0].value)
                return decode(logits)
            }
        }
    }

    private fun preprocess(src: Bitmap): FloatArray {
        val rotated = if (src.height > src.width) {
            Bitmap.createBitmap(src, 0, 0, src.width, src.height, Matrix().apply { postRotate(90f) }, true)
        } else {
            src
        }
        val resized = Bitmap.createScaledBitmap(rotated, inputWidth, inputHeight, true)

        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        val out = FloatArray(3 * inputWidth * inputHeight)
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val idx = y * inputWidth + x
                val p = pixels[idx]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f

                // BGR and normalize to [-1, 1]
                out[idx] = 2f * (b - 0.5f)
                out[inputWidth * inputHeight + idx] = 2f * (g - 0.5f)
                out[2 * inputWidth * inputHeight + idx] = 2f * (r - 0.5f)
            }
        }
        if (rotated !== src) {
            rotated.recycle()
        }
        resized.recycle()
        return out
    }

    private fun decode(logits: FloatArray): String {
        if (logits.isEmpty()) return ""

        val sb = StringBuilder()
        for (t in 0 until outputSeqLen) {
            val base = t * outputVocab
            if (base + outputVocab > logits.size) break

            var bestIdx = 0
            var best = Float.NEGATIVE_INFINITY
            for (v in 0 until outputVocab) {
                val s = logits[base + v]
                if (s > best) {
                    best = s
                    bestIdx = v
                }
            }

            if (bestIdx == 0) break
            val charIdx = bestIdx - 1
            if (charIdx in charset.indices) {
                sb.append(charset[charIdx])
            }
        }
        return sb.toString()
    }

    private fun cropBitmap(bitmap: Bitmap, roi: RoiBox): Bitmap? {
        val x1 = roi.x1.coerceIn(0, bitmap.width - 1)
        val y1 = roi.y1.coerceIn(0, bitmap.height - 1)
        val x2 = roi.x2.coerceIn(0, bitmap.width)
        val y2 = roi.y2.coerceIn(0, bitmap.height)
        val w = x2 - x1
        val h = y2 - y1
        if (w <= 1 || h <= 1) return null
        return Bitmap.createBitmap(bitmap, x1, y1, w, h)
    }

    private fun flattenFloat(value: Any?): FloatArray {
        return when (value) {
            null -> FloatArray(0)
            is FloatArray -> value
            is Array<*> -> value.flatMap { flattenFloat(it).asList() }.toFloatArray()
            else -> FloatArray(0)
        }
    }
}
