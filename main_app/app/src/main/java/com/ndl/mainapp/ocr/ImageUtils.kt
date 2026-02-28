package com.ndl.mainapp.ocr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicInteger

object ImageUtils {
    private const val TAG = "ImageUtils"
    private val debugCounter = AtomicInteger(0)

    fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        if (debugCounter.getAndIncrement() < 5) {
            Log.d(
                TAG,
                "frame format=${image.format}, size=${image.width}x${image.height}, " +
                    "planes=${image.planes.size}, rotation=${image.imageInfo.rotationDegrees}",
            )
        }

        val bitmap = if (isRgbaImage(image)) {
            rgba8888ToBitmap(image)
        } else {
            yuv420888ToBitmap(image)
        } ?: return null

        val rotation = image.imageInfo.rotationDegrees
        if (rotation == 0) {
            return bitmap
        }
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) {
            bitmap.recycle()
        }
        return rotated
    }

    fun resizeLongEdge(src: Bitmap, targetLongEdge: Int): Bitmap {
        if (targetLongEdge <= 0) return src

        val w = src.width
        val h = src.height
        if (w <= 0 || h <= 0) return src

        val longEdge = maxOf(w, h)
        if (longEdge == targetLongEdge) {
            return src
        }

        val scale = targetLongEdge.toFloat() / longEdge.toFloat()
        val outW = (w * scale).toInt().coerceAtLeast(1)
        val outH = (h * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, outW, outH, true)
    }

    private fun isRgbaImage(image: ImageProxy): Boolean {
        val plane = image.planes.firstOrNull()
        return image.planes.size == 1 && plane != null && plane.pixelStride == 4
    }

    private fun rgba8888ToBitmap(image: ImageProxy): Bitmap? {
        val plane = image.planes.firstOrNull() ?: return null
        val width = image.width
        val height = image.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val buffer = plane.buffer

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        if (pixelStride == 4 && rowStride == width * 4) {
            buffer.rewind()
            bitmap.copyPixelsFromBuffer(buffer)
            return bitmap
        }

        // Fallback for row padding/stride mismatch.
        val out = IntArray(width * height)
        for (y in 0 until height) {
            val rowStart = y * rowStride
            for (x in 0 until width) {
                val i = rowStart + x * pixelStride
                val r = buffer.get(i).toInt() and 0xFF
                val g = buffer.get(i + 1).toInt() and 0xFF
                val b = buffer.get(i + 2).toInt() and 0xFF
                val a = if (pixelStride > 3) buffer.get(i + 3).toInt() and 0xFF else 0xFF
                out[y * width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        bitmap.setPixels(out, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun yuv420888ToBitmap(image: ImageProxy): Bitmap? {
        val nv21 = yuv420888ToNv21(image) ?: return null
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val jpegBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray? {
        if (image.planes.size < 3) return null

        val width = image.width
        val height = image.height
        val out = ByteArray(width * height + width * height / 2)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride

        var dst = 0
        for (row in 0 until height) {
            val rowOffset = row * yRowStride
            for (col in 0 until width) {
                val i = rowOffset + col * yPixelStride
                out[dst++] = yBuffer.get(i)
            }
        }

        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride
        val chromaHeight = height / 2
        val chromaWidth = width / 2

        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vIndex = row * vRowStride + col * vPixelStride
                val uIndex = row * uRowStride + col * uPixelStride
                out[dst++] = vBuffer.get(vIndex)
                out[dst++] = uBuffer.get(uIndex)
            }
        }

        return out
    }
}
