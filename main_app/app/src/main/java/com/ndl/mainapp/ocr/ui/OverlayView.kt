package com.ndl.mainapp.ocr.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.ndl.mainapp.ocr.model.OcrSnapshot

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    companion object {
        const val OCR_TEXT_SIZE_PX: Float = 26f
    }

    private val boxPaint = Paint().apply {
        color = Color.argb(220, 0, 230, 180)
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = OCR_TEXT_SIZE_PX
        isAntiAlias = true
    }

    private var snapshot: OcrSnapshot? = null

    fun render(newSnapshot: OcrSnapshot?) {
        snapshot = newSnapshot
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val snap = snapshot ?: return
        if (snap.frameWidth <= 0 || snap.frameHeight <= 0) return

        val scaleX = width.toFloat() / snap.frameWidth.toFloat()
        val scaleY = height.toFloat() / snap.frameHeight.toFloat()

        val textMap = snap.lines.associateBy {
            "${it.roi.x1},${it.roi.y1},${it.roi.x2},${it.roi.y2}"
        }

        for (roi in snap.rois) {
            val x1 = roi.x1 * scaleX
            val y1 = roi.y1 * scaleY
            val x2 = roi.x2 * scaleX
            val y2 = roi.y2 * scaleY
            canvas.drawRect(x1, y1, x2, y2, boxPaint)

            val key = "${roi.x1},${roi.y1},${roi.x2},${roi.y2}"
            val text = textMap[key]?.text?.takeIf { it.isNotBlank() } ?: continue

            val textWidth = textPaint.measureText(text)
            val textHeight = textPaint.textSize + 8f
            canvas.drawRect(x1, y1 - textHeight, x1 + textWidth + 12f, y1, textBgPaint)
            canvas.drawText(text, x1 + 6f, y1 - 6f, textPaint)
        }
    }
}
