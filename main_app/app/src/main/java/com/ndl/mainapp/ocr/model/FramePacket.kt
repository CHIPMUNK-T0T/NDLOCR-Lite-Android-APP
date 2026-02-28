package com.ndl.mainapp.ocr.model

import android.graphics.Bitmap

data class FramePacket(
    val id: Long,
    val timestampMs: Long,
    val width: Int,
    val height: Int,
    val bitmap: Bitmap,
)
