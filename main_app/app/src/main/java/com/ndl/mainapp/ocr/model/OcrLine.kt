package com.ndl.mainapp.ocr.model

data class OcrLine(
    val roi: RoiBox,
    val text: String,
)
