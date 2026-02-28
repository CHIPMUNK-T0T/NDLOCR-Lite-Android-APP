package com.ndl.mainapp.ocr.model

data class PipelineStatus(
    val detectorRunning: Boolean,
    val recognizerRunning: Boolean,
    val lastDetectorRoiCount: Int,
    val lastRecognizerLineCount: Int,
    val lastError: String?,
    val updatedAtMs: Long,
)

