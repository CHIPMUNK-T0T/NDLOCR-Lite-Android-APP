package com.ndl.mainapp.ocr.model

data class MetricsSnapshot(
    val detectorMs: Long,
    val recognizerMs: Long,
    val droppedFrames: Long,
)
