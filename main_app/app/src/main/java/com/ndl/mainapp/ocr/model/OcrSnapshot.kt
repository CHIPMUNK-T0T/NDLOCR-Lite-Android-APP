package com.ndl.mainapp.ocr.model

data class OcrSnapshot(
    val frameId: Long,
    val frameWidth: Int,
    val frameHeight: Int,
    val timestampMs: Long,
    val rois: List<RoiBox>,
    val lines: List<OcrLine>,
    val metrics: MetricsSnapshot,
)
