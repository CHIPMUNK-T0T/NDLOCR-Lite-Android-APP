package com.ndl.mainapp.ocr

import com.ndl.mainapp.ocr.model.RoiBox
import java.util.concurrent.atomic.AtomicReference

data class RoiSnapshot(
    val rois: List<RoiBox>,
    val updatedAtMs: Long,
)

class RoiState {
    private val state = AtomicReference(RoiSnapshot(emptyList(), 0L))

    fun update(rois: List<RoiBox>, timestampMs: Long) {
        state.set(RoiSnapshot(rois, timestampMs))
    }

    fun get(): RoiSnapshot = state.get()
}
