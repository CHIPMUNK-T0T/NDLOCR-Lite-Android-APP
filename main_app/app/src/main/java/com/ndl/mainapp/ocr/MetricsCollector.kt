package com.ndl.mainapp.ocr

import com.ndl.mainapp.ocr.model.MetricsSnapshot
import java.util.concurrent.atomic.AtomicLong

class MetricsCollector(
    private val frameStore: LatestFrameStore,
) {
    private val detectorMs = AtomicLong(0)
    private val recognizerMs = AtomicLong(0)

    fun updateDetector(ms: Long) {
        detectorMs.set(ms)
    }

    fun updateRecognizer(ms: Long) {
        recognizerMs.set(ms)
    }

    fun snapshot(): MetricsSnapshot {
        return MetricsSnapshot(
            detectorMs = detectorMs.get(),
            recognizerMs = recognizerMs.get(),
            droppedFrames = frameStore.droppedFrames(),
        )
    }
}
