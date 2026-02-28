package com.ndl.mainapp.ocr

import android.graphics.Bitmap
import com.ndl.mainapp.ocr.model.FramePacket
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

class LatestFrameStore {
    private val latest = AtomicReference<FramePacket?>(null)
    private val dropped = AtomicLong(0)

    fun update(frame: FramePacket) {
        val prev = latest.getAndSet(frame)
        if (prev != null) {
            dropped.incrementAndGet()
            if (!prev.bitmap.isRecycled) {
                prev.bitmap.recycle()
            }
        }
    }

    fun getLatest(): FramePacket? = latest.get()

    fun snapshotCopy(): FramePacket? {
        val src = latest.get() ?: return null
        val bmp = if (src.bitmap.config != null) {
            src.bitmap.copy(src.bitmap.config ?: Bitmap.Config.ARGB_8888, false)
        } else {
            src.bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } ?: return null
        return FramePacket(
            id = src.id,
            timestampMs = src.timestampMs,
            width = bmp.width,
            height = bmp.height,
            bitmap = bmp,
        )
    }

    fun droppedFrames(): Long = dropped.get()

    fun clear() {
        latest.getAndSet(null)?.let { old ->
            if (!old.bitmap.isRecycled) {
                old.bitmap.recycle()
            }
        }
    }
}
