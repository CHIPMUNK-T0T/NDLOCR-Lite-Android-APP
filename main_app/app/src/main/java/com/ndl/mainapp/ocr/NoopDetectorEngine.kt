package com.ndl.mainapp.ocr

import com.ndl.mainapp.ocr.model.FramePacket
import com.ndl.mainapp.ocr.model.RoiBox

class NoopDetectorEngine : DetectorEngine {
    override suspend fun detect(frame: FramePacket): List<RoiBox> {
        return emptyList()
    }
}
