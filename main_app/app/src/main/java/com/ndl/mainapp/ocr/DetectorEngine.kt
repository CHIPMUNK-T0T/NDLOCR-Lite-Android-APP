package com.ndl.mainapp.ocr

import com.ndl.mainapp.ocr.model.FramePacket
import com.ndl.mainapp.ocr.model.RoiBox

interface DetectorEngine {
    suspend fun detect(frame: FramePacket): List<RoiBox>
}
