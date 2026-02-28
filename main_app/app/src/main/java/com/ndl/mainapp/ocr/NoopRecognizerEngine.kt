package com.ndl.mainapp.ocr

import com.ndl.mainapp.ocr.model.FramePacket
import com.ndl.mainapp.ocr.model.OcrLine
import com.ndl.mainapp.ocr.model.RoiBox

class NoopRecognizerEngine : RecognizerEngine {
    override suspend fun recognize(frame: FramePacket, rois: List<RoiBox>): List<OcrLine> {
        return rois.map { OcrLine(it, "") }
    }
}
