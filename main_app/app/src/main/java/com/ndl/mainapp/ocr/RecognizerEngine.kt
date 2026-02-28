package com.ndl.mainapp.ocr

import com.ndl.mainapp.ocr.model.FramePacket
import com.ndl.mainapp.ocr.model.OcrLine
import com.ndl.mainapp.ocr.model.RoiBox

interface RecognizerEngine {
    suspend fun recognize(frame: FramePacket, rois: List<RoiBox>): List<OcrLine>
}
