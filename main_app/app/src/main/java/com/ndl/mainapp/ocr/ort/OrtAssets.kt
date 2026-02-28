package com.ndl.mainapp.ocr.ort

import android.content.Context
import java.nio.charset.StandardCharsets

internal object OrtAssets {
    fun readAssetBytes(context: Context, assetPath: String): ByteArray {
        context.assets.open(assetPath).use { input ->
            return input.readBytes()
        }
    }

    fun readCharset(context: Context, assetPath: String): List<String> {
        val raw = context.assets.open(assetPath).use { input ->
            String(input.readBytes(), StandardCharsets.UTF_8)
        }
        return raw.map { it.toString() }
    }
}
