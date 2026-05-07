package com.example.documentai

import android.content.Context
import java.io.File
import java.io.FileOutputStream

/**
 * Copies ONNX model files and supporting assets out of the APK's assets/ folder into
 * the app's cache directory so that ONNX Runtime can open them as normal file paths.
 *
 * The bundle lives at:
 *   assets/document_ai/paddleocr_distil_old_bundle/<relativePath>
 */
internal object AssetUtils {

    private const val BUNDLE = "document_ai/paddleocr_distil_old_bundle"

    /**
     * Returns the absolute path to [relativePath] (relative to the bundle root),
     * copying it to the cache directory on first call.
     *
     * Example: `getModelFile(ctx, "paddleocr/det.onnx")`
     */
    fun getModelFile(context: Context, relativePath: String): String {
        val outFile = File(File(context.cacheDir, "documentai"), relativePath)
        if (outFile.exists()) return outFile.absolutePath
        outFile.parentFile?.mkdirs()
        context.assets.open("$BUNDLE/$relativePath").use { src ->
            FileOutputStream(outFile).use { dst -> src.copyTo(dst) }
        }
        return outFile.absolutePath
    }

    /** Reads the entire text of an asset file inside the bundle. */
    fun readAssetText(context: Context, relativePath: String): String =
        context.assets.open("$BUNDLE/$relativePath").bufferedReader().use { it.readText() }

    /** Reads all lines of an asset file inside the bundle. */
    fun readAssetLines(context: Context, relativePath: String): List<String> =
        context.assets.open("$BUNDLE/$relativePath").bufferedReader().use { it.readLines() }
}
