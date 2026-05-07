package com.example.documentai

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/**
 * Wraps `paddleocr/rec.onnx` (PP-OCRv5 mobile recognizer).
 *
 * Pre-processing (PaddleOCR v4/v5 standard):
 *   - Resize each crop to height 48, preserve aspect ratio
 *   - Normalize: pixel/255 × 2 − 1  (i.e. mean=0.5, std=0.5 per channel)
 *   - Layout: CHW float32
 *
 * Input  tensor: "x"            [1, 3, 48, W]  (W is dynamic)
 * Output tensor: "fetch_name_0" [1, T, 18385]  — raw CTC logits
 */
internal class OcrRecognizer(context: Context) : AutoCloseable {

    private val env     = OrtEnvironment.getEnvironment()
    private val session = OrtSession.SessionOptions().use { opts ->
        env.createSession(AssetUtils.getModelFile(context, "paddleocr/rec.onnx"), opts)
    }

    private val targetH = 48

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Runs the recognizer on each crop.
     * @return List of [T × numClasses] logit matrices (one per crop).
     */
    fun recognize(crops: List<Bitmap>): List<Array<FloatArray>> =
        crops.map { recognizeSingle(it) }

    // ── Single image ───────────────────────────────────────────────────────

    private fun recognizeSingle(bitmap: Bitmap): Array<FloatArray> {
        val aspect  = bitmap.width.toFloat() / bitmap.height.toFloat()
        val targetW = max((targetH * aspect).toInt(), 1)
        val resized = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)

        val inputTensor = bitmapToTensor(resized, targetH, targetW)
        val result      = session.run(mapOf("x" to inputTensor))

        // Output shape [1, T, numClasses] → Array<Array<FloatArray>>
        @Suppress("UNCHECKED_CAST")
        val logits = (result.get(0) as OnnxTensor).value as Array<Array<FloatArray>>
        val frames = logits[0]   // [T, numClasses]

        result.close()
        inputTensor.close()
        return frames
    }

    // ── Pre-processing ─────────────────────────────────────────────────────

    private fun bitmapToTensor(bitmap: Bitmap, h: Int, w: Int): OnnxTensor {
        val pixels = IntArray(h * w)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val buf = ByteBuffer
            .allocateDirect(3 * h * w * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        // CHW: write R plane, then G, then B
        // Normalise: v/255 * 2 - 1  ≡  (v/255 - 0.5) / 0.5
        for (c in 0..2) {
            for (px in pixels) {
                val v = when (c) {
                    0    -> (px shr 16) and 0xFF
                    1    -> (px shr  8) and 0xFF
                    else ->  px         and 0xFF
                }
                buf.put(v / 255f * 2f - 1f)
            }
        }
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, h.toLong(), w.toLong()))
    }

    override fun close() { session.close() }
}
