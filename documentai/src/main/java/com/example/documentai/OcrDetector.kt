package com.example.documentai

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Wraps `paddleocr/det.onnx` (PP-OCRv5 mobile detector).
 *
 * Pre-processing (from det_inference.yml):
 *   - Resize longest side → 960, align both dims to multiples of 32
 *   - Normalize: (pixel/255 − mean) / std  with ImageNet stats (BGR order)
 *   - Layout: CHW float32, batch = 1
 *
 * Input  tensor: "x"           [1, 3, H, W]
 * Output tensor: "fetch_name_0" [1, 1, H, W]  — DB probability map
 */
internal class OcrDetector(context: Context) : AutoCloseable {

    private val env     = OrtEnvironment.getEnvironment()
    private val session = OrtSession.SessionOptions().use { opts ->
        env.createSession(AssetUtils.getModelFile(context, "paddleocr/det.onnx"), opts)
    }

    // ImageNet normalisation constants (RGB channel order)
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    data class DetectorOutput(
        /** Flattened probability map, row-major [mapH × mapW] */
        val probMap: FloatArray,
        val mapH:    Int,
        val mapW:    Int,
        /** Scale factors to convert map coords → original bitmap coords */
        val scaleH:  Float,
        val scaleW:  Float
    )

    // ── Public API ─────────────────────────────────────────────────────────

    fun detect(bitmap: Bitmap): DetectorOutput {
        val (resized, scaleH, scaleW) = prepareResize(bitmap)
        val h = resized.height
        val w = resized.width

        val inputTensor = bitmapToTensor(resized, h, w)
        val inputs      = mapOf("x" to inputTensor)
        val result      = session.run(inputs)

        val probMap = extractProbMap(result.get(0) as OnnxTensor, h, w)

        result.close()
        inputTensor.close()

        return DetectorOutput(probMap, h, w, scaleH, scaleW)
    }

    // ── Pre-processing ─────────────────────────────────────────────────────

    /**
     * Resize so the longest side ≤ 960 and both dims are multiples of 32.
     * Returns (resizedBitmap, scaleH, scaleW) where scale = original / resized.
     */
    private fun prepareResize(src: Bitmap): Triple<Bitmap, Float, Float> {
        val maxLong = 960
        val oH = src.height.toFloat()
        val oW = src.width.toFloat()
        val scale = maxLong / maxOf(oH, oW)
        val newH = ((oH * scale + 31) / 32).toInt() * 32
        val newW = ((oW * scale + 31) / 32).toInt() * 32
        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        return Triple(resized, oH / newH, oW / newW)
    }

    private fun bitmapToTensor(bitmap: Bitmap, h: Int, w: Int): OnnxTensor {
        val pixels = IntArray(h * w)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        // Allocate direct buffer: 1 × 3 × H × W floats
        val buf = ByteBuffer
            .allocateDirect(3 * h * w * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        // Write R plane, then G, then B (CHW)
        for (c in 0..2) {
            val m = mean[c]; val s = std[c]
            for (px in pixels) {
                val channelVal = when (c) {
                    0    -> (px shr 16) and 0xFF
                    1    -> (px shr  8) and 0xFF
                    else ->  px         and 0xFF
                }
                buf.put((channelVal / 255f - m) / s)
            }
        }
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, h.toLong(), w.toLong()))
    }

    // ── Post-processing: extract float[][] → FloatArray ────────────────────

    /**
     * ORT returns the output as a nested Java array.
     * Shape [1, 1, H, W] → `float[][][][]`
     * Shape [1, H, W]    → `float[][][]`   (some export variants)
     */
    @Suppress("UNCHECKED_CAST")
    private fun extractProbMap(tensor: OnnxTensor, h: Int, w: Int): FloatArray {
        val raw = tensor.value
        // Navigate past batch (and optional channel) dims until we reach Array<FloatArray>
        var cur: Any = raw
        while (cur is Array<*> && cur[0] !is FloatArray) {
            cur = (cur as Array<*>)[0]!!
        }
        val rows = cur as Array<FloatArray>
        val out  = FloatArray(h * w)
        var idx  = 0
        for (row in rows) for (v in row) out[idx++] = v
        return out
    }

    override fun close() { session.close() }
}
