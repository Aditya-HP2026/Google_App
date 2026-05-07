package com.example.semanticsearch

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * Encodes text into a normalized L2 embedding using the MiniLM INT8 ONNX model.
 *
 * Model: all-MiniLM-L6-v2 (INT8, 22 MB)
 * Inputs:  input_ids [1, 128], attention_mask [1, 128], token_type_ids [1, 128]
 * Outputs: embeddings [1, 384] — mean-pooled sentence embedding
 */
internal class MiniLMEncoder(context: Context) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer = MiniLMTokenizer(context)

    init {
        // Copy from assets to cache so OrtSession can open it as a file path
        val cacheFile = File(context.cacheDir, "minilm_int8.onnx")
        if (!cacheFile.exists()) {
            context.assets.open("semantic_search/minilm/minilm_int8.onnx").use { src ->
                cacheFile.outputStream().use { dst -> src.copyTo(dst) }
            }
        }
        session = OrtSession.SessionOptions().use { opts ->
            env.createSession(cacheFile.absolutePath, opts)
        }
    }

    /**
     * Encodes [text] into an L2-normalized float array of length 384.
     */
    fun encode(text: String): FloatArray {
        val enc = tokenizer.encode(text)
        val seqLen = tokenizer.maxLen.toLong()

        val idsTensor   = OnnxTensor.createTensor(env, LongBuffer.wrap(enc.inputIds),      longArrayOf(1, seqLen))
        val maskTensor  = OnnxTensor.createTensor(env, LongBuffer.wrap(enc.attentionMask), longArrayOf(1, seqLen))
        val ttidTensor  = OnnxTensor.createTensor(env, LongBuffer.wrap(enc.tokenTypeIds),  longArrayOf(1, seqLen))

        val result = session.run(mapOf(
            "input_ids"      to idsTensor,
            "attention_mask" to maskTensor,
            "token_type_ids" to ttidTensor
        ))

        @Suppress("UNCHECKED_CAST")
        val raw = ((result.get(0) as OnnxTensor).value as Array<FloatArray>)[0]

        result.close()
        idsTensor.close(); maskTensor.close(); ttidTensor.close()

        return l2Normalize(raw)
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        if (norm < 1e-9f) return v
        return FloatArray(v.size) { v[it] / norm }
    }

    override fun close() = session.close()
}
