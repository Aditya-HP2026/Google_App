package com.example.documentai

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.documentai.model.ClassificationResult
import java.nio.LongBuffer
import kotlin.math.exp

/**
 * Wraps `distilbert/model.onnx` (DistilBERT sequence classifier).
 *
 * Architecture: DistilBertForSequenceClassification (4 classes).
 *   Input  "input_ids" [1, 512]  – int64 token IDs
 *   Output "logits"    [1, 4]   – raw logits → softmax → label
 *
 * Labels (from config.json id2label):
 *   0 → biology   1 → chemistry   2 → maths   3 → physics
 */
internal class DistilClassifier(context: Context) : AutoCloseable {

    private val env       = OrtEnvironment.getEnvironment()
    private val session   = OrtSession.SessionOptions().use { opts ->
        env.createSession(AssetUtils.getModelFile(context, "distilbert/model.onnx"), opts)
    }
    private val tokenizer = Tokenizer(context)

    // Ordered list matches config.json id2label
    private val labels = listOf("biology", "chemistry", "maths", "physics")

    // ── Public API ─────────────────────────────────────────────────────────

    fun classify(text: String): ClassificationResult {
        val inputIds = tokenizer.encode(text)

        val buf    = LongBuffer.wrap(inputIds)
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 512))
        val result = session.run(mapOf("input_ids" to tensor))

        @Suppress("UNCHECKED_CAST")
        val rawLogits = ((result.get(0) as OnnxTensor).value as Array<FloatArray>)[0]
        val probs     = softmax(rawLogits)

        result.close()
        tensor.close()

        val bestIdx  = probs.indices.maxByOrNull { probs[it] } ?: 0
        val allScores = labels.zip(probs.toList()).toMap()

        return ClassificationResult(
            label      = labels[bestIdx],
            confidence = probs[bestIdx],
            allScores  = allScores
        )
    }

    // ── Numerically-stable softmax ─────────────────────────────────────────

    private fun softmax(logits: FloatArray): FloatArray {
        val max     = logits.max()
        val expVals = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum     = expVals.sum()
        return FloatArray(expVals.size) { expVals[it] / sum }
    }

    override fun close() { session.close() }
}
