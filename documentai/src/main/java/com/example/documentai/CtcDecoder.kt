package com.example.documentai

import android.content.Context

/**
 * CTC greedy decoder backed by the character dictionary inside
 * `paddleocr/rec_inference.yml`.
 *
 * Token mapping (PaddleOCR convention):
 *   class 0          → CTC blank (discarded)
 *   class i (i ≥ 1)  → charDict[i − 1]
 *
 * The YAML character_dict contains 18 384 entries, matching the
 * 18 385 output classes (18 384 chars + 1 blank).
 */
internal class CtcDecoder(context: Context) {

    private val charDict: List<String>

    init {
        val lines  = AssetUtils.readAssetLines(context, "paddleocr/rec_inference.yml")
        val chars  = ArrayList<String>(18384)
        var inDict = false

        for (line in lines) {
            val trimmed = line.trimStart()
            when {
                // Detect the start of the character_dict block
                !inDict && trimmed.startsWith("character_dict:") -> inDict = true

                inDict && trimmed.startsWith("- ") ->
                    // Each list item: "- <char>"  (may be CJK, ASCII, emoji, …)
                    chars += trimmed.removePrefix("- ")

                // Any non-indented non-list key ends the block (won't normally happen
                // since character_dict is the last section in this file)
                inDict && trimmed.isNotEmpty()
                      && !trimmed.startsWith("-")
                      && !trimmed.startsWith(" ") -> inDict = false
            }
        }
        charDict = chars
    }

    /**
     * Decodes a [T × numClasses] logit matrix to a text string via greedy CTC.
     */
    fun decode(logits: Array<FloatArray>): String {
        // 1. Greedy argmax per time step
        val indices = IntArray(logits.size) { t ->
            var bestIdx = 0; var bestVal = Float.NEGATIVE_INFINITY
            for (c in logits[t].indices) if (logits[t][c] > bestVal) { bestVal = logits[t][c]; bestIdx = c }
            bestIdx
        }

        // 2. Collapse repeated indices, remove blank (0)
        val sb  = StringBuilder()
        var prev = -1
        for (idx in indices) {
            if (idx != prev && idx != 0) {
                val charIdx = idx - 1          // 0-based into charDict
                if (charIdx in charDict.indices) sb.append(charDict[charIdx])
            }
            prev = idx
        }
        return sb.toString()
    }
}
