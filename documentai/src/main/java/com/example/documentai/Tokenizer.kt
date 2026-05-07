package com.example.documentai

import android.content.Context
import org.json.JSONObject

/**
 * BERT WordPiece tokenizer backed by the `distilbert/tokenizer.json` asset.
 *
 * Implements the same pipeline as `BertTokenizer` with `do_lower_case = true`:
 *   1. Lower-case + Chinese-char spacing
 *   2. Whitespace / punctuation split
 *   3. WordPiece sub-word tokenization
 *   4. Wrap with [CLS] / [SEP], pad / truncate to 512
 */
internal class Tokenizer(context: Context) {

    private val vocab: Map<String, Int>

    // Special token IDs (BERT standard)
    private val clsId  = 101
    private val sepId  = 102
    private val padId  = 0
    private val unkId  = 100
    private val maxLen = 512

    init {
        val json  = AssetUtils.readAssetText(context, "distilbert/tokenizer.json")
        val model = JSONObject(json).getJSONObject("model")
        val vocabJson = model.getJSONObject("vocab")
        val map = HashMap<String, Int>(vocabJson.length() * 2)
        val keys = vocabJson.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            map[k] = vocabJson.getInt(k)
        }
        vocab = map
    }

    /**
     * Encodes [text] into a padded/truncated LongArray of length 512,
     * ready to feed directly into the DistilBERT model.
     */
    fun encode(text: String): LongArray {
        val tokens = wordpieceTokenize(basicTokenize(normalize(text)))

        val ids = ArrayList<Int>(maxLen)
        ids += clsId
        for (t in tokens) ids += vocab[t] ?: unkId
        ids += sepId

        // Truncate: keep [CLS] … up to 510 word-pieces … [SEP]
        val truncated = if (ids.size > maxLen) {
            val tmp = ids.subList(0, maxLen - 1).toMutableList()
            tmp += sepId
            tmp
        } else ids

        val result = LongArray(maxLen) { padId.toLong() }
        truncated.forEachIndexed { i, v -> result[i] = v.toLong() }
        return result
    }

    // ── Normalization ──────────────────────────────────────────────────────

    private fun normalize(text: String): String {
        val sb = StringBuilder(text.length * 2)
        for (ch in text.lowercase()) {
            when {
                isChineseLike(ch) -> sb.append(' ').append(ch).append(' ')
                isControl(ch)     -> Unit          // drop control chars
                ch.isWhitespace() -> sb.append(' ')
                else              -> sb.append(ch)
            }
        }
        return sb.toString().trim()
    }

    /** CJK Unified Ideographs and common extensions. */
    private fun isChineseLike(c: Char): Boolean {
        val cp = c.code
        return cp in 0x4E00..0x9FFF ||
               cp in 0x3400..0x4DBF ||
               cp in 0xF900..0xFAFF ||
               cp in 0x2F800..0x2FA1F
    }

    private fun isControl(c: Char): Boolean {
        if (c == '\t' || c == '\n' || c == '\r') return false
        return c.category == CharCategory.CONTROL || c.category == CharCategory.FORMAT
    }

    // ── Basic tokenization (whitespace + punctuation split) ────────────────

    private fun basicTokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val current = StringBuilder()
        for (c in text) {
            when {
                c.isWhitespace() -> {
                    if (current.isNotEmpty()) { tokens += current.toString(); current.clear() }
                }
                isPunctuation(c) -> {
                    if (current.isNotEmpty()) { tokens += current.toString(); current.clear() }
                    tokens += c.toString()
                }
                else -> current.append(c)
            }
        }
        if (current.isNotEmpty()) tokens += current.toString()
        return tokens
    }

    private fun isPunctuation(c: Char): Boolean {
        val cp = c.code
        if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) return true
        return c.category in PUNCT_CATEGORIES
    }

    // ── WordPiece ──────────────────────────────────────────────────────────

    private fun wordpieceTokenize(tokens: List<String>): List<String> {
        val out = mutableListOf<String>()
        for (word in tokens) out += wpSingle(word)
        return out
    }

    private fun wpSingle(word: String): List<String> {
        if (word.length > 100) return listOf("[UNK]")
        val result = mutableListOf<String>()
        var start = 0
        while (start < word.length) {
            var end   = word.length
            var found: String? = null
            while (start < end) {
                val sub = if (start == 0) word.substring(start, end)
                          else "##${word.substring(start, end)}"
                if (vocab.containsKey(sub)) { found = sub; break }
                end--
            }
            if (found == null) return listOf("[UNK]")
            result += found
            start = end
        }
        return result
    }

    companion object {
        private val PUNCT_CATEGORIES = setOf(
            CharCategory.OTHER_PUNCTUATION,
            CharCategory.CONNECTOR_PUNCTUATION,
            CharCategory.DASH_PUNCTUATION,
            CharCategory.END_PUNCTUATION,
            CharCategory.FINAL_QUOTE_PUNCTUATION,
            CharCategory.INITIAL_QUOTE_PUNCTUATION,
            CharCategory.START_PUNCTUATION
        )
    }
}
