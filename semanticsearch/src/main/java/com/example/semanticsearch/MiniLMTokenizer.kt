package com.example.semanticsearch

import android.content.Context
import org.json.JSONObject

/**
 * BERT WordPiece tokenizer backed by `semantic_search/minilm/tokenizer.json`.
 *
 * Produces input_ids + attention_mask + token_type_ids (all zeros) of fixed
 * length [maxLen], ready to feed into the MiniLM ONNX model.
 */
internal class MiniLMTokenizer(context: Context) {

    private val vocab: Map<String, Int>
    private val clsId  = 101
    private val sepId  = 102
    private val padId  = 0
    private val unkId  = 100
    val maxLen         = 128    // MiniLM was trained with 256 max, 128 is enough for queries

    init {
        val json = context.assets.open("semantic_search/minilm/tokenizer.json")
            .bufferedReader().readText()
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

    data class Encoding(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds: LongArray
    )

    fun encode(text: String): Encoding {
        val tokens = wordpiece(basicTokenize(normalize(text)))

        val ids = ArrayList<Int>(maxLen)
        ids += clsId
        for (t in tokens) ids += vocab[t] ?: unkId
        ids += sepId

        val truncated = if (ids.size > maxLen) {
            val tmp = ids.subList(0, maxLen - 1).toMutableList()
            tmp += sepId
            tmp
        } else ids

        val inputIds       = LongArray(maxLen) { padId.toLong() }
        val attentionMask  = LongArray(maxLen) { 0L }
        val tokenTypeIds   = LongArray(maxLen) { 0L }

        truncated.forEachIndexed { i, v ->
            inputIds[i]      = v.toLong()
            attentionMask[i] = 1L
        }
        return Encoding(inputIds, attentionMask, tokenTypeIds)
    }

    private fun normalize(text: String): String {
        val sb = StringBuilder(text.length * 2)
        for (ch in text.lowercase()) {
            when {
                isChineseLike(ch) -> sb.append(' ').append(ch).append(' ')
                isControl(ch)     -> Unit
                ch.isWhitespace() -> sb.append(' ')
                else              -> sb.append(ch)
            }
        }
        return sb.toString().trim()
    }

    private fun isChineseLike(c: Char): Boolean {
        val cp = c.code
        return cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF ||
               cp in 0xF900..0xFAFF || cp in 0x2F800..0x2FA1F
    }

    private fun isControl(c: Char): Boolean {
        if (c == '\t' || c == '\n' || c == '\r') return false
        return c.category == CharCategory.CONTROL || c.category == CharCategory.FORMAT
    }

    private fun basicTokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val cur = StringBuilder()
        for (c in text) {
            when {
                c.isWhitespace() -> {
                    if (cur.isNotEmpty()) { tokens += cur.toString(); cur.clear() }
                }
                isPunct(c) -> {
                    if (cur.isNotEmpty()) { tokens += cur.toString(); cur.clear() }
                    tokens += c.toString()
                }
                else -> cur.append(c)
            }
        }
        if (cur.isNotEmpty()) tokens += cur.toString()
        return tokens
    }

    private fun isPunct(c: Char): Boolean {
        val cp = c.code
        if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) return true
        return c.category in PUNCT_CATS
    }

    private fun wordpiece(tokens: List<String>): List<String> {
        val out = mutableListOf<String>()
        for (w in tokens) out += wpSingle(w)
        return out
    }

    private fun wpSingle(word: String): List<String> {
        if (word.length > 100) return listOf("[UNK]")
        val result = mutableListOf<String>()
        var start = 0
        while (start < word.length) {
            var end = word.length
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
        private val PUNCT_CATS = setOf(
            CharCategory.OTHER_PUNCTUATION, CharCategory.CONNECTOR_PUNCTUATION,
            CharCategory.DASH_PUNCTUATION,  CharCategory.END_PUNCTUATION,
            CharCategory.FINAL_QUOTE_PUNCTUATION, CharCategory.INITIAL_QUOTE_PUNCTUATION,
            CharCategory.START_PUNCTUATION
        )
    }
}
