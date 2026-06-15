package com.example.gemmaimage

import android.util.Log
import com.example.gemmaimage.model.GemmaImageResult
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min

object GemmaResponseParser {
    private const val TAG = "GemmaResponseParser"
    
    fun parse(rawResponse: String, elapsedMs: Long = 0L): GemmaImageResult {
        val extractedJson = extractJson(rawResponse)
        Log.d(TAG, "Extracted JSON: $extractedJson")
        
        val json = JSONObject(extractedJson)
        val label = normalizeLabel(json.optString("subject", "unknown"))
        val confidence = min(1f, max(0f, json.optDouble("confidence", 0.0).toFloat()))
        val keywordArray = json.optJSONArray("keywords")
        val keywords = if (keywordArray == null) {
            emptyList()
        } else {
            List(keywordArray.length()) { index -> keywordArray.optString(index) }
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .distinct()
        }
        val evidence = json.optString("evidence", "")
        
        Log.d(TAG, "Parsed: subject='$label', confidence=$confidence, keywords=${keywords.size}")
        
        return GemmaImageResult(
            label = label,
            confidence = confidence,
            keywords = keywords,
            evidence = evidence,
            rawResponse = rawResponse,
            elapsedMs = elapsedMs
        )
    }

    private fun extractJson(raw: String): String {
        // Try to extract JSON from markdown code fence (```json ... ```)
        // Note: backticks are escaped for Android's regex engine
        val fencedPattern = Regex("\\`\\`\\`(?:json)?\\s*(\\{.*?\\})\\s*\\`\\`\\`", RegexOption.DOT_MATCHES_ALL)
        val fenced = fencedPattern.find(raw)?.groupValues?.getOrNull(1)
        if (fenced != null) {
            Log.d(TAG, "Found JSON in code fence")
            return fenced
        }

        // Fallback: find the first { and last }
        val start = raw.indexOf('{')
        val end = raw.lastIndexOf('}')
        if (start >= 0 && end > start) {
            Log.d(TAG, "Extracted JSON from raw text (start=$start, end=$end)")
            return raw.substring(start, end + 1)
        }

        Log.e(TAG, "Failed to extract JSON from response: ${raw.take(200)}")
        throw IllegalArgumentException("Gemma response did not contain a JSON object.")
    }

    private fun normalizeLabel(value: String): String {
        val lower = value.trim().lowercase()
        return when (lower) {
            "math", "mathematics"          -> "maths"
            "chem"                         -> "chemistry"
            "bio"                          -> "biology"
            "phy", "phys"                  -> "physics"
            "social_sciences",
            "social sciences",
            "social science"               -> "social_studies"
            "business"                     -> "business_studies"
            "civic", "civics_studies"      -> "civics"
            "geo"                          -> "geography"
            "hist"                         -> "history"
            "accounting_studies"           -> "accounting"
            else                           -> lower
        }
    }
}