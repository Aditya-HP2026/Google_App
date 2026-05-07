package com.example.gemmaimage.model

data class GemmaImageResult(
    val label: String,
    val confidence: Float,
    val keywords: List<String>,
    val evidence: String,
    val rawResponse: String,
    val elapsedMs: Long
)
