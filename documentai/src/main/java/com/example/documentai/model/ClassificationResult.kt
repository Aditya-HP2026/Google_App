package com.example.documentai.model

/**
 * Subject classification result.
 * @param label  Predicted label: "biology" | "chemistry" | "maths" | "physics"
 * @param confidence Softmax probability of the winning class [0, 1].
 * @param allScores  Softmax probability for every class.
 */
data class ClassificationResult(
    val label: String,
    val confidence: Float,
    val allScores: Map<String, Float>
)
