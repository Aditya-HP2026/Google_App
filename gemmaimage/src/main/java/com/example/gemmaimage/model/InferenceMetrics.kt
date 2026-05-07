package com.example.gemmaimage.model

/**
 * Metrics captured per inference call — for proving GPU usage and tracking performance.
 */
data class InferenceMetrics(
    /** Which backend was used for this call. */
    val backend: Accelerator,
    /** Time from start of sendMessage to first token / response. */
    val endToEndMs: Long,
    /** Size of the input (prompt chars + image bytes if applicable). */
    val inputSize: Int,
    /** Length of the output response string. */
    val outputLength: Int,
    /** Whether vision (image) was involved. */
    val isVisionCall: Boolean,
    /** Run number (sequential counter). */
    val runNumber: Int
)
