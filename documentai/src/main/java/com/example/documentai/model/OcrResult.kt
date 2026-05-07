package com.example.documentai.model

/** A single recognized line of text with its bounding box. */
data class TextLine(val text: String, val box: OcrBox)

/**
 * Full OCR extraction result.
 * @param text All recognized lines joined by newline.
 * @param lines Per-line results with bounding boxes.
 */
data class OcrResult(
    val text: String,
    val lines: List<TextLine>
)
