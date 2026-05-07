package com.example.documentai.model

/** Combined result of the full OCR + classification pipeline. */
data class PipelineResult(
    val ocr: OcrResult,
    val classification: ClassificationResult
)
