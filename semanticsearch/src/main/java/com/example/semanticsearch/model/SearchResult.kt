package com.example.semanticsearch.model

/**
 * One semantic search result returned by [com.example.semanticsearch.SemanticSearchEngine].
 *
 * @param sourcePath  Original file path stored when the document was indexed.
 * @param pageNum     Page number (0-based), or -1 for non-paged documents.
 * @param subject     Classified subject label (e.g. "chemistry").
 * @param keywords    Keywords extracted at index time.
 * @param textChunk   The text passage that matched.
 * @param score       Cosine similarity in [0, 1].
 */
data class SearchResult(
    val sourcePath: String,
    val pageNum: Int,
    val subject: String,
    val keywords: List<String>,
    val textChunk: String,
    val score: Float
)
