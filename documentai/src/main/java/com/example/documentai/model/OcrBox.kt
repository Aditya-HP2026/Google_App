package com.example.documentai.model

/**
 * A single detected text region as four corner points [x, y] in clockwise order.
 * Coordinates are in the **original** bitmap's pixel space.
 */
data class OcrBox(
    /** 4 × 2 array: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] clockwise from top-left */
    val points: Array<FloatArray>
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is OcrBox) return false
        return points.contentDeepEquals(other.points)
    }
    override fun hashCode(): Int = points.contentDeepHashCode()
}
