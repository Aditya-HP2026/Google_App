package com.example.documentai

import android.graphics.Bitmap
import kotlin.math.max

/**
 * Crops detected text regions out of the original bitmap for the recognizer.
 *
 * Each box is a float array [x0,y0, x1,y1, x2,y2, x3,y3] in clockwise order
 * (top-left, top-right, bottom-right, bottom-left).
 */
internal object CropUtils {

    /**
     * Returns one cropped sub-bitmap per detected box.
     * The crop is taken as the axis-aligned bounding rectangle of each box's
     * four corners, clamped to the bitmap dimensions.
     */
    fun cropBoxes(bitmap: Bitmap, boxes: List<FloatArray>): List<Bitmap> =
        boxes.map { box -> cropSingle(bitmap, box) }

    private fun cropSingle(bitmap: Bitmap, box: FloatArray): Bitmap {
        // box = [x0,y0, x1,y1, x2,y2, x3,y3]  (8 values)
        val xs = floatArrayOf(box[0], box[2], box[4], box[6])
        val ys = floatArrayOf(box[1], box[3], box[5], box[7])

        val left   = xs.min().toInt().coerceIn(0, bitmap.width  - 1)
        val top    = ys.min().toInt().coerceIn(0, bitmap.height - 1)
        val right  = xs.max().toInt().coerceIn(0, bitmap.width  - 1)
        val bottom = ys.max().toInt().coerceIn(0, bitmap.height - 1)

        val cropW = max(right  - left, 1)
        val cropH = max(bottom - top,  1)

        return Bitmap.createBitmap(bitmap, left, top, cropW, cropH)
    }
}
