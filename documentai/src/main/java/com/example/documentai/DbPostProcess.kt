package com.example.documentai

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * DB (Differentiable Binarization) post-processor.
 *
 * Converts the raw probability map from the detector into axis-aligned bounding
 * boxes scaled back to original-image coordinates.
 *
 * Parameters match det_inference.yml:
 *   thresh       = 0.3   – binarisation threshold
 *   boxThresh    = 0.6   – min mean probability to keep a box
 *   maxCandidates= 1000
 *   unclipRatio  = 1.5   – polygon expansion factor
 */
internal class DbPostProcess(
    private val thresh:        Float = 0.3f,
    private val boxThresh:     Float = 0.6f,
    private val maxCandidates: Int   = 1000,
    private val unclipRatio:   Float = 1.5f
) {
    /**
     * @param probMap  Flattened [mapH × mapW] float probability map.
     * @param scaleH   origH / mapH  (to convert map → original coords)
     * @param scaleW   origW / mapW
     * @return List of boxes, each as [x0,y0, x1,y1, x2,y2, x3,y3] clockwise.
     */
    fun process(
        probMap: FloatArray,
        mapH:    Int,
        mapW:    Int,
        scaleH:  Float,
        scaleW:  Float
    ): List<FloatArray> {

        // ── 1. Binarise ──────────────────────────────────────────────────
        val binary = BooleanArray(probMap.size) { probMap[it] > thresh }

        // ── 2. Connected-component labelling (iterative flood-fill) ──────
        val labels          = IntArray(binary.size) { -1 }
        // label → list of flat pixel indices belonging to that component
        val components      = ArrayList<IntArray>(256)
        var numLabels        = 0

        outer@ for (y in 0 until mapH) {
            for (x in 0 until mapW) {
                val idx = y * mapW + x
                if (!binary[idx] || labels[idx] >= 0) continue
                if (numLabels >= maxCandidates) break@outer

                val collected = ArrayList<Int>(64)
                floodFill(binary, labels, mapH, mapW, y, x, numLabels, collected)
                components += collected.toIntArray()
                numLabels++
            }
        }

        // ── 3. Build boxes ───────────────────────────────────────────────
        val boxes = ArrayList<FloatArray>(numLabels)

        for (label in 0 until numLabels) {
            val pixels = components[label]
            if (pixels.size < 4) continue

            // Score: mean probability inside the component
            var scoreSum = 0f
            var minX = mapW; var maxX = 0
            var minY = mapH; var maxY = 0
            for (pidx in pixels) {
                scoreSum += probMap[pidx]
                val px = pidx % mapW
                val py = pidx / mapW
                if (px < minX) minX = px
                if (px > maxX) maxX = px
                if (py < minY) minY = py
                if (py > maxY) maxY = py
            }
            val score = scoreSum / pixels.size
            if (score < boxThresh) continue

            // Unclip: expand bounding rect by unclipRatio
            val bw   = (maxX - minX).toFloat()
            val bh   = (maxY - minY).toFloat()
            val area = bw * bh
            val peri = 2f * (bw + bh)
            if (peri == 0f) continue
            val dist = area * unclipRatio / peri

            val x0 = max(0f,          minX - dist)
            val y0 = max(0f,          minY - dist)
            val x1 = min(mapW - 1f,   maxX + dist)
            val y1 = min(mapH - 1f,   maxY + dist)

            // Scale back to original coordinates
            boxes += floatArrayOf(
                x0 * scaleW, y0 * scaleH,   // top-left
                x1 * scaleW, y0 * scaleH,   // top-right
                x1 * scaleW, y1 * scaleH,   // bottom-right
                x0 * scaleW, y1 * scaleH    // bottom-left
            )
        }
        return boxes
    }

    // ── Iterative 4-connected flood-fill ──────────────────────────────────

    private fun floodFill(
        binary:    BooleanArray,
        labels:    IntArray,
        h: Int,    w: Int,
        startY: Int, startX: Int,
        label: Int,
        collected: ArrayList<Int>
    ) {
        val stack = IntArray(h * w)
        var top   = 0
        stack[top++] = startY * w + startX

        while (top > 0) {
            val idx = stack[--top]
            if (labels[idx] >= 0) continue
            labels[idx] = label
            collected   += idx

            val cy = idx / w
            val cx = idx % w
            if (cy > 0   ) { val n = idx - w; if (binary[n] && labels[n] < 0) stack[top++] = n }
            if (cy < h-1 ) { val n = idx + w; if (binary[n] && labels[n] < 0) stack[top++] = n }
            if (cx > 0   ) { val n = idx - 1; if (binary[n] && labels[n] < 0) stack[top++] = n }
            if (cx < w-1 ) { val n = idx + 1; if (binary[n] && labels[n] < 0) stack[top++] = n }
        }
    }
}
