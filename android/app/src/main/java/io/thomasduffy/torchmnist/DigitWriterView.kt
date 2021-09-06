package io.thomasduffy.torchmnist

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import androidx.core.content.res.ResourcesCompat

private const val STROKE_WIDTH = 12f

class DigitWriterView(ctx: Context, attrs: AttributeSet): View(ctx, attrs) {

    private lateinit var extraCanvas: Canvas
    private lateinit var extraBitmap: Bitmap

    private var motionTouchEventX = 0f
    private var motionTouchEventY = 0f

    private var currentX = 0f
    private var currentY = 0f

    private val touchTolerance = ViewConfiguration.get(ctx).scaledTouchSlop

    private val bgColor = ResourcesCompat.getColor(resources, R.color.purple_200, null)
    private val drawColor = ResourcesCompat.getColor(resources, R.color.black, null)

    private var allPoints: MutableList<MutableList<Pair<Float, Float>>> = ArrayList()
    private var pointSegment: MutableList<Pair<Float, Float>> = ArrayList()

    private val paint = Paint().apply {
        color = drawColor
        // Smooth edges
        isAntiAlias = true
        // downsampling
        isDither = true
        style = Paint.Style.STROKE // default: FILL
        strokeJoin = Paint.Join.ROUND // default: MITER
        strokeCap = Paint.Cap.ROUND // default: Butt
        strokeWidth = STROKE_WIDTH
    }

    private var path = Path()

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (::extraBitmap.isInitialized) extraBitmap.recycle()
        extraBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        extraCanvas = Canvas(extraBitmap)
        extraCanvas.drawColor(bgColor)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawBitmap(extraBitmap, 0f, 0f, null)
    }

    // finger presses down
    private fun touchStart() {
        path.reset()
        path.moveTo(motionTouchEventX, motionTouchEventY)
        currentX = motionTouchEventX
        currentY = motionTouchEventY
    }

    // finger moves
    private fun touchMove() {
        val dx = Math.abs(motionTouchEventX - currentX)
        val dy = Math.abs(motionTouchEventY - currentY)
        if (dx >= touchTolerance || dy >= touchTolerance) {
            // QuadTo() adds a quadratic bezier from the last point,
            // approaching control point (x1,y1), and ending at (x2,y2).
            path.quadTo(currentX, currentY, (motionTouchEventX + currentX) / 2, (motionTouchEventY + currentY) / 2)
            currentX = motionTouchEventX
            currentY = motionTouchEventY

            // add point to current segment being drawn
            pointSegment.add(Pair(currentX, currentY))

            extraCanvas.drawPath(path, paint)
        }
        invalidate()
    }

    // finger comes up
    private fun touchUp() {
        // add the drawn segment to the set of drawn segments (allPoints)
        allPoints.add(pointSegment)
        pointSegment.clear()
        path.reset()
    }

    fun getPoints(): MutableList<MutableList<Pair<Float, Float>>> {
        return allPoints
    }

    fun clear() {
        allPoints.clear()
        extraCanvas.drawColor(bgColor)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        motionTouchEventX = event.x
        motionTouchEventY = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> touchStart()
            MotionEvent.ACTION_MOVE -> touchMove()
            MotionEvent.ACTION_UP -> touchUp()
        }

        return true
    }

}