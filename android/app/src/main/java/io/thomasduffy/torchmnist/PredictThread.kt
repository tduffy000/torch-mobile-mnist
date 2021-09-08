package io.thomasduffy.torchmnist

import android.util.Log
import android.widget.TextView
import org.pytorch.Tensor
import org.pytorch.IValue
import org.pytorch.Module

class PredictThread(
    val mModule: Module,
    val drawView: DigitWriterView,
    val predView: TextView,
): Thread() {

    override fun run() {
        var prediction = -1
        val points = drawView.getPoints()
        Log.d(javaClass.canonicalName, "points.size = " + points.size)
        if (points.size > 0) {
            val tensor = TensorUtils.convertToTensor(points, drawView.height, drawView.width)
            prediction = predict(tensor)
            Log.d(javaClass.canonicalName, "pred = " + prediction)
        }
        predView.text = prediction.toString()
    }

    private fun predict(t: Tensor): Int {
        val outputs = mModule.forward(IValue.from(t)).toTensor().dataAsFloatArray

        val sum = outputs.reduce { acc, v -> acc + kotlin.math.exp(v) }
        val pos = outputs.map { v -> kotlin.math.exp(v) / sum }

        var pred = -1
        var maxScore = - Float.MAX_VALUE
        for (i in 0..outputs.size - 1) {
            if (pos[i] > maxScore) {
                maxScore = pos[i]
                pred = i
            }
        }

        return pred
    }
}