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
        if (points.size > 0) {
            val tensor = TensorUtils.convertToTensor(points, drawView.height, drawView.width)
            prediction = predict(tensor)
        }
        predView.text = prediction.toString()
    }

    private fun predict(t: Tensor): Int {
        val outputs = mModule.forward(IValue.from(t)).toTensor().dataAsFloatArray

        var pred = -1
        var maxScore = - Float.MAX_VALUE
        for (i in outputs.indices) {
            if (outputs[i] > maxScore) {
                maxScore = outputs[i]
                pred = i
            }
        }

        return pred
    }
}