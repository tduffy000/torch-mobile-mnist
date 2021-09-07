package io.thomasduffy.torchmnist

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import org.pytorch.Module
import org.pytorch.Tensor

class MainActivity : AppCompatActivity() {

    val path = FileHandler.assetFilePath(this,"mobile_model.pt")
    val mModule = Module.load(path)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

//        val predictionView = findViewById<TextView>(R.id.prediction)
        val drawView = findViewById<DigitWriterView>(R.id.digitWriterView)
        val predictButton = findViewById<Button>(R.id.predictButton)

        // need another thread for this?
//        predictButton.setOnClickListener {
//            // make digit prediction
//            val points = drawView.getPoints()
//            if (points.size > 0 ) {
//                val tensor = TensorUtils.convertToTensor(points, drawView.height, drawView.width)
//                val prediction = predict(tensor)
//                // print prediction in text view
//            }
//        }

        val clearButton = findViewById<Button>(R.id.clearButton)
        clearButton.setOnClickListener {
            drawView.clear()
            // clear the prediction text
        }

    }

//    private fun predict(t: Tensor): Int {
//        val outputs = mModule.forward(IValue.from(t)).toTensor().dataAsDoubleArray
//
//        val sum = outputs.reduce { acc, v -> acc + Math.exp(v) }
//        val pos = outputs.map { v -> (Math.exp(v) / sum).toFloat() }
//
//        var pred = -1
//        var maxScore = - Float.MAX_VALUE
//        for (i in 0..outputs.size - 1) {
//            if (pos[i] > maxScore) {
//                maxScore = pos[i]
//                pred = i
//            }
//        }
//
//        return pred
//    }
}