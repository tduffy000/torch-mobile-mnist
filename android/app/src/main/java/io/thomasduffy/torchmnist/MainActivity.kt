package io.thomasduffy.torchmnist

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.Tensor
import java.io.File
import java.util.*
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    val MNIST_MEAN = 0.3081
    val MNIST_STD = 0.1307
    val BLANK = - MNIST_STD / MNIST_MEAN
    val FILLED = (1.0f - MNIST_STD) / MNIST_MEAN
    val IMG_SIZE = 28
//
//    val path = "file:///android_asset/mobile_model.pt"
//    val mModule = Module.load(path)

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
//                val tensor = convertToTensor(points, drawView.height, drawView.width)
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

    private fun convertToTensor(points: MutableList<MutableList<Pair<Float, Float>>>, h: Int, w: Int): Tensor {
        val inputs = Array<Double>(IMG_SIZE * IMG_SIZE) {_ -> BLANK}
        points.forEach {segment ->
            segment.forEach {pair ->
                val pX = pair.first.toInt()
                val pY = pair.second.toInt()
                if (pX < w && pY < h && pX > 0 && pY > 0) {
                    val x = IMG_SIZE * pX / w
                    val y = IMG_SIZE * pY / h
                    val loc = y * IMG_SIZE + x
                    inputs[loc] = FILLED
                }
            }
        }

        // convert to Tensor
        val buf = Tensor.allocateFloatBuffer(IMG_SIZE * IMG_SIZE)
        inputs.forEach { el -> buf.put(el.toFloat()) }
        return Tensor.fromBlob(buf, longArrayOf(1, 1, 28, 28))
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