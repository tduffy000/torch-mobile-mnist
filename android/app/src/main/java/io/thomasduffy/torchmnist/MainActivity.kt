package io.thomasduffy.torchmnist

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import org.pytorch.Module

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        val path = FileHandler.assetFilePath(this,"mobile_model.pt")
        val mModule = Module.load(path)

        val predictionView = findViewById<TextView>(R.id.prediction)
        val drawView = findViewById<DigitWriterView>(R.id.digitWriterView)
        val predictButton = findViewById<Button>(R.id.predictButton)

        val t = PredictThread(mModule, drawView, predictionView)
        predictButton.setOnClickListener {
            t.run()
        }

        val clearButton = findViewById<Button>(R.id.clearButton)
        clearButton.setOnClickListener {
            drawView.clear()
            predictionView.text = ""
        }

    }

}