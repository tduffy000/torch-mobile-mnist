package io.thomasduffy.torchmnist

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

//        val predictionView = findViewById<TextView>(R.id.prediction)
        val drawView = findViewById<DigitWriterView>(R.id.digitWriterView)
//        val predictButton = findViewById<Button>(R.id.predictButton)

        val clearButton = findViewById<Button>(R.id.clearButton)
        clearButton.setOnClickListener {
            drawView.clear()
        }

    }
}