package io.thomasduffy.torchmnist

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val digitView = DigitWriterView(this)
        digitView.contentDescription = getString(R.string.canvasContentDescription)

        setContentView(digitView)
    }
}