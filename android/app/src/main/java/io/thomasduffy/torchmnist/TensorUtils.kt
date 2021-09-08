package io.thomasduffy.torchmnist

import android.util.Log
import org.pytorch.Tensor

class TensorUtils {
    companion object Converter {
        val MNIST_MEAN = 0.3081f
        val MNIST_STD = 0.1307f
        val BLANK = - MNIST_STD / MNIST_MEAN
        val FILLED = (1.0f - MNIST_STD) / MNIST_MEAN
        val IMG_SIZE = 28

        fun convertToTensor(points: MutableList<MutableList<Pair<Float, Float>>>, h: Int, w: Int): Tensor {
            val inputs = Array<Float>(IMG_SIZE * IMG_SIZE) {_ -> BLANK}
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
    }
}