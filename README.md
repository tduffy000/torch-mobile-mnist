# PyTorch Mobile `mnist`
A simple (and not super functional) example of writing an Application for Android to perform 
hand-written digit prediction using the very popular MNIST dataset.

<img title="screenshot" src="./images/3_prediction.png" height=350> 

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is like the "Hello World" of Deep Learning applications. It contains hand-written single digit characters
and thus provides a reasonable, but not overly difficult, prediction task. The more difficult part (at least for me) is then porting such a model over to an Android device and providing the features to the model. That was the primary goal of this exercise.

## `model`
This uses a rather simple, shallow, network with a frontend of 2-dimensional Convolutional layers followed by some fully connected layers. The final trained model (version `1631154078`) used a learning rate of 0.001 over 25 epochs and achieves a test accuracy of ~98%.
* `model`
    * [`artifacts/`](./model/artifacts) Frozen PyTorch models, both large and quantized for model.
    * [`notebooks/`](./model/notebooks) Jupyter notebooks with testing and sanity checks.
    * [`model.py`](./model/model.py) The model extending `torch.nn.Module` used to predict digits.
    * [`train.py`](./model/train.py) The training script with various hyperparameter arguments.

## `android`
This is the real meat & potatoes, given I've never done this. I followed along with [various](https://github.com/pytorch/android-demo-app/tree/master/ViT4MNIST) [examples](https://github.com/pytorch/android-demo-app/tree/master/PyTorchDemoApp) in the PyTorch Examples [android-demo-app repo](https://github.com/pytorch/android-demo-app).

### Overview
* [`android/app/src/main`](./android/app/src/main/)
    * [`/assets/mobile_model.pt`](./android/app/src/main/assets/) The frozen PyTorch model to deploy.
    * [`/res`](./android/app/src/main/res/) The set of XML files specifying the objects in the Android app.
    * [`/java/io/thomasduffy/torchmnist`](./android/app/src/main/java/io/thomasduffy/torchmnist/)
        * [`/MainActivity.kt`](./android/app/src/main/java/io/thomasduffy/torchmnist/MainActivity.kt) The main entrypoint to the Android app.
        * [`/DigitWriterView.kt`](./android/app/src/main/java/io/thomasduffy/torchmnist/DigitWriterView.kt) The custom `View` to represent the drawing cell.
        * [`/FileHandler.kt`](./android/app/src/main/java/io/thomasduffy/torchmnist/FileHandler.kt) The class responsible for loading the PyTorch model onto the proper `assets` path.
        * [`/PredictThread.kt`](./android/app/src/main/java/io/thomasduffy/torchmnist/PredictThread.kt) The `Thread` extending class responsible for responding to clicks on the `PREDICT` button.
        * [`/TensorUtils.kt`](./android/app/src/main/java/io/thomasduffy/torchmnist/TensorUtils.kt) The class responsible for turning the `points` representation from the drawing board to an input `Tensor`.

### Drawing

In order to convert the drawn picture on the face of the screen, we need to provide a `Canvas` and a `Bitmap` in a `View`. To store the pixels the user draws on, we extend the `android.view.View` class and override the `onSizeChanged` (which is called on instantiation) and `onDraw` (which is called when it's rendered), 

```kotlin
class DigitWriterView(ctx: Context, attrs: AttributeSet): View(ctx, attrs) {

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (::extraBitmap.isInitialized) extraBitmap.recycle()
        extraBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        extraCanvas = Canvas(extraBitmap)
        extraCanvas.drawColor(bgColor)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawBitmap(extraBitmap, 0f, 0f, null)
    }

}
```

the user drawing moves are stored by overwriting the `onTouchEvent` method,
```kotlin
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
```
The entirety of that code is contained in the [`DigitWriterView.kt`](./android/app/src/main/java/io/thomasduffy/torchmnist/DigitWriterView.kt) class.

Once our custom View works, we can add it to our `XML` file to specify its formatting on the screen
```xml
<io.thomasduffy.torchmnist.DigitWriterView
    android:id="@+id/digitWriterView"
    android:layout_width="350dp"
    android:layout_height="0dp"
    android:layout_marginTop="191dp"
    android:layout_marginBottom="28dp"
    app:layout_constraintBottom_toTopOf="@+id/prediction"
    app:layout_constraintLeft_toLeftOf="parent"
    app:layout_constraintRight_toRightOf="parent"
    app:layout_constraintTop_toTopOf="parent" />
```
and then it's accessible in our `MainActivity.kt` which is the main entrypoint for the app,
```kotlin
class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        val path = FileHandler.assetFilePath(this,"mobile_model.pt")
        val mModule = Module.load(path)

        val predictionView = findViewById<TextView>(R.id.prediction)
        val drawView = findViewById<DigitWriterView>(R.id.digitWriterView)
        val predictButton = findViewById<Button>(R.id.predictButton)
    }
}
```

### Model
Our frozen PyTorch model gets on the device via a little trickery. This seems to me the thing PyTorch still needs to figure out a bit. We have to place it on the Android filesystem in the `assets`. In order to do that, on startup we have to copy the file into that location, which is where our `FileHander` comes in,
```kotlin
class FileHandler {

    companion object FilePath {

        // determine the size of this buffer
        // based on the size of your serialized PyTorch model!
        final private val MODEL_BYTE_SIZE = 8 * 1024

        /**
         * Get the file path on the device to the serialized model.
         */
        fun assetFilePath(ctx: Context, assetName: String): String? {
            val f = File(ctx.filesDir, assetName)

            if (f.exists() && f.length() > 0) return f.absolutePath
            return try {
                val inStream = ctx.assets.open(assetName)
                val outStream = FileOutputStream(f)

                val buf = ByteArray(MODEL_BYTE_SIZE)
                var read = 0
                while (read != -1) {
                    read = inStream.read(buf)
                    outStream.write(buf, 0, read)
                }
                outStream.flush()
                f.absolutePath
            } catch (e: IOException) {
                null
            } finally {
                null
            }
        }
    }

}
```

This will allow us to load in the model Module in our `MainActivity.kt` and use it to make predictions,
```kotlin

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        val path = FileHandler.assetFilePath(this,"mobile_model.pt")
        val mModule = Module.load(path)
    }
}
```

### Resources
- [Kotlin Training](https://developer.android.com/codelabs/advanced-android-kotlin-training-canvas?hl=en&continue=https%3A%2F%2Fcodelabs.developers.google.com%2F%3Fcat%3Dandroid#0)
- [Getting Started with constraintLayout](https://www.raywenderlich.com/9193-constraintlayout-tutorial-for-android-getting-started)
- [Google Constraint Layout](https://developer.android.com/training/constraint-layout)
- [PyTorch 1.9.0 Object Detection Example](https://github.com/pytorch/android-demo-app/blob/master/ObjectDetection/app/src/main/java/org/pytorch/demo/objectdetection/MainActivity.java)