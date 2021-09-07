package io.thomasduffy.torchmnist

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

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