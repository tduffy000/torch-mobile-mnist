package io.thomasduffy.torchmnist

import android.content.Context
import java.io.File
import java.io.FileOutputStream

class FileHandler {

    companion object FilePath {
        fun assetFilePath(ctx: Context, assetName: String): String? {
            val f = File(ctx.filesDir, assetName)
            if (f.exists() && f.length() > 0) return f.absolutePath
//        try {
//            val inStream = ctx.assets.open(assetName)
//            val outStream = FileOutputStream(f)
//
//            val buf = byteArrayOf()
//            val read = 0
//
//            return
//        }
            return null
        }
    }

}