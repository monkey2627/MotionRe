package org.mobileposer.smpl

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.Closeable
import java.nio.FloatBuffer
import java.security.MessageDigest

class PoseEngine(context: Context, layout: String) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val window = ImuWindow()

    init {
        require(ImuClip.layouts.containsKey(layout))
        val manifest = JSONObject(context.assets.open("manifest.json").bufferedReader().use { it.readText() })
        val models = manifest.getJSONArray("models")
        val entry = (0 until models.length()).map { models.getJSONObject(it) }.first { it.getString("layout") == layout }
        val bytes = context.assets.open(entry.getString("model")).use { it.readBytes() }
        val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(hash == entry.getString("sha256")) { "Model checksum mismatch" }
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            options.setInterOpNumThreads(1)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            session = env.createSession(bytes, options)
        }
    }

    fun predict(frame: ImuFrame): FloatArray {
        window.push(frame)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(window.values), longArrayOf(1, 45, 60)).use { input ->
            session.run(mapOf("imu" to input)).use { result ->
                val buffer = (result[0] as OnnxTensor).floatBuffer
                val out = FloatArray(216)
                buffer.get(out)
                require(out.all { it.isFinite() }) { "Non-finite model output" }
                return out
            }
        }
    }

    override fun close() { session.close() }
}
