package org.mobileposer.smpl

import org.json.JSONObject
import java.io.InputStream
import java.io.ByteArrayOutputStream
import kotlin.math.abs

data class ImuFrame(val timeSeconds: Double, val features: FloatArray)
data class ImuClip(val layout: String, val frames: List<ImuFrame>) {
    companion object {
        val layouts = linkedMapOf(
            "wrists_thighs_waist" to listOf("lw", "rw", "lt", "rt", "waist"),
            "wrists_shanks_waist" to listOf("lw", "rw", "ls", "rs", "waist"),
            "wrists_feet_waist" to listOf("lw", "rw", "lf", "rf", "waist"),
            "upperarms_thighs_waist" to listOf("lu", "ru", "lt", "rt", "waist"),
            "wrists_upperarms_waist" to listOf("lw", "rw", "lu", "ru", "waist"),
            "legs_waist" to listOf("lt", "rt", "ls", "rs", "waist")
        )

        fun read(input: InputStream): ImuClip {
            // Bound imported JSON before allocating a full parse tree.
            val output = ByteArrayOutputStream()
            val buffer = ByteArray(8192)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                require(output.size() + count <= 20 * 1024 * 1024) { "Input exceeds 20 MiB" }
                output.write(buffer, 0, count)
            }
            val bytes = output.toByteArray()
            return parse(JSONObject(bytes.toString(Charsets.UTF_8)))
        }

        fun parse(json: JSONObject): ImuClip {
            require(json.getInt("format_version") == 1)
            require(json.getInt("fps") == 30) { "Input must be resampled to 30 Hz" }
            val layout = json.getString("layout")
            val labels = requireNotNull(layouts[layout]) { "Unknown layout" }
            val inputLabels = json.getJSONArray("sensor_labels")
            require(inputLabels.length() == 5 && labels.indices.all { inputLabels.getString(it) == labels[it] }) {
                "Sensor order does not match layout"
            }
            require(json.getString("acceleration_units") == "m/s2")
            require(json.getString("acceleration_space") == "world_linear")
            require(json.getString("orientation_space") == "bone_calibrated")
            val raw = json.getJSONArray("frames")
            require(raw.length() in 1..18000) { "Expected 1..18000 frames" }
            var last = -1.0
            val frames = (0 until raw.length()).map { index ->
                val frame = raw.getJSONObject(index)
                val time = frame.getDouble("time_seconds")
                require(time.isFinite() && time >= 0 && (index == 0 || abs(time - last - 1.0 / 30) < 0.001)) {
                    "Timestamps must be finite, increasing, and spaced at 30 Hz"
                }
                last = time
                val acc = frame.getJSONArray("acceleration")
                val ori = frame.getJSONArray("orientation")
                require(acc.length() == 5 && ori.length() == 5)
                val features = FloatArray(60)
                for (sensor in 0..4) {
                    val a = acc.getJSONArray(sensor)
                    val r = ori.getJSONArray(sensor)
                    require(a.length() == 3 && r.length() == 3)
                    for (row in 0..2) {
                        features[sensor * 3 + row] = (a.getDouble(row) / 30).toFloat()
                        val rowValues = r.getJSONArray(row)
                        require(rowValues.length() == 3)
                        for (col in 0..2) features[15 + sensor * 9 + row * 3 + col] = rowValues.getDouble(col).toFloat()
                    }
                    val offset = 15 + sensor * 9
                    for (row in 0..2) for (col in 0..2) {
                        val dot = (0..2).sumOf { k -> (features[offset + row * 3 + k] * features[offset + col * 3 + k]).toDouble() }
                        require(abs(dot - if (row == col) 1.0 else 0.0) < 0.02) { "Invalid orientation matrix" }
                    }
                    val m = features.copyOfRange(offset, offset + 9)
                    val det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6])
                    require(det > 0.98f) { "Orientation must be a proper rotation" }
                }
                require(features.all { it.isFinite() }) { "Non-finite input" }
                ImuFrame(time, features)
            }
            return ImuClip(layout, frames)
        }
    }
}

class ImuWindow {
    val values = FloatArray(45 * 60)
    private val times = DoubleArray(45)
    var frameCount = 0
        private set
    val poseTime get() = times[40]

    fun reset() { frameCount = 0; values.fill(0f); times.fill(0.0) }

    fun push(frame: ImuFrame) {
        require(frame.features.size == 60 && frame.features.all { it.isFinite() })
        if (frameCount == 0) {
            repeat(45) { frame.features.copyInto(values, it * 60) }
            times.fill(frame.timeSeconds)
        } else {
            values.copyInto(values, 0, 60, values.size)
            frame.features.copyInto(values, 44 * 60)
            times.copyInto(times, 0, 1, 45)
            times[44] = frame.timeSeconds
        }
        frameCount++
    }
}
