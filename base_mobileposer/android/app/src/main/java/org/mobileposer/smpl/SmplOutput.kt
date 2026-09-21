package org.mobileposer.smpl

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.*

object SmplOutput {
    val ignoredJoints = listOf(7, 8, 10, 11, 20, 21, 22, 23)

    // Matrix -> unit quaternion -> shortest axis angle. Stable at both 0 and pi.
    fun axisAngle(m: FloatArray): DoubleArray {
        require(m.size == 9 && m.all { it.isFinite() })
        val a = DoubleArray(9) { m[it].toDouble() }
        var w: Double; var x: Double; var y: Double; var z: Double
        val trace = a[0] + a[4] + a[8]
        if (trace > 0) {
            val s = sqrt(trace + 1.0) * 2
            w = s / 4; x = (a[7] - a[5]) / s; y = (a[2] - a[6]) / s; z = (a[3] - a[1]) / s
        } else if (a[0] > a[4] && a[0] > a[8]) {
            val s = sqrt(max(0.0, 1 + a[0] - a[4] - a[8])) * 2
            w = (a[7] - a[5]) / s; x = s / 4; y = (a[1] + a[3]) / s; z = (a[2] + a[6]) / s
        } else if (a[4] > a[8]) {
            val s = sqrt(max(0.0, 1 + a[4] - a[0] - a[8])) * 2
            w = (a[2] - a[6]) / s; x = (a[1] + a[3]) / s; y = s / 4; z = (a[5] + a[7]) / s
        } else {
            val s = sqrt(max(0.0, 1 + a[8] - a[0] - a[4])) * 2
            w = (a[3] - a[1]) / s; x = (a[2] + a[6]) / s; y = (a[5] + a[7]) / s; z = s / 4
        }
        val norm = sqrt(w*w + x*x + y*y + z*z)
        require(norm.isFinite() && norm > 1e-10) { "Invalid output rotation" }
        val scale = (if (w < 0) -1 else 1) / norm
        w *= scale; x *= scale; y *= scale; z *= scale
        val sinHalf = sqrt(x*x + y*y + z*z)
        val factor = if (sinHalf < 1e-8) 2.0 else 2 * atan2(sinHalf, w) / sinHalf
        return doubleArrayOf(x * factor, y * factor, z * factor)
    }

    fun json(layout: String, index: Long, inputTime: Double, poseTime: Double, inferenceMs: Double,
             rotations: FloatArray): JSONObject {
        require(rotations.size == 216)
        val pose = (0 until 24).flatMap { axisAngle(rotations.copyOfRange(it * 9, (it + 1) * 9)).toList() }
        return JSONObject().apply {
            put("format_version", 1); put("model_type", "smpl"); put("layout", layout)
            put("frame_index", index); put("input_time_seconds", inputTime); put("pose_time_seconds", poseTime)
            put("inference_ms", inferenceMs); put("rotation_units", "radians")
            put("coordinate_space", "training_world_y_up"); put("rotation_space", "local_parent")
            put("global_orient", JSONArray(pose.take(3))); put("body_pose", JSONArray(pose.drop(3)))
            put("transl", JSONArray(listOf(0, 0, 0))); put("betas", JSONArray(List(10) { 0 }))
            put("default_fields", JSONArray(listOf("transl", "betas")))
            put("identity_joint_indices", JSONArray(ignoredJoints))
            put("smpl_local_rotations", JSONArray(rotations.map { it.toDouble() }))
        }
    }
}
