package org.mobileposer.smpl

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class ContractTest {
    private fun rotation(axis: DoubleArray, angle: Double): FloatArray {
        val norm = sqrt(axis.sumOf { it * it })
        val x = axis[0]/norm; val y = axis[1]/norm; val z = axis[2]/norm
        val c = cos(angle); val s = sin(angle); val t = 1-c
        return doubleArrayOf(t*x*x+c, t*x*y-s*z, t*x*z+s*y, t*x*y+s*z, t*y*y+c, t*y*z-s*x,
            t*x*z-s*y, t*y*z+s*x, t*z*z+c).map { it.toFloat() }.toFloatArray()
    }

    @Test fun axisAnglesRoundTripAtSingularities() {
        for (axis in listOf(doubleArrayOf(1.0,0.0,0.0), doubleArrayOf(0.0,1.0,0.0),
            doubleArrayOf(0.0,0.0,1.0), doubleArrayOf(1.0,-2.0,3.0))) {
            for (angle in listOf(0.0, 1e-9, 0.5, PI-1e-7, PI, PI+1e-7, 4.0)) {
                val expected = rotation(axis, angle)
                val aa = SmplOutput.axisAngle(expected)
                assertTrue(aa.all { it.isFinite() })
                val magnitude = sqrt(aa.sumOf { it * it })
                val actual = rotation(if (magnitude < 1e-15) doubleArrayOf(1.0,0.0,0.0) else aa, magnitude)
                assertArrayEquals(expected, actual, 1e-5f)
            }
        }
    }

    @Test fun windowUsesDelayedFrameAndResets() {
        val w = ImuWindow()
        repeat(60) { i ->
            w.push(ImuFrame(i/30.0, FloatArray(60) { i.toFloat() }))
            assertEquals(max(0, i-4)/30.0, w.poseTime, 1e-10)
            assertEquals(max(0, i-4).toFloat(), w.values[40*60], 0f)
        }
        w.reset()
        w.push(ImuFrame(7.0, FloatArray(60) { 123f }))
        assertTrue(w.values.all { it == 123f })
        assertEquals(7.0, w.poseTime, 0.0)
    }

    private fun clip(): JSONObject {
        val identity = JSONArray("[[1,0,0],[0,1,0],[0,0,1]]")
        val acceleration = JSONArray(List(5) { listOf(30,60,90) })
        val orientations = JSONArray().apply { repeat(5) { put(identity) } }
        return JSONObject().apply {
            put("format_version",1); put("fps",30); put("layout","wrists_thighs_waist")
            put("sensor_labels",JSONArray(ImuClip.layouts.getValue("wrists_thighs_waist")))
            put("acceleration_units","m/s2"); put("acceleration_space","world_linear")
            put("orientation_space","bone_calibrated")
            put("frames",JSONArray().put(JSONObject().put("time_seconds",0)
                .put("acceleration",acceleration).put("orientation",orientations)))
        }
    }

    @Test fun featureOrderingAndScaleMatchTraining() {
        val features = ImuClip.parse(clip()).frames.single().features
        assertArrayEquals(FloatArray(15) { (it%3+1).toFloat() }, features.copyOfRange(0,15),0f)
        repeat(5) { assertArrayEquals(floatArrayOf(1f,0f,0f,0f,1f,0f,0f,0f,1f), features.copyOfRange(15+9*it,24+9*it),0f) }
    }

    @Test fun malformedInputIsRejected() {
        val wrongOrder = clip().put("sensor_labels",JSONArray(listOf("rw","lw","lt","rt","waist")))
        assertThrows(IllegalArgumentException::class.java) { ImuClip.parse(wrongOrder) }
        assertThrows(IllegalArgumentException::class.java) { ImuClip.parse(clip().put("fps",60)) }
        val invalidRotation = clip()
        invalidRotation.getJSONArray("frames").getJSONObject(0).getJSONArray("orientation")
            .getJSONArray(0).getJSONArray(0).put(0,-1)
        assertThrows(IllegalArgumentException::class.java) { ImuClip.parse(invalidRotation) }
    }

    @Test fun outputIsSmplNotSmplx() {
        val identity = FloatArray(216) { if (it%9 in listOf(0,4,8)) 1f else 0f }
        val json = SmplOutput.json("legs_waist",0,0.0,0.0,1.0,identity)
        assertEquals(3,json.getJSONArray("global_orient").length())
        assertEquals(69,json.getJSONArray("body_pose").length())
        assertEquals(10,json.getJSONArray("betas").length())
        assertEquals(3,json.getJSONArray("transl").length())
        assertEquals("smpl",json.getString("model_type"))
        assertFalse(json.has("expression"))
        assertTrue((0 until 69).all { json.getJSONArray("body_pose").getDouble(it) == 0.0 })
    }
}
