package org.mobileposer.smpl

import android.content.Intent
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class ModelParityTest {
    @Test fun allSixModelsMatchOriginalPythonOnlinePose() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val report = JSONObject().put("device", android.os.Build.MODEL)
            .put("abi", android.os.Build.SUPPORTED_ABIS.joinToString()).put("tolerance", 1e-4)
        val results = JSONArray()
        val sparse = InstrumentationRegistry.getArguments().getString("sparseFrames") == "true"
        report.put("sparse_frames", sparse)
        for (layout in ImuClip.layouts.keys) {
            val clip = context.assets.open("$layout.input.json").use { ImuClip.read(it) }
            val refs = JSONObject(context.assets.open("$layout.reference.json").bufferedReader().use { it.readText() }).getJSONArray("frames")
            var maxError = 0.0
            val times = ArrayList<Double>()
            PoseEngine(context, layout).use { engine ->
                File(context.filesDir,"$layout.android.jsonl").bufferedWriter().use { writer ->
                    clip.frames.forEachIndexed { index, frame ->
                        if (sparse && index !in setOf(0, 4, 44, 45, 90, clip.frames.lastIndex)) {
                            engine.window.push(frame)
                            return@forEachIndexed
                        }
                        val start = SystemClock.elapsedRealtimeNanos()
                        val actual = engine.predict(frame)
                        val ms = (SystemClock.elapsedRealtimeNanos()-start)/1e6
                        times += ms
                        val expected = refs.getJSONObject(index).getJSONArray("smpl_local_rotations")
                        for (j in 0..23) for (r in 0..2) for (c in 0..2) {
                            maxError = maxOf(maxError,abs(actual[j*9+r*3+c]-expected.getJSONArray(j).getJSONArray(r).getDouble(c)))
                        }
                        val result = SmplOutput.json(layout,index.toLong(),frame.timeSeconds,engine.window.poseTime,ms,actual)
                        assertEquals(69,result.getJSONArray("body_pose").length())
                        assertEquals(refs.getJSONObject(index).getDouble("pose_time_seconds"),engine.window.poseTime,1e-8)
                        writer.appendLine(result.toString())
                    }
                    engine.window.reset()
                    val reset = engine.predict(clip.frames.first())
                    val first = refs.getJSONObject(0).getJSONArray("smpl_local_rotations")
                    for (j in 0..23) for (r in 0..2) for (c in 0..2) {
                        assertEquals(first.getJSONArray(j).getJSONArray(r).getDouble(c),reset[j*9+r*3+c].toDouble(),1e-4)
                    }
                }
            }
            times.sort()
            results.put(JSONObject().put("layout",layout).put("frames",times.size).put("source_frames",clip.frames.size).put("max_abs_error",maxError)
                .put("p50_ms",times[times.size/2]).put("p95_ms",times[(times.size*0.95).toInt().coerceAtMost(times.lastIndex)]))
            report.put("layouts",results)
            File(context.filesDir,"validation_report.android.json").writeText(report.toString(2))
            android.util.Log.i("MobilePoseParity", "$layout: ${times.size} frames, max error $maxError")
            assertTrue("$layout max error $maxError",maxError <= 1e-4)
        }
    }

    @Test fun activityLaunches() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val activity = instrumentation.startActivitySync(Intent(instrumentation.targetContext, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        assertNotNull(activity)
        instrumentation.runOnMainSync { activity.finish() }
    }
}
