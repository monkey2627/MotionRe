package org.mobileposer.smpl

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import android.view.View
import android.view.WindowManager
import android.widget.*
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.ceil

class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val running = AtomicBoolean(false)
    private lateinit var layoutPicker: Spinner
    private lateinit var status: TextView
    private lateinit var preview: TextView
    private lateinit var realtime: CheckBox
    private lateinit var benchmark: CheckBox
    private val lockedControls = mutableListOf<View>()
    private var imported: ImuClip? = null
    private var lastOutput: File? = null
    private var pendingExport: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 32, 24, 24)
        }
        container.addView(TextView(this).apply { text = "MobilePoser → SMPL"; textSize = 25f })
        container.addView(TextView(this).apply { text = "手机 CPU 推理 · 5 IMU 模拟输入 · 无需电脑或联网" })
        layoutPicker = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, ImuClip.layouts.keys.toList())
        }
        container.addView(layoutPicker); lockedControls += layoutPicker
        realtime = CheckBox(this).apply { text = "按 30 Hz 回放（较慢设备会显示实际帧率）"; isChecked = true }
        benchmark = CheckBox(this).apply { text = "连续运行 10 分钟（循环片段，保存性能报告）" }
        container.addView(realtime); container.addView(benchmark)
        lockedControls += realtime; lockedControls += benchmark
        fun button(title: String, lock: Boolean = true, action: () -> Unit) {
            val b = Button(this).apply { text = title; setOnClickListener { action() } }
            container.addView(b)
            if (lock) lockedControls += b
        }
        button("使用内置模拟数据") { imported = null; status.text = "已选择内置数据" }
        button("导入 IMU JSON") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                type = "*/*"; addCategory(Intent.CATEGORY_OPENABLE)
            }, 1)
        }
        button("开始推理") { startRun() }
        button("停止", false) { running.set(false) }
        button("导出最近 SMPL JSONL") { exportFile(lastOutput) }
        button("导出最近性能报告") { exportFile(lastOutput?.let { File(it.path + ".report.json") }) }
        status = TextView(this).apply { text = "就绪。默认片段为 6 秒；输出包含姿态参数，不包含预测体型或位移。" }
        preview = TextView(this).apply { textSize = 11f; setTextIsSelectable(true) }
        container.addView(status); container.addView(preview)
        setContentView(ScrollView(this).apply { addView(container) })
    }

    private fun setBusy(value: Boolean) {
        lockedControls.forEach { it.isEnabled = !value }
        if (value) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun startRun() {
        if (!running.compareAndSet(false, true)) return
        val layout = layoutPicker.selectedItem.toString()
        val custom = imported
        if (custom != null && custom.layout != layout) {
            running.set(false); status.text = "导入数据的布局与所选模型不一致"; return
        }
        val paced = realtime.isChecked
        val tenMinutes = benchmark.isChecked
        setBusy(true); status.text = "正在加载模型…"
        worker.execute {
            val timings = ArrayList<Double>()
            var peakPssKb = 0
            val intervalStats = org.json.JSONArray()
            var intervalCount = 0
            var intervalStart = 0L
            var start = 0L
            var output: File? = null
            var failure: String? = null
            try {
                val clip = custom ?: assets.open("$layout.input.json").use { ImuClip.read(it) }
                PoseEngine(this, layout).use { engine ->
                    output = File(filesDir, "smpl_${System.currentTimeMillis()}.jsonl")
                    output!!.bufferedWriter().use { writer ->
                        start = SystemClock.elapsedRealtimeNanos()
                        intervalStart = start
                        var nextTick = start
                        var frameIndex = 0
                        var cycle = 0
                        var count = 0L
                        while (running.get()) {
                            if (tenMinutes && SystemClock.elapsedRealtimeNanos() - start >= 600_000_000_000L) break
                            if (frameIndex == clip.frames.size) {
                                if (!tenMinutes) break
                                frameIndex = 0; cycle++; engine.window.reset()
                            }
                            val original = clip.frames[frameIndex]
                            val input = ImuFrame(original.timeSeconds + cycle * clip.frames.size / 30.0, original.features)
                            val before = SystemClock.elapsedRealtimeNanos()
                            val rotations = engine.predict(input)
                            val ms = (SystemClock.elapsedRealtimeNanos() - before) / 1e6
                            timings += ms
                            val json = SmplOutput.json(layout, count, input.timeSeconds, engine.window.poseTime, ms, rotations)
                            json.put("source", if (custom == null) "bundled_synthetic_imu" else "imported_imu")
                            json.put("replay_cycle", cycle)
                            writer.appendLine(json.toString())
                            frameIndex++; count++
                            intervalCount++
                            val now = SystemClock.elapsedRealtimeNanos()
                            if (count == 1L || now - intervalStart >= 1_000_000_000L) {
                                val info = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
                                peakPssKb = maxOf(peakPssKb, info.totalPss)
                                if (now - intervalStart >= 1_000_000_000L) {
                                    intervalStats.put(JSONObject().put("elapsed_seconds", (now-start)/1e9)
                                        .put("fps", intervalCount * 1e9 / (now-intervalStart))
                                        .put("pss_kb", info.totalPss))
                                    intervalStart = now; intervalCount = 0
                                }
                                val fps = count * 1e9 / (now - start)
                                val text = "帧数 $count · %.1f FPS · 推理 %.1f ms\n内存 PSS %.1f MiB".format(fps, ms, info.totalPss / 1024.0)
                                val display = JSONObject(json.toString()).apply { remove("smpl_local_rotations") }.toString(2)
                                runOnUiThread { status.text = text; preview.text = display }
                            }
                            if (paced) {
                                // No producer queue: late devices slow replay instead of accumulating work.
                                nextTick += 33_333_333L
                                val remaining = nextTick - SystemClock.elapsedRealtimeNanos()
                                if (remaining > 0) Thread.sleep(remaining / 1_000_000L, (remaining % 1_000_000L).toInt())
                                else nextTick = SystemClock.elapsedRealtimeNanos()
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                failure = e.message ?: e.javaClass.simpleName
            } finally {
                val seconds = if (start == 0L) 0.0 else (SystemClock.elapsedRealtimeNanos() - start) / 1e9
                timings.sort()
                fun percentile(p: Double) = if (timings.isEmpty()) 0.0 else timings[(ceil(p * timings.size).toInt() - 1).coerceAtLeast(0)]
                val report = JSONObject().apply {
                    put("layout", layout); put("frames", timings.size); put("elapsed_seconds", seconds)
                    put("actual_fps", if (seconds > 0) timings.size / seconds else 0)
                    put("inference_p50_ms", percentile(0.5)); put("inference_p95_ms", percentile(0.95))
                    put("peak_sampled_pss_kb", peakPssKb); put("intervals", intervalStats)
                    put("device", android.os.Build.MODEL); put("abis", android.os.Build.SUPPORTED_ABIS.joinToString())
                    put("android_api", android.os.Build.VERSION.SDK_INT)
                    put("runtime", "onnxruntime-android 1.19.2 CPU FP32, 2 threads")
                    put("ten_minute_run_completed", tenMinutes && seconds >= 600)
                    put("paced_replay", paced); put("error", failure ?: JSONObject.NULL)
                }
                try { output?.let { File(it.path + ".report.json").writeText(report.toString(2)) } }
                catch (e: Exception) { failure = "Report write failed: ${e.message}" }
                val error = failure
                runOnUiThread {
                    running.set(false); setBusy(false)
                    lastOutput = output
                    status.text = if (error != null) "运行失败：$error" else
                        "完成 ${timings.size} 帧 · %.1f FPS\nP50 %.1f ms · P95 %.1f ms\n可导出 SMPL 参数和性能报告。".format(
                            if (seconds > 0) timings.size / seconds else 0.0, percentile(0.5), percentile(0.95))
                }
            }
        }
    }

    private fun exportFile(file: File?) {
        if (file == null || !file.isFile) { status.text = "请先完成一次推理"; return }
        pendingExport = file
        startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            type = "application/json"; addCategory(Intent.CATEGORY_OPENABLE); putExtra(Intent.EXTRA_TITLE, file.name)
        }, 2)
    }

    @Deprecated("Platform callback")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        setBusy(true)
        worker.execute {
            try {
                if (requestCode == 1) {
                    val clip = requireNotNull(contentResolver.openInputStream(uri)).use { ImuClip.read(it) }
                    runOnUiThread {
                        imported = clip
                        layoutPicker.setSelection(ImuClip.layouts.keys.indexOf(clip.layout))
                        status.text = "已导入 ${clip.frames.size} 帧，布局 ${clip.layout}"
                    }
                } else if (requestCode == 2) {
                    val file = requireNotNull(pendingExport)
                    requireNotNull(contentResolver.openOutputStream(uri)).use { out -> file.inputStream().use { it.copyTo(out) } }
                    runOnUiThread { status.text = "导出完成" }
                }
            } catch (e: Exception) { runOnUiThread { status.text = "文件操作失败：${e.message}" } }
            finally { runOnUiThread { setBusy(false) } }
        }
    }

    override fun onPause() { running.set(false); super.onPause() }
    override fun onDestroy() { running.set(false); worker.shutdown(); super.onDestroy() }
}
