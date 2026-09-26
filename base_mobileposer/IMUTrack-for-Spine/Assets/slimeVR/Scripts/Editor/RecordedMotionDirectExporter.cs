#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using SpineFlow.MotionPackages;
using UnityEditor;
using UnityEngine;

/// <summary>
/// One-click demo export invoked by RecordingController's existing Game UI.
/// It treats the latest recording as one complete segment.
/// </summary>
public static class RecordedMotionDirectExporter
{
    private const string DefaultActionId = "dead_bug_demo";
    private const string DefaultDisplayName = "Dead Bug Demo";
    private const float JsonFps = 30f;

    [MenuItem("IMUTrack/Export Latest Recording Directly")]
    public static void ExportLatest()
    {
        RecordingController controller = UnityEngine.Object.FindObjectOfType<RecordingController>();
        PoseFbxRecorder recorder = controller != null && controller.poseRecorder != null
            ? controller.poseRecorder
            : UnityEngine.Object.FindObjectOfType<PoseFbxRecorder>();

        if (recorder == null || recorder.AvatarAnimator == null)
        {
            Report(controller, false, "导出失败：没有找到录制角色的 Humanoid Animator");
            return;
        }

        string actionId = DefaultActionId;
        string displayName = DefaultDisplayName;
        if (controller != null && !controller.TryGetLatestRecordingForExport(
                out _, out actionId, out displayName, out _, out string identityError))
        {
            Report(controller, false, "导出失败：" + identityError);
            return;
        }
        if (!MotionPackageStorage.IsValidIdentifier(actionId))
        {
            Report(controller, false, "导出失败：actionId 只能包含小写字母、数字和下划线");
            return;
        }

        AnimationClip sourceClip = recorder.LastExportedClip;
        if (sourceClip == null) sourceClip = LoadLatestRecording();
        if (sourceClip == null)
        {
            Report(controller, false, "导出失败：Assets/Recordings 中没有录制 .anim");
            return;
        }

        GameObject sourceAvatar = recorder.AvatarAnimator.gameObject;
        string outputDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "Recordings"));
        Directory.CreateDirectory(outputDirectory);

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string fbxPath = Path.Combine(outputDirectory, $"{actionId}_{timestamp}.fbx");
        string jsonPath = Path.Combine(outputDirectory, $"{actionId}.motion.json");

        if (!RecordedFbxPackageExporter.TryExportPackage(
                sourceAvatar, sourceClip, fbxPath, out string exportError))
        {
            Report(controller, false, "FBX 导出失败：" + exportError);
            return;
        }

        int lastFrame = Mathf.Max(1, Mathf.RoundToInt(sourceClip.length * sourceClip.frameRate));
        if (!RecordedFbxPackageExporter.ConfigureClipRanges(
                fbxPath,
                new[] { actionId },
                new[] { 0 },
                new[] { lastFrame },
                out string rangeError))
        {
            Report(controller, false, "FBX 片段配置失败：" + rangeError);
            return;
        }

        GameObject sampleTarget = UnityEngine.Object.Instantiate(sourceAvatar);
        sampleTarget.name = "__IMUTrackDirectExportSample";
        Animator sampleAnimator = sampleTarget.GetComponentInChildren<Animator>();

        try
        {
            if (sampleAnimator == null || sampleAnimator.avatar == null ||
                !sampleAnimator.avatar.isValid || !sampleAnimator.avatar.isHuman)
            {
                Report(controller, false, "JSON 导出失败：录制角色不是有效 Humanoid");
                return;
            }

            RecordedCoachMotionFrame[] frames = RecordedCoachMotionSampler.SampleClip(
                sourceClip, sampleTarget, sampleAnimator, JsonFps);
            if (frames.Length == 0)
            {
                Report(controller, false, "JSON 导出失败：没有采样到姿态帧");
                return;
            }

            var package = new RecordedCoachMotionPackage
            {
                formatVersion = 1,
                actionId = actionId,
                displayName = displayName,
                fps = JsonFps,
                sourceClipName = sourceClip.name,
                setCount = 1,
                segments = new[]
                {
                    new RecordedCoachMotionSegment
                    {
                        label = actionId,
                        sourceStateName = actionId,
                        repeatCount = 1,
                        scoreLeniency = 1f,
                        scoringDuration = Mathf.Max(0.1f, sourceClip.length),
                        bodyParts = new[] { "whole_body" },
                        holdSeconds = 0f,
                        applyFormalActionScoreMapping = true,
                        voiceClipName = null,
                        frames = frames
                    }
                }
            };

            File.WriteAllText(jsonPath, JsonUtility.ToJson(package, true));
            string jsonAssetPath = RecordedFbxPackageExporter.ToAssetPath(jsonPath);
            AssetDatabase.ImportAsset(jsonAssetPath, ImportAssetOptions.ForceUpdate);
            AssetDatabase.SaveAssets();

            var manifestSegments = new[]
            {
                new MotionPackageSegmentManifest
                {
                    name = actionId,
                    firstFrame = 0,
                    lastFrame = lastFrame,
                    durationSeconds = Mathf.Max(0.1f, sourceClip.length)
                }
            };
            if (!MotionPackageStorage.TryCreateVersionedPackage(
                    fbxPath, jsonPath, controller?.GetLatestTrackerJsonPath(),
                    controller?.GetLatestRawImuJsonPathForAction(actionId),
                    actionId, displayName, manifestSegments,
                    out MotionPackageManifest manifest, out string manifestPath, out string packageError))
            {
                Report(controller, false, "共享动作包生成失败：" + packageError);
                return;
            }

            Report(controller, true,
                $"导出成功：{actionId} v{manifest.version}（{frames.Length} 帧，" +
                $"Raw IMU {manifest.rawImuFrameCount} 帧/{manifest.rawImuSensorCount} 个传感器）");
            Debug.Log($"[IMUTrack Direct Export] FBX: {fbxPath}\nJSON: {jsonPath}\nManifest: {manifestPath}");
        }
        catch (Exception exception)
        {
            Report(controller, false, "动作包导出失败：" + exception.Message);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(sampleTarget);
        }
    }

    private static AnimationClip LoadLatestRecording()
    {
        string directory = Path.GetFullPath(Path.Combine(Application.dataPath, "Recordings"));
        if (!Directory.Exists(directory)) return null;

        FileInfo latest = new DirectoryInfo(directory).GetFiles("*.anim", SearchOption.AllDirectories)
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .FirstOrDefault();
        if (latest == null) return null;

        string assetPath = RecordedFbxPackageExporter.ToAssetPath(latest.FullName);
        return string.IsNullOrEmpty(assetPath)
            ? null
            : AssetDatabase.LoadAssetAtPath<AnimationClip>(assetPath);
    }

    private static void Report(RecordingController controller, bool success, string message)
    {
        if (controller != null) controller.ShowOperationStatus(message);
        if (success) Debug.Log("[IMUTrack Direct Export] " + message);
        else Debug.LogError("[IMUTrack Direct Export] " + message);
    }
}
#endif
