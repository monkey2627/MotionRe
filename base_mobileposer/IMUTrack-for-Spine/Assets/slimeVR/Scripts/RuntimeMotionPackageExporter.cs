using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using SpineFlow.MotionEditing;
using SpineFlow.MotionPackages;
using SpineFlow.PoseRecording;
using UnityEngine;

#if UNITY_EDITOR || FBXSDK_RUNTIME
using Autodesk.Fbx;
#endif

namespace SpineFlow.RuntimeExport
{
    /// <summary>
    /// Creates the same shared motion-package contract as the Editor exporter,
    /// directly from a runtime pose recording.
    /// </summary>
    public static class RuntimeMotionPackageExporter
    {
        private static readonly string[] ValidBodyParts =
        {
            "whole_body", "torso", "hips", "head_neck", "left_arm", "right_arm", "left_leg", "right_leg"
        };

        public static bool TryExport(
            string poseRecordingPath,
            string trackerJsonPath,
            string rawImuJsonPath,
            string actionId,
            string displayName,
            out MotionPackageManifest manifest,
            out string manifestPath,
            out string error)
        {
            manifest = null;
            manifestPath = null;
            error = null;

            if (!RuntimePoseRecordingStorage.TryLoad(poseRecordingPath,
                    out RuntimePoseRecordingPackage pose, out error))
                return false;
            if (!string.IsNullOrWhiteSpace(pose.actionId)) actionId = pose.actionId.Trim();
            if (!string.IsNullOrWhiteSpace(pose.displayName)) displayName = pose.displayName.Trim();
            if (!MotionPackageStorage.IsValidIdentifier(actionId))
            {
                error = "录制开始时指定的 actionId 无效";
                return false;
            }
            if (pose.frames.Length < 2)
            {
                error = "姿态录制至少需要两帧才能导出动作包";
                return false;
            }

            string keyframesPath = RuntimePoseRecordingStorage.GetKeyframesPath(poseRecordingPath);
            if (!TryLoadOrCreateEditDefinition(keyframesPath, poseRecordingPath, pose, actionId, displayName,
                    out RecordedMotionEditDefinition definition, out error))
                return false;
            if (!TryNormalizeAndValidate(definition, pose.frames.Length - 1, actionId, out error))
                return false;

            try
            {
                string exportDirectory = Path.Combine(RuntimePoseRecordingStorage.RootDirectory, "Exports");
                Directory.CreateDirectory(exportDirectory);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string baseName = actionId + "_" + timestamp;
                string fbxPath = GetUniquePath(Path.Combine(exportDirectory, baseName + ".fbx"));
                string motionJsonPath = Path.ChangeExtension(fbxPath, ".motion.json");

                MotionPackageSegmentManifest[] manifestSegments = BuildManifestSegments(definition);
                if (!RuntimeFbxExporter.TryExport(pose, definition, fbxPath, out error)) return false;

                RecordedCoachMotionPackage motionPackage = BuildMotionPackage(pose, definition,
                    actionId, displayName);
                File.WriteAllText(motionJsonPath, JsonUtility.ToJson(motionPackage, true),
                    new UTF8Encoding(false));

                if (!MotionPackageStorage.TryCreateVersionedPackage(
                        fbxPath, motionJsonPath, ExistingOrNull(trackerJsonPath), ExistingOrNull(rawImuJsonPath),
                        keyframesPath, actionId,
                        string.IsNullOrWhiteSpace(displayName) ? actionId : displayName.Trim(),
                        manifestSegments, out manifest, out manifestPath, out error))
                    return false;

                return true;
            }
            catch (Exception exception)
            {
                error = "动作包导出失败：" + exception.Message;
                return false;
            }
        }

        public static RecordedMotionEditDefinition CreateDefaultEditDefinition(
            RuntimePoseRecordingPackage pose, string actionId, string displayName)
        {
            int lastFrame = Mathf.Max(1, pose.frames.Length - 1);
            float naturalDuration = Mathf.Max(0.1f,
                pose.frames[lastFrame].timeSeconds - pose.frames[0].timeSeconds);
            return new RecordedMotionEditDefinition
            {
                formatVersion = 1,
                sourceClipAssetPath = pose.trajectoryName,
                sourceClipName = pose.trajectoryName,
                sourceClipLengthSeconds = pose.durationSeconds,
                sourceFrameRate = pose.sampleRate,
                totalFrames = lastFrame,
                actionId = actionId,
                displayName = string.IsNullOrWhiteSpace(displayName) ? actionId : displayName,
                sampleFps = Mathf.Clamp(Mathf.RoundToInt(pose.sampleRate), 1, 120),
                setCount = 1,
                savedAtUtc = DateTime.UtcNow.ToString("o"),
                keyframes = new[]
                {
                    new RecordedMotionKeyframeDefinition
                    {
                        name = "start",
                        frame = 0,
                        bodyParts = new[] { "whole_body" },
                        followingSegmentName = actionId,
                        followingSegmentDurationSeconds = naturalDuration
                    },
                    new RecordedMotionKeyframeDefinition
                    {
                        name = "end",
                        frame = lastFrame,
                        bodyParts = new[] { "whole_body" }
                    }
                }
            };
        }

        public static bool TrySaveEditDefinition(string path, RecordedMotionEditDefinition definition,
            out string error)
        {
            error = null;
            try
            {
                definition.savedAtUtc = DateTime.UtcNow.ToString("o");
                Directory.CreateDirectory(Path.GetDirectoryName(path) ?? RuntimePoseRecordingStorage.RootDirectory);
                File.WriteAllText(path, JsonUtility.ToJson(definition, true), new UTF8Encoding(false));
                return true;
            }
            catch (Exception exception)
            {
                error = "无法保存关键帧定义：" + exception.Message;
                return false;
            }
        }

        private static bool TryLoadOrCreateEditDefinition(string path, string poseRecordingPath,
            RuntimePoseRecordingPackage pose, string actionId, string displayName,
            out RecordedMotionEditDefinition definition, out string error)
        {
            definition = null;
            error = null;
            try
            {
                if (File.Exists(path))
                {
                    definition = JsonUtility.FromJson<RecordedMotionEditDefinition>(
                        File.ReadAllText(path, Encoding.UTF8));
                    if (definition == null || definition.formatVersion != 1)
                    {
                        error = "关键帧定义格式无效：" + path;
                        return false;
                    }
                }
                else
                {
                    definition = CreateDefaultEditDefinition(pose, actionId, displayName);
                }

                definition.actionId = actionId;
                definition.displayName = string.IsNullOrWhiteSpace(displayName) ? actionId : displayName.Trim();
                definition.sourceClipAssetPath = Path.GetFullPath(poseRecordingPath);
                definition.sourceClipName = pose.trajectoryName;
                definition.sourceClipLengthSeconds = pose.durationSeconds;
                definition.sourceFrameRate = pose.sampleRate;
                definition.totalFrames = pose.frames.Length - 1;
                if (!TryNormalizeAndValidate(definition, pose.frames.Length - 1, actionId, out error))
                    return false;
                if (!TrySaveEditDefinition(path, definition, out error)) return false;
                return true;
            }
            catch (Exception exception)
            {
                error = "无法读取关键帧定义：" + exception.Message;
                return false;
            }
        }

        internal static bool TryNormalizeAndValidate(RecordedMotionEditDefinition definition, int lastFrame,
            string fallbackActionId, out string error)
        {
            error = null;
            if (definition?.keyframes == null || definition.keyframes.Length < 2)
            {
                error = "至少需要起点和终点两个关键帧";
                return false;
            }

            var boundaries = definition.keyframes.Where(item => item != null)
                .OrderBy(item => item.frame).ToArray();
            if (boundaries.Length < 2)
            {
                error = "关键帧定义缺少有效边界";
                return false;
            }

            boundaries[0].frame = 0;
            boundaries[boundaries.Length - 1].frame = lastFrame;
            var keyframeNames = new HashSet<string>(StringComparer.Ordinal);
            var segmentNames = new HashSet<string>(StringComparer.Ordinal);
            for (int index = 0; index < boundaries.Length; index++)
            {
                RecordedMotionKeyframeDefinition boundary = boundaries[index];
                if (index > 0 && boundary.frame <= boundaries[index - 1].frame)
                {
                    error = "关键帧位置必须严格递增且不能位于同一帧";
                    return false;
                }

                if (!MotionPackageStorage.IsValidIdentifier(boundary.name) ||
                    !keyframeNames.Add(boundary.name))
                {
                    error = "关键帧名称必须唯一，并且只能包含小写字母、数字和下划线";
                    return false;
                }

                boundary.bodyParts = NormalizeBodyParts(boundary.bodyParts);
                boundary.holdSeconds = Mathf.Max(0f, boundary.holdSeconds);
                if (index >= boundaries.Length - 1)
                {
                    boundary.followingSegmentName = string.Empty;
                    boundary.followingSegmentDurationSeconds = 0f;
                    continue;
                }

                if (string.IsNullOrWhiteSpace(boundary.followingSegmentName))
                    boundary.followingSegmentName = index == 0
                        ? fallbackActionId
                        : fallbackActionId + "_segment_" + (index + 1);
                if (!MotionPackageStorage.IsValidIdentifier(boundary.followingSegmentName) ||
                    !segmentNames.Add(boundary.followingSegmentName))
                {
                    error = "动作片段名称必须唯一，并且只能包含小写字母、数字和下划线";
                    return false;
                }

                boundary.followingSegmentDurationSeconds =
                    Mathf.Max(0.1f, boundary.followingSegmentDurationSeconds);
            }

            definition.keyframes = boundaries;
            definition.totalFrames = lastFrame;
            definition.setCount = Mathf.Max(1, definition.setCount);
            return true;
        }

        internal static string[] NormalizeBodyParts(string[] bodyParts)
        {
            if (bodyParts == null || bodyParts.Length == 0) return new[] { "whole_body" };
            if (bodyParts.Contains("whole_body")) return new[] { "whole_body" };

            string[] result = bodyParts.Where(part => ValidBodyParts.Contains(part) && part != "whole_body")
                .Distinct().ToArray();
            return result.Length == 0 ? new[] { "whole_body" } : result;
        }

        private static RecordedCoachMotionPackage BuildMotionPackage(RuntimePoseRecordingPackage pose,
            RecordedMotionEditDefinition definition, string actionId, string displayName)
        {
            var segments = new RecordedCoachMotionSegment[definition.keyframes.Length - 1];
            for (int segmentIndex = 0; segmentIndex < segments.Length; segmentIndex++)
            {
                RecordedMotionKeyframeDefinition from = definition.keyframes[segmentIndex];
                RecordedMotionKeyframeDefinition destination = definition.keyframes[segmentIndex + 1];
                float startTime = pose.frames[from.frame].timeSeconds;
                int count = destination.frame - from.frame + 1;
                var frames = new RecordedCoachMotionFrame[count];
                for (int localIndex = 0; localIndex < count; localIndex++)
                {
                    RuntimePoseFrame source = pose.frames[from.frame + localIndex];
                    frames[localIndex] = new RecordedCoachMotionFrame
                    {
                        time = Mathf.Max(0f, source.timeSeconds - startTime),
                        bodyPosition = source.bodyPosition,
                        bodyRotation = source.bodyRotation,
                        muscles = (float[])source.muscles.Clone()
                    };
                }

                segments[segmentIndex] = new RecordedCoachMotionSegment
                {
                    label = from.followingSegmentName,
                    sourceStateName = from.followingSegmentName,
                    repeatCount = 1,
                    scoreLeniency = 1f,
                    scoringDuration = Mathf.Max(0.1f, from.followingSegmentDurationSeconds),
                    bodyParts = NormalizeBodyParts(destination.bodyParts),
                    holdSeconds = Mathf.Max(0f, destination.holdSeconds),
                    applyFormalActionScoreMapping = true,
                    voiceClipName = null,
                    frames = frames
                };
            }

            return new RecordedCoachMotionPackage
            {
                actionId = actionId,
                displayName = string.IsNullOrWhiteSpace(displayName) ? actionId : displayName.Trim(),
                fps = pose.sampleRate,
                sourceClipName = pose.trajectoryName,
                setCount = Mathf.Max(1, definition.setCount),
                segments = segments
            };
        }

        private static MotionPackageSegmentManifest[] BuildManifestSegments(
            RecordedMotionEditDefinition definition)
        {
            var result = new MotionPackageSegmentManifest[definition.keyframes.Length - 1];
            for (int index = 0; index < result.Length; index++)
            {
                result[index] = new MotionPackageSegmentManifest
                {
                    name = definition.keyframes[index].followingSegmentName,
                    firstFrame = definition.keyframes[index].frame,
                    lastFrame = definition.keyframes[index + 1].frame,
                    durationSeconds = Mathf.Max(0.1f,
                        definition.keyframes[index].followingSegmentDurationSeconds)
                };
            }

            return result;
        }

        private static string ExistingOrNull(string path)
        {
            return !string.IsNullOrWhiteSpace(path) && File.Exists(path) ? path : null;
        }

        private static string GetUniquePath(string path)
        {
            if (!File.Exists(path)) return path;
            string directory = Path.GetDirectoryName(path) ?? string.Empty;
            string name = Path.GetFileNameWithoutExtension(path);
            string extension = Path.GetExtension(path);
            for (int index = 2;; index++)
            {
                string candidate = Path.Combine(directory, name + "_" + index + extension);
                if (!File.Exists(candidate)) return candidate;
            }
        }
    }

    internal static class RuntimeFbxExporter
    {
        public static bool TryExport(RuntimePoseRecordingPackage pose,
            RecordedMotionEditDefinition definition, string path, out string error)
        {
            error = null;
#if UNITY_EDITOR || FBXSDK_RUNTIME
            try
            {
                using (FbxManager manager = FbxManager.Create())
                {
                    FbxIOSettings ioSettings = FbxIOSettings.Create(manager, Globals.IOSROOT);
                    manager.SetIOSettings(ioSettings);
                    using (FbxExporter exporter = FbxExporter.Create(manager, "SpineFlowRuntimeExporter"))
                    {
                        int format = manager.GetIOPluginRegistry()
                            .FindWriterIDByDescription("FBX binary (*.fbx)");
                        if (format < 0)
                            format = manager.GetIOPluginRegistry()
                                .FindWriterIDByDescription("FBX ascii (*.fbx)");
                        if (!exporter.Initialize(path, format, ioSettings))
                        {
                            error = "FBX 初始化失败：" + exporter.GetStatus().GetErrorString();
                            return false;
                        }

                        using (FbxScene scene = FbxScene.Create(manager, "SpineFlowMotion"))
                        {
                            scene.GetGlobalSettings().SetSystemUnit(FbxSystemUnit.m);
                            scene.GetGlobalSettings().SetAxisSystem(FbxAxisSystem.MayaYUp);
                            scene.GetGlobalSettings().SetTimeMode(GetNearestTimeMode(pose.sampleRate));
                            FbxNode[] nodes = CreateSkeleton(scene, pose);
                            FbxAnimStack firstStack = null;
                            for (int segmentIndex = 0;
                                 segmentIndex < definition.keyframes.Length - 1;
                                 segmentIndex++)
                            {
                                FbxAnimStack stack = AddAnimationStack(scene, nodes, pose,
                                    definition.keyframes[segmentIndex],
                                    definition.keyframes[segmentIndex + 1]);
                                if (firstStack == null) firstStack = stack;
                            }

                            if (firstStack != null) scene.SetCurrentAnimationStack(firstStack);
                            if (!exporter.Export(scene))
                            {
                                error = "FBX 写入失败：" + exporter.GetStatus().GetErrorString();
                                return false;
                            }
                        }
                    }
                }

                if (File.Exists(path) && new FileInfo(path).Length > 0) return true;
                error = "FBX 导出完成但没有生成有效文件";
                return false;
            }
            catch (Exception exception)
            {
                error = "FBX 导出失败：" + exception.Message;
                return false;
            }
#else
            error = "当前平台没有启用 FBXSDK_RUNTIME；请为 Standalone 添加该脚本宏";
            return false;
#endif
        }

#if UNITY_EDITOR || FBXSDK_RUNTIME
        private static FbxNode[] CreateSkeleton(FbxScene scene, RuntimePoseRecordingPackage pose)
        {
            var nodes = new FbxNode[pose.transformPaths.Length];
            var byPath = new Dictionary<string, FbxNode>(StringComparer.Ordinal);
            RuntimePoseFrame firstFrame = pose.frames[0];
            for (int index = 0; index < pose.transformPaths.Length; index++)
            {
                string transformPath = pose.transformPaths[index] ?? string.Empty;
                string nodeName = string.IsNullOrEmpty(transformPath)
                    ? (string.IsNullOrWhiteSpace(pose.avatarName) ? "Avatar" : pose.avatarName)
                    : Path.GetFileName(transformPath.Replace('/', Path.DirectorySeparatorChar));
                FbxNode node = FbxNode.Create(scene, nodeName);
                // Unity exposes localEulerAngles in Z-X-Y order. Declaring the
                // same order in FBX preserves the quaternion represented by the curves.
                node.SetRotationOrder(FbxNode.EPivotSet.eSourcePivot, FbxEuler.EOrder.eOrderZXY);
                var skeleton = FbxSkeleton.Create(scene, nodeName + "_Skeleton");
                skeleton.SetSkeletonType(string.IsNullOrEmpty(transformPath)
                    ? FbxSkeleton.EType.eRoot
                    : FbxSkeleton.EType.eLimbNode);
                node.SetNodeAttribute(skeleton);

                RuntimeTransformPose transform = firstFrame.transforms[index];
                Vector3 position = ConvertPosition(transform.localPosition);
                Vector3 rotation = ConvertRotation(transform.localRotation).eulerAngles;
                node.LclTranslation.Set(ToDouble3(position));
                node.LclRotation.Set(ToDouble3(rotation));
                node.LclScaling.Set(ToDouble3(transform.localScale));

                string parentPath = GetParentPath(transformPath);
                if (!string.IsNullOrEmpty(transformPath) && byPath.TryGetValue(parentPath, out FbxNode parent))
                    parent.AddChild(node);
                else
                    scene.GetRootNode().AddChild(node);

                nodes[index] = node;
                byPath[transformPath] = node;
            }

            return nodes;
        }

        private static FbxAnimStack AddAnimationStack(FbxScene scene, FbxNode[] nodes,
            RuntimePoseRecordingPackage pose, RecordedMotionKeyframeDefinition from,
            RecordedMotionKeyframeDefinition to)
        {
            FbxAnimStack stack = FbxAnimStack.Create(scene, from.followingSegmentName);
            FbxAnimLayer layer = FbxAnimLayer.Create(scene, "BaseLayer");
            stack.AddMember(layer);
            float sourceStart = pose.frames[from.frame].timeSeconds;
            float duration = Mathf.Max(0f, pose.frames[to.frame].timeSeconds - sourceStart);
            stack.SetLocalTimeSpan(new FbxTimeSpan(FbxTime.FromSecondDouble(0d),
                FbxTime.FromSecondDouble(duration)));

            for (int trackIndex = 0; trackIndex < nodes.Length; trackIndex++)
            {
                FbxNode node = nodes[trackIndex];
                FbxAnimCurve[] translation = CreateCurves(node.LclTranslation, layer);
                FbxAnimCurve[] rotation = CreateCurves(node.LclRotation, layer);
                FbxAnimCurve[] scale = CreateCurves(node.LclScaling, layer);
                BeginCurves(translation);
                BeginCurves(rotation);
                BeginCurves(scale);
                Vector3 previousEuler = Vector3.zero;
                bool hasPreviousEuler = false;

                for (int frameIndex = from.frame; frameIndex <= to.frame; frameIndex++)
                {
                    RuntimeTransformPose transform = pose.frames[frameIndex].transforms[trackIndex];
                    Vector3 position = ConvertPosition(transform.localPosition);
                    Vector3 euler = ConvertRotation(transform.localRotation).eulerAngles;
                    if (hasPreviousEuler) euler = UnwrapEuler(euler, previousEuler);
                    previousEuler = euler;
                    hasPreviousEuler = true;
                    FbxTime time = FbxTime.FromSecondDouble(
                        Mathf.Max(0f, pose.frames[frameIndex].timeSeconds - sourceStart));
                    AddVectorKey(translation, time, position);
                    AddVectorKey(rotation, time, euler);
                    AddVectorKey(scale, time, transform.localScale);
                }

                EndCurves(translation);
                EndCurves(rotation);
                EndCurves(scale);
            }

            return stack;
        }

        private static FbxAnimCurve[] CreateCurves(FbxPropertyDouble3 property, FbxAnimLayer layer)
        {
            return new[]
            {
                property.GetCurve(layer, Globals.FBXSDK_CURVENODE_COMPONENT_X, true),
                property.GetCurve(layer, Globals.FBXSDK_CURVENODE_COMPONENT_Y, true),
                property.GetCurve(layer, Globals.FBXSDK_CURVENODE_COMPONENT_Z, true)
            };
        }

        private static void BeginCurves(FbxAnimCurve[] curves)
        {
            foreach (FbxAnimCurve curve in curves) curve.KeyModifyBegin();
        }

        private static void EndCurves(FbxAnimCurve[] curves)
        {
            foreach (FbxAnimCurve curve in curves) curve.KeyModifyEnd();
        }

        private static void AddVectorKey(FbxAnimCurve[] curves, FbxTime time, Vector3 value)
        {
            float[] values = { value.x, value.y, value.z };
            for (int axis = 0; axis < curves.Length; axis++)
            {
                int keyIndex = curves[axis].KeyAdd(time);
                curves[axis].KeySet(keyIndex, time, values[axis],
                    FbxAnimCurveDef.EInterpolationType.eInterpolationLinear);
            }
        }

        private static Vector3 ConvertPosition(Vector3 value)
        {
            return new Vector3(-value.x, value.y, value.z);
        }

        private static Quaternion ConvertRotation(Quaternion value)
        {
            return new Quaternion(value.x, -value.y, -value.z, value.w).normalized;
        }

        private static Vector3 UnwrapEuler(Vector3 current, Vector3 previous)
        {
            current.x = UnwrapAngle(current.x, previous.x);
            current.y = UnwrapAngle(current.y, previous.y);
            current.z = UnwrapAngle(current.z, previous.z);
            return current;
        }

        private static float UnwrapAngle(float value, float previous)
        {
            while (value - previous > 180f) value -= 360f;
            while (value - previous < -180f) value += 360f;
            return value;
        }

        private static FbxDouble3 ToDouble3(Vector3 value)
        {
            return new FbxDouble3(value.x, value.y, value.z);
        }

        private static FbxTime.EMode GetNearestTimeMode(float sampleRate)
        {
            if (sampleRate >= 90f) return FbxTime.EMode.eFrames120;
            if (sampleRate >= 55f) return FbxTime.EMode.eFrames60;
            if (sampleRate >= 45f) return FbxTime.EMode.eFrames50;
            if (sampleRate >= 36f) return FbxTime.EMode.eFrames48;
            if (sampleRate >= 27f) return FbxTime.EMode.eFrames30;
            return FbxTime.EMode.eFrames24;
        }

        private static string GetParentPath(string path)
        {
            if (string.IsNullOrEmpty(path)) return string.Empty;
            int slash = path.LastIndexOf('/');
            return slash < 0 ? string.Empty : path.Substring(0, slash);
        }
#endif
    }
}
