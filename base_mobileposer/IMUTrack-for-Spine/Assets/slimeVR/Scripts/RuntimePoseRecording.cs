using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace SpineFlow.PoseRecording
{
    [Serializable]
    public sealed class RuntimePoseRecordingPackage
    {
        public const int CurrentFormatVersion = 1;

        public int formatVersion = CurrentFormatVersion;
        public string actionId;
        public string displayName;
        public string trajectoryName;
        public string recordedAtUtc;
        public string avatarName;
        public float sampleRate = 30f;
        public float durationSeconds;
        public int frameCount;
        public string[] transformPaths = Array.Empty<string>();
        public RuntimePoseFrame[] frames = Array.Empty<RuntimePoseFrame>();
    }

    [Serializable]
    public sealed class RuntimePoseFrame
    {
        public float timeSeconds;
        public Vector3 bodyPosition;
        public Quaternion bodyRotation = Quaternion.identity;
        public float[] muscles = Array.Empty<float>();
        public RuntimeTransformPose[] transforms = Array.Empty<RuntimeTransformPose>();
    }

    [Serializable]
    public sealed class RuntimeTransformPose
    {
        public Vector3 localPosition;
        public Quaternion localRotation = Quaternion.identity;
        public Vector3 localScale = Vector3.one;
    }

    /// <summary>
    /// Build-safe storage for the avatar pose stream. Unlike .anim assets, these
    /// files do not depend on AssetDatabase and can be recorded and loaded by a player.
    /// </summary>
    public static class RuntimePoseRecordingStorage
    {
        public const string FileExtension = ".pose.json";

        public static string RootDirectory
        {
            get
            {
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                return Path.Combine(localAppData, "SpineFlow", "PoseRecordings");
            }
        }

        public static bool TrySave(RuntimePoseRecordingPackage package, out string path, out string error)
        {
            path = null;
            error = null;
            if (!TryValidate(package, out error)) return false;

            try
            {
                Directory.CreateDirectory(RootDirectory);
                string trajectoryName = MakeSafeFileName(package.trajectoryName);
                if (string.IsNullOrWhiteSpace(trajectoryName))
                    trajectoryName = DateTime.Now.ToString("yyyyMMdd_HHmmss");

                string candidate = Path.Combine(RootDirectory, trajectoryName + FileExtension);
                for (int suffix = 2; File.Exists(candidate); suffix++)
                    candidate = Path.Combine(RootDirectory, trajectoryName + "_" + suffix + FileExtension);

                File.WriteAllText(candidate, JsonUtility.ToJson(package, false), new UTF8Encoding(false));
                path = Path.GetFullPath(candidate);
                return true;
            }
            catch (Exception exception)
            {
                error = "Could not save runtime pose recording: " + exception.Message;
                return false;
            }
        }

        public static bool TryLoad(string path, out RuntimePoseRecordingPackage package, out string error)
        {
            package = null;
            error = null;
            try
            {
                if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                {
                    error = "Pose recording does not exist: " + path;
                    return false;
                }

                package = JsonUtility.FromJson<RuntimePoseRecordingPackage>(
                    File.ReadAllText(path, Encoding.UTF8));
                if (TryValidate(package, out error)) return true;
                package = null;
                return false;
            }
            catch (Exception exception)
            {
                package = null;
                error = "Could not load runtime pose recording: " + exception.Message;
                return false;
            }
        }

        public static bool TryFindLatest(out string path)
        {
            path = null;
            if (!Directory.Exists(RootDirectory)) return false;

            DateTime latestTime = DateTime.MinValue;
            foreach (string candidate in Directory.GetFiles(RootDirectory, "*" + FileExtension,
                         SearchOption.TopDirectoryOnly))
            {
                DateTime writeTime = File.GetLastWriteTimeUtc(candidate);
                if (writeTime <= latestTime) continue;
                latestTime = writeTime;
                path = candidate;
            }

            return !string.IsNullOrWhiteSpace(path);
        }

        public static string GetKeyframesPath(string poseRecordingPath)
        {
            if (string.IsNullOrWhiteSpace(poseRecordingPath)) return null;
            return poseRecordingPath.EndsWith(FileExtension, StringComparison.OrdinalIgnoreCase)
                ? poseRecordingPath.Substring(0, poseRecordingPath.Length - FileExtension.Length) +
                  ".keyframes.json"
                : Path.ChangeExtension(poseRecordingPath, ".keyframes.json");
        }

        public static bool IsOwnedRecordingPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            try
            {
                string root = Path.GetFullPath(RootDirectory)
                    .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                              Path.DirectorySeparatorChar;
                string candidate = Path.GetFullPath(path);
                return candidate.StartsWith(root, StringComparison.OrdinalIgnoreCase) &&
                       candidate.EndsWith(FileExtension, StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return false;
            }
        }

        private static bool TryValidate(RuntimePoseRecordingPackage package, out string error)
        {
            if (package == null || package.formatVersion != RuntimePoseRecordingPackage.CurrentFormatVersion)
            {
                error = "Unsupported pose recording format.";
                return false;
            }

            if (package.frames == null || package.frames.Length == 0 ||
                package.frameCount != package.frames.Length)
            {
                error = "Pose recording has no frames or its frame count is invalid.";
                return false;
            }

            if (package.transformPaths == null) package.transformPaths = Array.Empty<string>();
            float previousTime = -0.0001f;
            for (int index = 0; index < package.frames.Length; index++)
            {
                RuntimePoseFrame frame = package.frames[index];
                if (frame == null || frame.timeSeconds < previousTime ||
                    frame.muscles == null || frame.muscles.Length != HumanTrait.MuscleCount ||
                    frame.transforms == null || frame.transforms.Length != package.transformPaths.Length)
                {
                    error = "Pose recording contains an invalid frame at index " + index + ".";
                    return false;
                }

                previousTime = frame.timeSeconds;
            }

            package.durationSeconds = Mathf.Max(package.durationSeconds,
                package.frames[package.frames.Length - 1].timeSeconds);
            package.sampleRate = Mathf.Max(1f, package.sampleRate);
            error = null;
            return true;
        }

        private static string MakeSafeFileName(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return string.Empty;
            value = value.Trim();
            foreach (char invalid in Path.GetInvalidFileNameChars()) value = value.Replace(invalid, '_');
            return value;
        }
    }

    public static class RuntimePosePlaybackUtility
    {
        public static bool IsValidHumanoid(Animator animator)
        {
            return animator != null && animator.avatar != null && animator.avatar.isValid &&
                   animator.avatar.isHuman;
        }

        public static void ApplyAtTime(RuntimePoseRecordingPackage recording, float timeSeconds,
            ref int lowerFrameIndex, HumanPoseHandler handler, ref HumanPose pose)
        {
            if (recording?.frames == null || recording.frames.Length == 0 || handler == null) return;

            RuntimePoseFrame[] frames = recording.frames;
            lowerFrameIndex = Mathf.Clamp(lowerFrameIndex, 0, frames.Length - 1);
            while (lowerFrameIndex + 1 < frames.Length &&
                   frames[lowerFrameIndex + 1].timeSeconds <= timeSeconds)
                lowerFrameIndex++;
            while (lowerFrameIndex > 0 && frames[lowerFrameIndex].timeSeconds > timeSeconds)
                lowerFrameIndex--;

            int upperFrameIndex = Mathf.Min(lowerFrameIndex + 1, frames.Length - 1);
            RuntimePoseFrame lower = frames[lowerFrameIndex];
            RuntimePoseFrame upper = frames[upperFrameIndex];
            float blend = upperFrameIndex == lowerFrameIndex
                ? 0f
                : Mathf.InverseLerp(lower.timeSeconds, upper.timeSeconds, timeSeconds);

            EnsureMuscles(ref pose);
            // The recorder viewport is a pose viewer, not a world-locomotion
            // camera. Keep the Humanoid body origin anchored to the first frame;
            // otherwise accumulated tracker/root correction can move an old
            // recording several metres and make it fly out of the viewport.
            pose.bodyPosition = GetPreviewBodyPosition(recording);
            pose.bodyRotation = Quaternion.Slerp(lower.bodyRotation, upper.bodyRotation, blend);
            for (int muscle = 0; muscle < HumanTrait.MuscleCount; muscle++)
                pose.muscles[muscle] = Mathf.Lerp(lower.muscles[muscle], upper.muscles[muscle], blend);
            handler.SetHumanPose(ref pose);
        }

        public static void ApplyFrame(RuntimePoseRecordingPackage recording, int frameIndex,
            HumanPoseHandler handler, ref HumanPose pose)
        {
            if (recording?.frames == null || recording.frames.Length == 0 || handler == null) return;
            RuntimePoseFrame frame = recording.frames[Mathf.Clamp(frameIndex, 0, recording.frames.Length - 1)];
            EnsureMuscles(ref pose);
            pose.bodyPosition = GetPreviewBodyPosition(recording);
            pose.bodyRotation = frame.bodyRotation;
            Array.Copy(frame.muscles, pose.muscles, HumanTrait.MuscleCount);
            handler.SetHumanPose(ref pose);
        }

        private static void EnsureMuscles(ref HumanPose pose)
        {
            if (pose.muscles == null || pose.muscles.Length != HumanTrait.MuscleCount)
                pose.muscles = new float[HumanTrait.MuscleCount];
        }

        private static Vector3 GetPreviewBodyPosition(RuntimePoseRecordingPackage recording)
        {
            Vector3 position = recording.frames[0].bodyPosition;
            return IsFinite(position) ? position : Vector3.zero;
        }

        private static bool IsFinite(Vector3 value)
        {
            return !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
                   !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
                   !float.IsNaN(value.z) && !float.IsInfinity(value.z);
        }
    }
}
