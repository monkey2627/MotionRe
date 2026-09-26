using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace SpineFlow.TrackerRecording
{
    [Serializable]
    public sealed class CoachProfileData
    {
        public float heightCm;
        public float weightKg;
        public string sex = "unspecified";
    }

    [Serializable]
    public sealed class TrackerPoseData
    {
        public string name;
        public Vector3 position;
        public Quaternion rotation = Quaternion.identity;

        public TrackerPoseData Clone()
        {
            return new TrackerPoseData
            {
                name = name,
                position = position,
                rotation = rotation
            };
        }
    }

    [Serializable]
    public sealed class TrackerRecordingFrame
    {
        public float timeSeconds;
        public TrackerPoseData[] trackers = Array.Empty<TrackerPoseData>();
    }

    [Serializable]
    public sealed class TrackerRecordingPackage
    {
        public int formatVersion = TrackerRecordingStorage.CurrentFormatVersion;
        public string actionId;
        public string displayName;
        public string trajectoryName;
        public string recordedAtUtc;
        public string source = "SlimeVR VMC /VMC/Ext/Tra/Pos";
        public string coordinateSpace = "VMC Unity coordinates";
        public float sampleRate;
        public float durationSeconds;
        public int frameCount;
        public string[] trackerNames = Array.Empty<string>();
        public CoachProfileData coach = new CoachProfileData();
        public TrackerRecordingFrame[] frames = Array.Empty<TrackerRecordingFrame>();
    }

    public static class TrackerRecordingStorage
    {
        public const int CurrentFormatVersion = 1;

        public static string RootDirectory
        {
            get
            {
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                return Path.Combine(localAppData, "SpineFlow", "TrackerRecordings");
            }
        }

        public static bool TrySave(TrackerRecordingPackage package, out string filePath, out string error)
        {
            filePath = null;
            error = null;

            if (!Validate(package, out error)) return false;

            try
            {
                Directory.CreateDirectory(RootDirectory);
                string baseName = SanitizeFileName(package.trajectoryName);
                if (string.IsNullOrEmpty(baseName)) baseName = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                filePath = GetUniquePath(Path.Combine(RootDirectory, baseName + ".tracker.json"));
                File.WriteAllText(filePath, JsonUtility.ToJson(package, true), new UTF8Encoding(false));
                return true;
            }
            catch (Exception exception)
            {
                error = "Tracker JSON 写入失败：" + exception.Message;
                filePath = null;
                return false;
            }
        }

        public static bool TryLoad(string filePath, out TrackerRecordingPackage package, out string error)
        {
            package = null;
            error = null;

            try
            {
                if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
                {
                    error = "Tracker JSON 文件不存在";
                    return false;
                }

                package = JsonUtility.FromJson<TrackerRecordingPackage>(
                    File.ReadAllText(filePath, Encoding.UTF8));
                if (!Validate(package, out error))
                {
                    package = null;
                    return false;
                }

                return true;
            }
            catch (Exception exception)
            {
                error = "Tracker JSON 读取失败：" + exception.Message;
                package = null;
                return false;
            }
        }

        public static bool TryFindLatest(out string filePath)
        {
            filePath = null;
            if (!Directory.Exists(RootDirectory)) return false;

            FileInfo latest = new DirectoryInfo(RootDirectory)
                .GetFiles("*.tracker.json", SearchOption.TopDirectoryOnly)
                .OrderByDescending(file => file.LastWriteTimeUtc)
                .FirstOrDefault();
            if (latest == null) return false;

            filePath = latest.FullName;
            return true;
        }

        public static bool Validate(TrackerRecordingPackage package, out string error)
        {
            if (package == null || package.formatVersion != CurrentFormatVersion)
            {
                error = "不支持的 Tracker JSON 格式版本";
                return false;
            }

            if (package.frames == null || package.frames.Length == 0)
            {
                error = "Tracker JSON 没有姿态帧";
                return false;
            }

            float previousTime = -1f;
            var allNames = new HashSet<string>(StringComparer.Ordinal);
            foreach (TrackerRecordingFrame frame in package.frames)
            {
                if (frame == null || !IsFinite(frame.timeSeconds) || frame.timeSeconds < previousTime)
                {
                    error = "Tracker JSON 包含无效或乱序的时间戳";
                    return false;
                }

                previousTime = frame.timeSeconds;
                if (frame.trackers == null) continue;

                var frameNames = new HashSet<string>(StringComparer.Ordinal);
                foreach (TrackerPoseData tracker in frame.trackers)
                {
                    if (tracker == null || string.IsNullOrWhiteSpace(tracker.name) ||
                        !frameNames.Add(tracker.name) || !IsFinite(tracker.position) ||
                        !IsFinite(tracker.rotation))
                    {
                        error = "Tracker JSON 包含无效或重复的 Tracker 姿态";
                        return false;
                    }

                    allNames.Add(tracker.name);
                }
            }

            if (allNames.Count == 0)
            {
                error = "Tracker JSON 中没有任何 Tracker 数据";
                return false;
            }

            error = null;
            return true;
        }

        private static string GetUniquePath(string desiredPath)
        {
            if (!File.Exists(desiredPath)) return desiredPath;

            string directory = Path.GetDirectoryName(desiredPath) ?? RootDirectory;
            string extension = ".tracker.json";
            string fileName = Path.GetFileName(desiredPath);
            string stem = fileName.EndsWith(extension, StringComparison.OrdinalIgnoreCase)
                ? fileName.Substring(0, fileName.Length - extension.Length)
                : Path.GetFileNameWithoutExtension(fileName);

            for (int index = 2; index < int.MaxValue; index++)
            {
                string candidate = Path.Combine(directory, stem + "_" + index + extension);
                if (!File.Exists(candidate)) return candidate;
            }

            return Path.Combine(directory, stem + "_" + Guid.NewGuid().ToString("N") + extension);
        }

        private static string SanitizeFileName(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return string.Empty;
            var invalid = new HashSet<char>(Path.GetInvalidFileNameChars());
            var builder = new StringBuilder(value.Length);
            foreach (char character in value.Trim())
            {
                if (!invalid.Contains(character)) builder.Append(character);
            }

            return builder.ToString();
        }

        private static bool IsFinite(Vector3 value)
        {
            return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
        }

        private static bool IsFinite(Quaternion value)
        {
            return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z) && IsFinite(value.w);
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }
    }
}
