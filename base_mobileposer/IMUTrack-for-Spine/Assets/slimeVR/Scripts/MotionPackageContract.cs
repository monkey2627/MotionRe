using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using SpineFlow.RawImu;
using UnityEngine;

namespace SpineFlow.MotionPackages
{
    [Serializable]
    public sealed class MotionPackageManifest
    {
        public int formatVersion = 1;
        public string actionId;
        public string displayName;
        public int version = 1;
        public string createdAtUtc;
        public string fbxFile;
        public string motionJsonFile;
        public string trackerJsonFile;
        public string rawImuJsonFile;
        public string keyframesFile;
        public string fbxSha256;
        public string motionJsonSha256;
        public string trackerJsonSha256;
        public string rawImuJsonSha256;
        public string keyframesSha256;
        public int rawImuFrameCount;
        public int rawImuSensorCount;
        public MotionPackageSegmentManifest[] segments = Array.Empty<MotionPackageSegmentManifest>();
    }

    [Serializable]
    public sealed class MotionPackageSegmentManifest
    {
        public string name;
        public int firstFrame;
        public int lastFrame;
        public float durationSeconds;
    }

    /// <summary>
    /// Shared on-disk contract used by IMUTrack and SpineFlowAdmin.
    /// In a player build, packages are stored in a Recordings directory beside
    /// the executable. In the Editor the equivalent location is the project root.
    /// </summary>
    public static class MotionPackageStorage
    {
        public const int CurrentFormatVersion = 1;
        public const string ManifestFileName = "manifest.json";

        public static string RootDirectory
        {
            get
            {
                string applicationDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
                return Path.Combine(applicationDirectory, "Recordings");
            }
        }

        public static bool IsValidIdentifier(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return false;

            foreach (char character in value)
            {
                bool lowerLetter = character >= 'a' && character <= 'z';
                bool digit = character >= '0' && character <= '9';
                if (!lowerLetter && !digit && character != '_') return false;
            }

            return true;
        }

        public static string GetPackageDirectory(string actionId, int version)
        {
            return GetPackageDirectory(actionId, actionId, version);
        }

        public static string GetPackageDirectory(string actionId, string displayName, int version)
        {
            if (!IsValidIdentifier(actionId))
                throw new ArgumentException("Invalid actionId.", nameof(actionId));
            if (version <= 0)
                throw new ArgumentOutOfRangeException(nameof(version));

            return Path.Combine(RootDirectory, GetPackageFolderName(actionId, displayName),
                version.ToString());
        }

        public static string GetPackageFolderName(string actionId, string displayName)
        {
            if (!IsValidIdentifier(actionId))
                throw new ArgumentException("Invalid actionId.", nameof(actionId));

            string safeDisplayName = MakeSafeDirectoryName(displayName);
            if (string.IsNullOrWhiteSpace(safeDisplayName) ||
                string.Equals(actionId, safeDisplayName, StringComparison.OrdinalIgnoreCase))
                return actionId;
            return actionId + "_" + safeDisplayName;
        }

        public static string GetRelativeManifestReference(string actionId, int version)
        {
            if (!IsValidIdentifier(actionId) || version <= 0) return string.Empty;
            return actionId + "/" + version + "/" + ManifestFileName;
        }

        public static string GetRelativeManifestReference(string actionId, string displayName, int version)
        {
            if (!IsValidIdentifier(actionId) || version <= 0) return string.Empty;
            return GetPackageFolderName(actionId, displayName) + "/" + version + "/" + ManifestFileName;
        }

        public static bool TryResolveManifestReference(string manifestReference, out string manifestPath,
            out string error)
        {
            manifestPath = null;
            error = null;
            if (string.IsNullOrWhiteSpace(manifestReference))
            {
                error = "Motion package manifest reference is empty.";
                return false;
            }

            try
            {
                string candidate = Path.IsPathRooted(manifestReference)
                    ? manifestReference
                    : Path.Combine(RootDirectory,
                        manifestReference.Replace('/', Path.DirectorySeparatorChar));
                candidate = Path.GetFullPath(candidate);
                if (!IsPathInsideRoot(candidate))
                {
                    error = "Motion package manifest must be inside the shared SpineFlow package directory.";
                    return false;
                }

                manifestPath = candidate;
                return true;
            }
            catch (Exception exception)
            {
                error = "Invalid motion package manifest path: " + exception.Message;
                return false;
            }
        }

        public static bool TryFindLatestManifest(string actionId, out string manifestPath)
        {
            manifestPath = null;
            if (!IsValidIdentifier(actionId) || !Directory.Exists(RootDirectory)) return false;

            DateTime latestWriteTime = DateTime.MinValue;
            foreach (string actionDirectory in Directory.GetDirectories(RootDirectory))
            {
                foreach (string versionDirectory in Directory.GetDirectories(actionDirectory))
                {
                    string candidate = Path.Combine(versionDirectory, ManifestFileName);
                    if (!File.Exists(candidate)) continue;

                    try
                    {
                        MotionPackageManifest candidateManifest = JsonUtility.FromJson<MotionPackageManifest>(
                            File.ReadAllText(candidate, Encoding.UTF8));
                        if (candidateManifest == null ||
                            !string.Equals(candidateManifest.actionId, actionId, StringComparison.Ordinal))
                            continue;

                        DateTime writeTime = File.GetLastWriteTimeUtc(candidate);
                        if (writeTime <= latestWriteTime) continue;
                        latestWriteTime = writeTime;
                        manifestPath = candidate;
                    }
                    catch
                    {
                        // Ignore incomplete or unrelated folders in Recordings.
                    }
                }
            }

            return manifestPath != null;
        }

        public static bool TryLoadAndValidate(string manifestPath, out MotionPackageManifest manifest,
            out string fbxPath, out string motionJsonPath, out string error)
        {
            manifest = null;
            fbxPath = null;
            motionJsonPath = null;
            error = null;

            try
            {
                manifestPath = Path.GetFullPath(manifestPath ?? string.Empty);
                if (!IsPathInsideRoot(manifestPath))
                {
                    error = "Motion package manifest is outside the shared package directory.";
                    return false;
                }

                if (!File.Exists(manifestPath))
                {
                    error = "Motion package manifest does not exist: " + manifestPath;
                    return false;
                }

                manifest = JsonUtility.FromJson<MotionPackageManifest>(
                    File.ReadAllText(manifestPath, Encoding.UTF8));
                if (!ValidateManifest(manifest, out error)) return false;

                string expectedDirectory = GetPackageDirectory(
                    manifest.actionId, manifest.displayName, manifest.version);
                string actualDirectory = Path.GetDirectoryName(manifestPath);
                if (!string.Equals(Path.GetFullPath(expectedDirectory), Path.GetFullPath(actualDirectory ?? string.Empty),
                        StringComparison.OrdinalIgnoreCase))
                {
                    error = "Manifest actionId/version does not match its directory.";
                    return false;
                }

                if (!TryResolvePackageFile(manifestPath, manifest.fbxFile, out fbxPath, out error) ||
                    !TryResolvePackageFile(manifestPath, manifest.motionJsonFile, out motionJsonPath, out error))
                    return false;

                if (!File.Exists(fbxPath) || !File.Exists(motionJsonPath))
                {
                    error = "Motion package FBX or JSON file is missing.";
                    return false;
                }

                if (!HashMatches(fbxPath, manifest.fbxSha256) ||
                    !HashMatches(motionJsonPath, manifest.motionJsonSha256))
                {
                    error = "Motion package checksum validation failed.";
                    return false;
                }

                if (!string.IsNullOrWhiteSpace(manifest.trackerJsonFile))
                {
                    if (!TryResolvePackageFile(manifestPath, manifest.trackerJsonFile,
                            out string trackerJsonPath, out error) || !File.Exists(trackerJsonPath) ||
                        !HashMatches(trackerJsonPath, manifest.trackerJsonSha256))
                    {
                        error = "Motion package Tracker JSON validation failed.";
                        return false;
                    }
                }

                if (!string.IsNullOrWhiteSpace(manifest.rawImuJsonFile))
                {
                    if (!TryResolvePackageFile(manifestPath, manifest.rawImuJsonFile,
                            out string rawImuJsonPath, out error) || !File.Exists(rawImuJsonPath) ||
                        !HashMatches(rawImuJsonPath, manifest.rawImuJsonSha256))
                    {
                        error = "Motion package Raw IMU JSON validation failed.";
                        return false;
                    }

                    if (!RawImuRecordingStorage.TryLoad(rawImuJsonPath,
                            out RawImuRecordingPackage rawImuPackage, out error) ||
                        !string.Equals(rawImuPackage.actionId, manifest.actionId, StringComparison.Ordinal) ||
                        rawImuPackage.frameCount != manifest.rawImuFrameCount ||
                        rawImuPackage.sensorIds.Length != manifest.rawImuSensorCount)
                    {
                        error = "Motion package Raw IMU JSON metadata does not match the manifest.";
                        return false;
                    }
                }

                if (!string.IsNullOrWhiteSpace(manifest.keyframesFile))
                {
                    if (!TryResolvePackageFile(manifestPath, manifest.keyframesFile,
                            out string keyframesPath, out error) || !File.Exists(keyframesPath) ||
                        !HashMatches(keyframesPath, manifest.keyframesSha256))
                    {
                        error = "Motion package keyframe definition validation failed.";
                        return false;
                    }
                }

                return true;
            }
            catch (Exception exception)
            {
                error = "Motion package validation failed: " + exception.Message;
                return false;
            }
        }

        public static bool TryCreateVersionedPackage(string sourceFbxPath, string sourceMotionJsonPath,
            string actionId, string displayName, MotionPackageSegmentManifest[] segments,
            out MotionPackageManifest manifest, out string manifestPath, out string error)
        {
            return TryCreateVersionedPackage(sourceFbxPath, sourceMotionJsonPath, null,
                actionId, displayName, segments, out manifest, out manifestPath, out error);
        }

        public static bool TryCreateVersionedPackage(string sourceFbxPath, string sourceMotionJsonPath,
            string sourceTrackerJsonPath, string actionId, string displayName,
            MotionPackageSegmentManifest[] segments,
            out MotionPackageManifest manifest, out string manifestPath, out string error)
        {
            return TryCreateVersionedPackage(sourceFbxPath, sourceMotionJsonPath, sourceTrackerJsonPath, null,
                actionId, displayName, segments, out manifest, out manifestPath, out error);
        }

        public static bool TryCreateVersionedPackage(string sourceFbxPath, string sourceMotionJsonPath,
            string sourceTrackerJsonPath, string sourceRawImuJsonPath, string actionId, string displayName,
            MotionPackageSegmentManifest[] segments,
            out MotionPackageManifest manifest, out string manifestPath, out string error)
        {
            return TryCreateVersionedPackage(sourceFbxPath, sourceMotionJsonPath, sourceTrackerJsonPath,
                sourceRawImuJsonPath, null, actionId, displayName, segments,
                out manifest, out manifestPath, out error);
        }

        public static bool TryCreateVersionedPackage(string sourceFbxPath, string sourceMotionJsonPath,
            string sourceTrackerJsonPath, string sourceRawImuJsonPath, string sourceKeyframesPath,
            string actionId, string displayName, MotionPackageSegmentManifest[] segments,
            out MotionPackageManifest manifest, out string manifestPath, out string error)
        {
            manifest = null;
            manifestPath = null;
            error = null;
            string stagingDirectory = null;

            try
            {
                if (!IsValidIdentifier(actionId))
                {
                    error = "actionId may contain only lowercase letters, digits, and underscores.";
                    return false;
                }

                if (!File.Exists(sourceFbxPath) || !File.Exists(sourceMotionJsonPath))
                {
                    error = "Source FBX or motion JSON does not exist.";
                    return false;
                }

                if (segments == null || segments.Length == 0)
                {
                    error = "A motion package must contain at least one segment.";
                    return false;
                }

                int version = GetNextVersion(actionId, displayName);
                string actionDirectory = Path.Combine(
                    RootDirectory, GetPackageFolderName(actionId, displayName));
                string finalDirectory = GetPackageDirectory(actionId, displayName, version);
                Directory.CreateDirectory(actionDirectory);
                stagingDirectory = Path.Combine(actionDirectory, "." + version + "_" + Guid.NewGuid().ToString("N") + ".tmp");
                Directory.CreateDirectory(stagingDirectory);

                string fbxFile = actionId + "_v" + version + ".fbx";
                string jsonFile = actionId + "_v" + version + ".motion.json";
                string stagedFbxPath = Path.Combine(stagingDirectory, fbxFile);
                string stagedJsonPath = Path.Combine(stagingDirectory, jsonFile);
                File.Copy(sourceFbxPath, stagedFbxPath, false);
                File.Copy(sourceMotionJsonPath, stagedJsonPath, false);

                string trackerJsonFile = null;
                string stagedTrackerJsonPath = null;
                if (!string.IsNullOrWhiteSpace(sourceTrackerJsonPath))
                {
                    if (!File.Exists(sourceTrackerJsonPath))
                    {
                        error = "Source Tracker JSON does not exist.";
                        return false;
                    }

                    trackerJsonFile = actionId + "_v" + version + ".tracker.json";
                    stagedTrackerJsonPath = Path.Combine(stagingDirectory, trackerJsonFile);
                    File.Copy(sourceTrackerJsonPath, stagedTrackerJsonPath, false);
                }

                string rawImuJsonFile = null;
                string stagedRawImuJsonPath = null;
                RawImuRecordingPackage rawImuPackage = null;
                if (!string.IsNullOrWhiteSpace(sourceRawImuJsonPath))
                {
                    if (!RawImuRecordingStorage.TryLoad(sourceRawImuJsonPath, out rawImuPackage,
                            out string rawImuError))
                    {
                        error = "Source Raw IMU JSON is invalid: " + rawImuError;
                        return false;
                    }

                    if (!string.Equals(rawImuPackage.actionId, actionId, StringComparison.Ordinal))
                    {
                        error = "Source Raw IMU JSON actionId does not match the motion package actionId.";
                        return false;
                    }

                    rawImuJsonFile = actionId + "_v" + version + ".raw-imu.json";
                    stagedRawImuJsonPath = Path.Combine(stagingDirectory, rawImuJsonFile);
                    File.Copy(sourceRawImuJsonPath, stagedRawImuJsonPath, false);
                }

                string keyframesFile = null;
                string stagedKeyframesPath = null;
                if (!string.IsNullOrWhiteSpace(sourceKeyframesPath))
                {
                    if (!File.Exists(sourceKeyframesPath))
                    {
                        error = "Source keyframe definition does not exist.";
                        return false;
                    }

                    keyframesFile = actionId + "_v" + version + ".keyframes.json";
                    stagedKeyframesPath = Path.Combine(stagingDirectory, keyframesFile);
                    File.Copy(sourceKeyframesPath, stagedKeyframesPath, false);
                }

                manifest = new MotionPackageManifest
                {
                    formatVersion = CurrentFormatVersion,
                    actionId = actionId,
                    displayName = string.IsNullOrWhiteSpace(displayName) ? actionId : displayName,
                    version = version,
                    createdAtUtc = DateTime.UtcNow.ToString("o"),
                    fbxFile = fbxFile,
                    motionJsonFile = jsonFile,
                    trackerJsonFile = trackerJsonFile,
                    rawImuJsonFile = rawImuJsonFile,
                    keyframesFile = keyframesFile,
                    fbxSha256 = ComputeSha256(stagedFbxPath),
                    motionJsonSha256 = ComputeSha256(stagedJsonPath),
                    trackerJsonSha256 = stagedTrackerJsonPath == null
                        ? null
                        : ComputeSha256(stagedTrackerJsonPath),
                    rawImuJsonSha256 = stagedRawImuJsonPath == null
                        ? null
                        : ComputeSha256(stagedRawImuJsonPath),
                    keyframesSha256 = stagedKeyframesPath == null
                        ? null
                        : ComputeSha256(stagedKeyframesPath),
                    rawImuFrameCount = rawImuPackage?.frameCount ?? 0,
                    rawImuSensorCount = rawImuPackage?.sensorIds?.Length ?? 0,
                    segments = segments
                };

                if (!ValidateManifest(manifest, out error)) return false;

                string stagedManifestPath = Path.Combine(stagingDirectory, ManifestFileName);
                File.WriteAllText(stagedManifestPath, JsonUtility.ToJson(manifest, true),
                    new UTF8Encoding(false));

                Directory.Move(stagingDirectory, finalDirectory);
                stagingDirectory = null;
                manifestPath = Path.Combine(finalDirectory, ManifestFileName);
                return TryLoadAndValidate(manifestPath, out manifest, out _, out _, out error);
            }
            catch (Exception exception)
            {
                error = "Could not create motion package: " + exception.Message;
                manifest = null;
                manifestPath = null;
                return false;
            }
            finally
            {
                if (!string.IsNullOrEmpty(stagingDirectory) && Directory.Exists(stagingDirectory))
                    Directory.Delete(stagingDirectory, true);
            }
        }

        public static string ComputeSha256(string filePath)
        {
            using (SHA256 sha256 = SHA256.Create())
            using (FileStream stream = File.OpenRead(filePath))
            {
                byte[] hash = sha256.ComputeHash(stream);
                var builder = new StringBuilder(hash.Length * 2);
                foreach (byte value in hash) builder.Append(value.ToString("x2"));
                return builder.ToString();
            }
        }

        private static int GetNextVersion(string actionId, string displayName)
        {
            string actionDirectory = Path.Combine(
                RootDirectory, GetPackageFolderName(actionId, displayName));
            if (!Directory.Exists(actionDirectory)) return 1;

            int highest = 0;
            foreach (string directory in Directory.GetDirectories(actionDirectory))
            {
                if (int.TryParse(Path.GetFileName(directory), out int version) && version > highest)
                    highest = version;
            }

            return highest + 1;
        }

        private static string MakeSafeDirectoryName(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return string.Empty;

            string result = value.Trim();
            foreach (char invalid in Path.GetInvalidFileNameChars())
                result = result.Replace(invalid, '_');
            return result.TrimEnd('.', ' ');
        }

        private static bool ValidateManifest(MotionPackageManifest manifest, out string error)
        {
            if (manifest == null || manifest.formatVersion != CurrentFormatVersion)
            {
                error = "Unsupported motion package manifest format.";
                return false;
            }

            if (!IsValidIdentifier(manifest.actionId) || manifest.version <= 0)
            {
                error = "Manifest actionId or version is invalid.";
                return false;
            }

            if (!IsLeafFileName(manifest.fbxFile) || !IsLeafFileName(manifest.motionJsonFile))
            {
                error = "Manifest package file names must not contain a directory.";
                return false;
            }


            bool hasTrackerJson = !string.IsNullOrWhiteSpace(manifest.trackerJsonFile);
            if (hasTrackerJson != !string.IsNullOrWhiteSpace(manifest.trackerJsonSha256) ||
                hasTrackerJson && (!IsLeafFileName(manifest.trackerJsonFile) ||
                                   !IsSha256(manifest.trackerJsonSha256)))
            {
                error = "Manifest Tracker JSON fields are invalid.";
                return false;
            }

            bool hasRawImuJson = !string.IsNullOrWhiteSpace(manifest.rawImuJsonFile);
            if (hasRawImuJson != !string.IsNullOrWhiteSpace(manifest.rawImuJsonSha256) ||
                hasRawImuJson && (!IsLeafFileName(manifest.rawImuJsonFile) ||
                                  !IsSha256(manifest.rawImuJsonSha256) ||
                                  manifest.rawImuFrameCount <= 0 || manifest.rawImuSensorCount <= 0) ||
                !hasRawImuJson && (manifest.rawImuFrameCount != 0 || manifest.rawImuSensorCount != 0))
            {
                error = "Manifest Raw IMU JSON fields are invalid.";
                return false;
            }

            bool hasKeyframes = !string.IsNullOrWhiteSpace(manifest.keyframesFile);
            if (hasKeyframes != !string.IsNullOrWhiteSpace(manifest.keyframesSha256) ||
                hasKeyframes && (!IsLeafFileName(manifest.keyframesFile) ||
                                 !IsSha256(manifest.keyframesSha256)))
            {
                error = "Manifest keyframe definition fields are invalid.";
                return false;
            }

            if (!IsSha256(manifest.fbxSha256) || !IsSha256(manifest.motionJsonSha256))
            {
                error = "Manifest checksums are invalid.";
                return false;
            }

            if (manifest.segments == null || manifest.segments.Length == 0)
            {
                error = "Manifest has no motion segments.";
                return false;
            }

            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (MotionPackageSegmentManifest segment in manifest.segments)
            {
                if (segment == null || !IsValidIdentifier(segment.name) || !names.Add(segment.name) ||
                    segment.firstFrame < 0 || segment.lastFrame <= segment.firstFrame ||
                    segment.durationSeconds <= 0f)
                {
                    error = "Manifest contains an invalid or duplicate motion segment.";
                    return false;
                }
            }

            error = null;
            return true;
        }

        private static bool TryResolvePackageFile(string manifestPath, string fileName, out string filePath,
            out string error)
        {
            filePath = null;
            if (!IsLeafFileName(fileName))
            {
                error = "Invalid file name in motion package manifest.";
                return false;
            }

            filePath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(manifestPath) ?? string.Empty, fileName));
            if (!IsPathInsideRoot(filePath))
            {
                error = "Motion package file is outside the shared package directory.";
                return false;
            }

            error = null;
            return true;
        }

        private static bool IsLeafFileName(string value)
        {
            return !string.IsNullOrWhiteSpace(value) &&
                   string.Equals(Path.GetFileName(value), value, StringComparison.Ordinal);
        }

        private static bool IsPathInsideRoot(string path)
        {
            string root = Path.GetFullPath(RootDirectory)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            string candidate = Path.GetFullPath(path);
            return candidate.StartsWith(root, StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsSha256(string value)
        {
            if (string.IsNullOrWhiteSpace(value) || value.Length != 64) return false;
            foreach (char character in value)
            {
                bool digit = character >= '0' && character <= '9';
                bool lower = character >= 'a' && character <= 'f';
                bool upper = character >= 'A' && character <= 'F';
                if (!digit && !lower && !upper) return false;
            }

            return true;
        }

        private static bool HashMatches(string path, string expected)
        {
            return string.Equals(ComputeSha256(path), expected, StringComparison.OrdinalIgnoreCase);
        }
    }
}
