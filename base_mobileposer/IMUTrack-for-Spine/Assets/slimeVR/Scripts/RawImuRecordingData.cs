using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace SpineFlow.RawImu
{
    [Serializable]
    public sealed class RawImuVector3
    {
        public double x;
        public double y;
        public double z;

        public RawImuVector3()
        {
        }

        public RawImuVector3(double x, double y, double z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public RawImuVector3 Clone()
        {
            return new RawImuVector3(x, y, z);
        }
    }

    [Serializable]
    public sealed class RawImuQuaternion
    {
        public double x;
        public double y;
        public double z;
        public double w = 1d;

        public RawImuQuaternion()
        {
        }

        public RawImuQuaternion(double x, double y, double z, double w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        public RawImuQuaternion Clone()
        {
            return new RawImuQuaternion(x, y, z, w);
        }
    }

    [Serializable]
    public sealed class RawImuCoachProfile
    {
        public float heightCm;
        public float weightKg;
        public string sex = "unspecified";
    }

    [Serializable]
    public sealed class RawImuSensorSample
    {
        public string sensorId;
        public string trackerRole;
        public bool online;
        public string deviceUpdatedAtUtc;
        public string source = "WitMotion";
        public string imuType;
        public string accelerationKind = "raw";
        public bool hasAcceleration = true;
        public bool hasAngularVelocity = true;
        public bool hasMagneticField = true;
        public bool hasEulerAngles = true;
        // Added in format v2. Keeping Euler angles makes old recordings readable,
        // while the fused quaternion avoids an Euler round trip during reconstruction.
        public bool hasOrientation;
        // Added in format v3. These SolarXR rotations preserve the SlimeVR
        // reference/reset state that cannot be recovered from Rotation alone.
        public bool hasRotationReferenceAdjusted;
        public bool hasRotationIdentityAdjusted;
        public bool hasMountingOrientation;
        public bool hasMountingResetOrientation;
        public bool hasBattery = true;
        public int ticksPerSecond;
        public RawImuVector3 accelerationG = new RawImuVector3();
        public RawImuVector3 angularVelocityDegPerSec = new RawImuVector3();
        public RawImuVector3 magneticFieldMicroTesla = new RawImuVector3();
        public RawImuVector3 eulerAnglesDeg = new RawImuVector3();
        public RawImuQuaternion orientation = new RawImuQuaternion();
        public RawImuQuaternion rotationReferenceAdjusted = new RawImuQuaternion();
        public RawImuQuaternion rotationIdentityAdjusted = new RawImuQuaternion();
        public RawImuQuaternion mountingOrientation = new RawImuQuaternion();
        public RawImuQuaternion mountingResetOrientation = new RawImuQuaternion();
        public int batteryPercent;

        public RawImuSensorSample Clone()
        {
            return new RawImuSensorSample
            {
                sensorId = sensorId,
                trackerRole = trackerRole,
                online = online,
                deviceUpdatedAtUtc = deviceUpdatedAtUtc,
                source = source,
                imuType = imuType,
                accelerationKind = accelerationKind,
                hasAcceleration = hasAcceleration,
                hasAngularVelocity = hasAngularVelocity,
                hasMagneticField = hasMagneticField,
                hasEulerAngles = hasEulerAngles,
                hasOrientation = hasOrientation,
                hasRotationReferenceAdjusted = hasRotationReferenceAdjusted,
                hasRotationIdentityAdjusted = hasRotationIdentityAdjusted,
                hasMountingOrientation = hasMountingOrientation,
                hasMountingResetOrientation = hasMountingResetOrientation,
                hasBattery = hasBattery,
                ticksPerSecond = ticksPerSecond,
                accelerationG = accelerationG?.Clone() ?? new RawImuVector3(),
                angularVelocityDegPerSec = angularVelocityDegPerSec?.Clone() ?? new RawImuVector3(),
                magneticFieldMicroTesla = magneticFieldMicroTesla?.Clone() ?? new RawImuVector3(),
                eulerAnglesDeg = eulerAnglesDeg?.Clone() ?? new RawImuVector3(),
                orientation = orientation?.Clone() ?? new RawImuQuaternion(),
                rotationReferenceAdjusted = rotationReferenceAdjusted?.Clone() ?? new RawImuQuaternion(),
                rotationIdentityAdjusted = rotationIdentityAdjusted?.Clone() ?? new RawImuQuaternion(),
                mountingOrientation = mountingOrientation?.Clone() ?? new RawImuQuaternion(),
                mountingResetOrientation = mountingResetOrientation?.Clone() ?? new RawImuQuaternion(),
                batteryPercent = batteryPercent
            };
        }
    }

    [Serializable]
    public sealed class RawImuCalibration
    {
        public string sensorId;
        public string capturedAtUtc;
        public RawImuVector3 eulerBaselineDeg = new RawImuVector3();
        public bool hasOrientationBaseline;
        public RawImuQuaternion orientationBaseline = new RawImuQuaternion();
    }

    [Serializable]
    public sealed class RawImuFrame
    {
        public float timeSeconds;
        public string capturedAtUtc;
        public RawImuSensorSample[] sensors = Array.Empty<RawImuSensorSample>();
    }

    [Serializable]
    public sealed class RawImuRecordingPackage
    {
        public int formatVersion = RawImuRecordingStorage.CurrentFormatVersion;
        public string actionId;
        public string displayName;
        public string trajectoryName;
        public string recordedAtUtc;
        public string source = "Direct physical IMU tracker data";
        public float sampleRate;
        public float durationSeconds;
        public int frameCount;
        public string[] sensorIds = Array.Empty<string>();
        public RawImuCoachProfile coach = new RawImuCoachProfile();
        public RawImuCalibration[] calibrations = Array.Empty<RawImuCalibration>();
        public RawImuFrame[] frames = Array.Empty<RawImuFrame>();
    }

    public static class RawImuRecordingStorage
    {
        public const int CurrentFormatVersion = 3;
        public const string FileExtension = ".raw-imu.json";

        public static string RootDirectory
        {
            get
            {
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                return Path.Combine(localAppData, "SpineFlow", "RawImuRecordings");
            }
        }

        public static bool TrySave(RawImuRecordingPackage package, out string filePath, out string error)
        {
            filePath = null;
            if (!Validate(package, out error)) return false;

            try
            {
                Directory.CreateDirectory(RootDirectory);
                string baseName = SanitizeFileName(package.trajectoryName);
                if (string.IsNullOrWhiteSpace(baseName)) baseName = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                filePath = GetUniquePath(Path.Combine(RootDirectory, baseName + FileExtension));
                File.WriteAllText(filePath, JsonUtility.ToJson(package, true), new UTF8Encoding(false));
                return true;
            }
            catch (Exception exception)
            {
                error = "Raw IMU JSON write failed: " + exception.Message;
                filePath = null;
                return false;
            }
        }

        public static bool TryLoad(string filePath, out RawImuRecordingPackage package, out string error)
        {
            package = null;
            error = null;

            try
            {
                if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
                {
                    error = "Raw IMU JSON file does not exist.";
                    return false;
                }

                package = JsonUtility.FromJson<RawImuRecordingPackage>(
                    File.ReadAllText(filePath, Encoding.UTF8));
                if (Validate(package, out error)) return true;

                package = null;
                return false;
            }
            catch (Exception exception)
            {
                error = "Raw IMU JSON read failed: " + exception.Message;
                package = null;
                return false;
            }
        }

        public static bool TryFindLatest(out string filePath)
        {
            filePath = null;
            if (!Directory.Exists(RootDirectory)) return false;

            FileInfo latest = new DirectoryInfo(RootDirectory)
                .GetFiles("*" + FileExtension, SearchOption.TopDirectoryOnly)
                .OrderByDescending(file => file.LastWriteTimeUtc)
                .FirstOrDefault();
            if (latest == null) return false;

            filePath = latest.FullName;
            return true;
        }

        public static bool Validate(RawImuRecordingPackage package, out string error)
        {
            if (package == null || package.formatVersion < 1 ||
                package.formatVersion > CurrentFormatVersion)
            {
                error = "Unsupported Raw IMU JSON format version.";
                return false;
            }

            if (!IsValidActionId(package.actionId))
            {
                error = "Raw IMU actionId is invalid.";
                return false;
            }

            if (!IsFinite(package.sampleRate) || package.sampleRate < 0f ||
                !IsFinite(package.durationSeconds) || package.durationSeconds < 0f)
            {
                error = "Raw IMU timing metadata is invalid.";
                return false;
            }

            if (package.frames == null || package.frames.Length == 0 ||
                package.frameCount != package.frames.Length)
            {
                error = "Raw IMU JSON has no frames or frameCount is inconsistent.";
                return false;
            }

            var packageSensorIds = new HashSet<string>(StringComparer.Ordinal);
            if (package.sensorIds == null || package.sensorIds.Length == 0 ||
                package.sensorIds.Any(id => string.IsNullOrWhiteSpace(id) || !packageSensorIds.Add(id)))
            {
                error = "Raw IMU sensorIds are empty or contain duplicates.";
                return false;
            }

            float previousTime = -1f;
            var observedSensorIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (RawImuFrame frame in package.frames)
            {
                if (frame == null || !IsFinite(frame.timeSeconds) || frame.timeSeconds < previousTime)
                {
                    error = "Raw IMU JSON contains an invalid or unordered frame timestamp.";
                    return false;
                }

                previousTime = frame.timeSeconds;
                if (frame.sensors == null || frame.sensors.Length == 0)
                {
                    error = "Raw IMU JSON contains an empty sensor frame.";
                    return false;
                }

                var frameSensorIds = new HashSet<string>(StringComparer.Ordinal);
                foreach (RawImuSensorSample sensor in frame.sensors)
                {
                    if (!ValidateSensor(sensor, frameSensorIds, out error)) return false;
                    observedSensorIds.Add(sensor.sensorId);
                }
            }

            if (!observedSensorIds.SetEquals(packageSensorIds))
            {
                error = "Raw IMU sensorIds do not match the sensors contained in frames.";
                return false;
            }

            if (package.coach == null || !IsFinite(package.coach.heightCm) ||
                !IsFinite(package.coach.weightKg))
            {
                error = "Raw IMU coach profile is invalid.";
                return false;
            }

            error = null;
            return true;
        }

        private static bool ValidateSensor(RawImuSensorSample sensor, HashSet<string> frameSensorIds,
            out string error)
        {
            if (sensor == null || string.IsNullOrWhiteSpace(sensor.sensorId) ||
                !frameSensorIds.Add(sensor.sensorId))
            {
                error = "Raw IMU frame contains an invalid or duplicate sensorId.";
                return false;
            }

            if (!IsFinite(sensor.accelerationG) || !IsFinite(sensor.angularVelocityDegPerSec) ||
                !IsFinite(sensor.magneticFieldMicroTesla) || !IsFinite(sensor.eulerAnglesDeg) ||
                (sensor.hasOrientation && !IsFinite(sensor.orientation)) ||
                (sensor.hasRotationReferenceAdjusted &&
                 !IsFinite(sensor.rotationReferenceAdjusted)) ||
                (sensor.hasRotationIdentityAdjusted &&
                 !IsFinite(sensor.rotationIdentityAdjusted)) ||
                (sensor.hasMountingOrientation && !IsFinite(sensor.mountingOrientation)) ||
                (sensor.hasMountingResetOrientation &&
                 !IsFinite(sensor.mountingResetOrientation)) ||
                sensor.batteryPercent < 0 || sensor.batteryPercent > 100)
            {
                error = "Raw IMU frame contains invalid sensor values.";
                return false;
            }

            error = null;
            return true;
        }

        private static bool IsFinite(RawImuQuaternion value)
        {
            return value != null && IsFinite(value.x) && IsFinite(value.y) &&
                   IsFinite(value.z) && IsFinite(value.w);
        }

        private static string GetUniquePath(string desiredPath)
        {
            if (!File.Exists(desiredPath)) return desiredPath;

            string directory = Path.GetDirectoryName(desiredPath) ?? RootDirectory;
            string fileName = Path.GetFileName(desiredPath);
            string stem = fileName.EndsWith(FileExtension, StringComparison.OrdinalIgnoreCase)
                ? fileName.Substring(0, fileName.Length - FileExtension.Length)
                : Path.GetFileNameWithoutExtension(fileName);

            for (int index = 2; index < int.MaxValue; index++)
            {
                string candidate = Path.Combine(directory, stem + "_" + index + FileExtension);
                if (!File.Exists(candidate)) return candidate;
            }

            return Path.Combine(directory, stem + "_" + Guid.NewGuid().ToString("N") + FileExtension);
        }

        private static string SanitizeFileName(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return string.Empty;
            var invalid = new HashSet<char>(Path.GetInvalidFileNameChars());
            var builder = new StringBuilder(value.Length);
            foreach (char character in value.Trim())
                if (!invalid.Contains(character)) builder.Append(character);
            return builder.ToString();
        }

        private static bool IsValidActionId(string value)
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

        private static bool IsFinite(RawImuVector3 value)
        {
            return value != null && IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
        }

        private static bool IsFinite(float value)
        {
            return !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static bool IsFinite(double value)
        {
            return !double.IsNaN(value) && !double.IsInfinity(value);
        }
    }
}
