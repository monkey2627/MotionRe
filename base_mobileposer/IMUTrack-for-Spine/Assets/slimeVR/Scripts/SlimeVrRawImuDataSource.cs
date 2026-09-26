using System;
using System.Collections.Generic;
using System.IO;
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;
using Google.FlatBuffers;
using solarxr_protocol;
using solarxr_protocol.data_feed;
using solarxr_protocol.data_feed.device_data;
using solarxr_protocol.data_feed.tracker;
using solarxr_protocol.datatypes;
using solarxr_protocol.datatypes.hardware_info;
using UnityEngine;

namespace SpineFlow.RawImu
{
    /// <summary>
    /// Reads physical tracker data directly from SlimeVR Server's SolarXR feed.
    /// This does not use VMC bones or solved skeleton transforms.
    /// </summary>
    public static class SlimeVrRawImuDataSource
    {
        private const string Endpoint = "ws://127.0.0.1:21110";
        private const double StandardGravity = 9.80665d;
        private const double RadiansToDegrees = 180d / Math.PI;
        private static readonly object Sync = new object();
        private static readonly Dictionary<string, TrackerState> Trackers =
            new Dictionary<string, TrackerState>(StringComparer.Ordinal);

        private static CancellationTokenSource _cancellation;
        private static Task _worker;
        private static string _lastError = "Connecting to SlimeVR Server...";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void ResetForPlayMode()
        {
            Stop();
            lock (Sync)
            {
                Trackers.Clear();
                _lastError = "Connecting to SlimeVR Server...";
            }

            Application.quitting -= Stop;
            Application.quitting += Stop;
        }

        public static bool TryStart(out string error)
        {
            lock (Sync)
            {
                if (_worker != null && !_worker.IsCompleted)
                {
                    error = null;
                    return true;
                }

                _cancellation?.Dispose();
                _cancellation = new CancellationTokenSource();
                CancellationToken token = _cancellation.Token;
                _worker = Task.Run(() => RunConnectionLoopAsync(token), token);
                _lastError = "Connecting to SlimeVR Server...";
                error = null;
                return true;
            }
        }

        public static bool TryCopyLatestSamples(List<RawImuSensorSample> destination,
            out string error)
        {
            return TryCopyLatestSamples(destination, false, out error);
        }

        public static bool TryCopyLatestSamples(List<RawImuSensorSample> destination,
            bool includeNonPhysical, out string error)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            destination.Clear();

            lock (Sync)
            {
                DateTime now = DateTime.UtcNow;
                foreach (TrackerState tracker in Trackers.Values)
                {
                    // Normal recording must never ingest the temporary
                    // reconstruction device, including a disconnected instance
                    // that an older SlimeVR server still publishes. Diagnostic and
                    // reconstruction verification callers opt in with true.
                    if (!includeNonPhysical &&
                        (!tracker.IsPhysicalImu || tracker.IsReconstructionVirtual))
                        continue;
                    destination.Add(tracker.ToSample(now));
                }

                destination.Sort((left, right) =>
                    string.CompareOrdinal(left.sensorId, right.sensorId));
                if (destination.Count > 0)
                {
                    error = null;
                    return true;
                }

                error = _lastError;
                return false;
            }
        }

        public static void Stop()
        {
            lock (Sync)
            {
                if (_cancellation == null) return;
                _cancellation.Cancel();
                _cancellation.Dispose();
                _cancellation = null;
                _worker = null;
            }
        }

        private static async Task RunConnectionLoopAsync(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    using (var socket = new ClientWebSocket())
                    {
                        await socket.ConnectAsync(new Uri(Endpoint), token).ConfigureAwait(false);
                        byte[] request = BuildStartFeedRequest();
                        await socket.SendAsync(new ArraySegment<byte>(request),
                            WebSocketMessageType.Binary, true, token).ConfigureAwait(false);

                        SetError(null);
                        await ReceiveMessagesAsync(socket, token).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (token.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception exception)
                {
                    SetError("SlimeVR direct tracker connection failed: " + exception.Message);
                }

                try
                {
                    await Task.Delay(1000, token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    return;
                }
            }
        }

        private static async Task ReceiveMessagesAsync(ClientWebSocket socket,
            CancellationToken token)
        {
            var buffer = new byte[32768];
            while (!token.IsCancellationRequested && socket.State == WebSocketState.Open)
            {
                using (var message = new MemoryStream())
                {
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), token)
                            .ConfigureAwait(false);
                        if (result.MessageType == WebSocketMessageType.Close) return;
                        message.Write(buffer, 0, result.Count);
                    } while (!result.EndOfMessage);

                    if (result.MessageType == WebSocketMessageType.Binary)
                        ProcessBinaryMessage(message.ToArray());
                }
            }
        }

        private static byte[] BuildStartFeedRequest()
        {
            var trackerMask = new TrackerDataMaskT
            {
                Info = true,
                Status = true,
                Rotation = true,
                RotationReferenceAdjusted = true,
                RotationIdentityAdjusted = true,
                RawAngularVelocity = true,
                RawAcceleration = true,
                LinearAcceleration = true,
                Tps = true,
                RawMagneticVector = true
            };
            var deviceMask = new DeviceDataMaskT
            {
                TrackerData = trackerMask,
                DeviceData = true
            };
            var config = new DataFeedConfigT
            {
                MinimumTimeSinceLast = 20,
                DataMask = deviceMask
            };
            var start = new StartDataFeedT
            {
                DataFeeds = new List<DataFeedConfigT> { config }
            };
            var header = new DataFeedMessageHeaderT
            {
                Message = DataFeedMessageUnion.FromStartDataFeed(start)
            };
            var bundle = new MessageBundleT
            {
                DataFeedMsgs = new List<DataFeedMessageHeaderT> { header }
            };

            var builder = new FlatBufferBuilder(1024);
            Offset<MessageBundle> offset = MessageBundle.Pack(builder, bundle);
            builder.Finish(offset.Value);
            return builder.SizedByteArray();
        }

        private static void ProcessBinaryMessage(byte[] bytes)
        {
            if (bytes == null || bytes.Length < 4) return;

            MessageBundle bundle = MessageBundle.GetRootAsMessageBundle(new ByteBuffer(bytes));
            for (int messageIndex = 0; messageIndex < bundle.DataFeedMsgsLength; messageIndex++)
            {
                DataFeedMessageHeader? optionalHeader = bundle.DataFeedMsgs(messageIndex);
                if (!optionalHeader.HasValue) continue;
                DataFeedMessageHeader header = optionalHeader.Value;
                if (header.MessageType != DataFeedMessage.DataFeedUpdate) continue;
                MergeUpdate(header.MessageAsDataFeedUpdate().UnPack());
            }
        }

        private static void MergeUpdate(DataFeedUpdateT update)
        {
            if (update?.Devices == null) return;
            DateTime updatedAtUtc = DateTime.UtcNow;

            lock (Sync)
            {
                foreach (DeviceDataT device in update.Devices)
                {
                    if (device?.Id == null || device.Trackers == null) continue;
                    byte parentDeviceId = device.Id.Id;
                    int? battery = device.HardwareStatus?.BatteryPctEstimate;

                    foreach (TrackerDataT tracker in device.Trackers)
                    {
                        if (tracker?.TrackerId == null) continue;
                        byte deviceId = tracker.TrackerId.DeviceId?.Id ?? parentDeviceId;
                        byte trackerNumber = tracker.TrackerId.TrackerNum;
                        string key = deviceId + ":" + trackerNumber;
                        if (!Trackers.TryGetValue(key, out TrackerState state))
                        {
                            state = new TrackerState(deviceId, trackerNumber);
                            Trackers.Add(key, state);
                        }

                        state.Merge(device, tracker, battery, updatedAtUtc);
                    }
                }

                _lastError = null;
            }
        }

        private static void SetError(string error)
        {
            lock (Sync) _lastError = error;
        }

        private sealed class TrackerState
        {
            private readonly byte _deviceId;
            private readonly byte _trackerNumber;
            private string _sensorId;
            private BodyPart _bodyPart = BodyPart.NONE;
            private string _imuType = "Unknown IMU";
            private TrackerStatus _status = TrackerStatus.NONE;
            private bool _hasRawAcceleration;
            private bool _hasLinearAcceleration;
            private bool _hasAngularVelocity;
            private bool _hasMagneticField;
            private bool _hasEulerAngles;
            private bool _hasOrientation;
            private bool _hasRotationReferenceAdjusted;
            private bool _hasRotationIdentityAdjusted;
            private bool _hasMountingOrientation;
            private bool _hasMountingResetOrientation;
            private bool _hasBattery;
            private RawImuVector3 _rawAcceleration = new RawImuVector3();
            private RawImuVector3 _linearAcceleration = new RawImuVector3();
            private RawImuVector3 _angularVelocity = new RawImuVector3();
            private RawImuVector3 _magneticField = new RawImuVector3();
            private RawImuVector3 _eulerAngles = new RawImuVector3();
            private RawImuQuaternion _orientation = new RawImuQuaternion();
            private RawImuQuaternion _rotationReferenceAdjusted = new RawImuQuaternion();
            private RawImuQuaternion _rotationIdentityAdjusted = new RawImuQuaternion();
            private RawImuQuaternion _mountingOrientation = new RawImuQuaternion();
            private RawImuQuaternion _mountingResetOrientation = new RawImuQuaternion();
            private int _batteryPercent;
            private int _ticksPerSecond;
            private DateTime _updatedAtUtc;

            public bool IsPhysicalImu { get; private set; } = true;
            public bool IsReconstructionVirtual { get; private set; }

            public TrackerState(byte deviceId, byte trackerNumber)
            {
                _deviceId = deviceId;
                _trackerNumber = trackerNumber;
                _sensorId = $"slimevr_{deviceId}_{trackerNumber}";
            }

            public void Merge(DeviceDataT device, TrackerDataT tracker, int? battery,
                DateTime updatedAtUtc)
            {
                if (tracker.Info != null)
                {
                    IsPhysicalImu = tracker.Info.IsImu && !tracker.Info.IsComputed;
                    _bodyPart = tracker.Info.BodyPart;
                    _imuType = tracker.Info.ImuType.ToString();
                    _sensorId = FirstText(tracker.Info.CustomName, tracker.Info.DisplayName,
                        device.CustomName, device.HardwareInfo?.DisplayName,
                        $"slimevr_{_deviceId}_{_trackerNumber}");
                    if (tracker.Info.MountingOrientation != null)
                    {
                        _mountingOrientation = ToQuaternion(tracker.Info.MountingOrientation);
                        _hasMountingOrientation = true;
                    }
                    if (tracker.Info.MountingResetOrientation != null)
                    {
                        _mountingResetOrientation = ToQuaternion(
                            tracker.Info.MountingResetOrientation);
                        _hasMountingResetOrientation = true;
                    }
                }

                if (device.HardwareInfo != null)
                    IsReconstructionVirtual = string.Equals(
                        device.HardwareInfo.HardwareIdentifier, "127.0.0.42",
                        StringComparison.OrdinalIgnoreCase);

                if (tracker.Status != TrackerStatus.NONE) _status = tracker.Status;
                if (tracker.RawAcceleration != null)
                {
                    _rawAcceleration = ToAccelerationG(tracker.RawAcceleration);
                    _hasRawAcceleration = true;
                }
                if (tracker.LinearAcceleration != null)
                {
                    _linearAcceleration = ToAccelerationG(tracker.LinearAcceleration);
                    _hasLinearAcceleration = true;
                }
                if (tracker.RawAngularVelocity != null)
                {
                    _angularVelocity = new RawImuVector3(
                        tracker.RawAngularVelocity.X * RadiansToDegrees,
                        tracker.RawAngularVelocity.Y * RadiansToDegrees,
                        tracker.RawAngularVelocity.Z * RadiansToDegrees);
                    _hasAngularVelocity = true;
                }
                if (tracker.RawMagneticVector != null)
                {
                    _magneticField = new RawImuVector3(
                        tracker.RawMagneticVector.X * 0.1d,
                        tracker.RawMagneticVector.Y * 0.1d,
                        tracker.RawMagneticVector.Z * 0.1d);
                    _hasMagneticField = true;
                }
                if (tracker.Rotation != null)
                {
                    _eulerAngles = QuaternionToEulerDegrees(tracker.Rotation);
                    _hasEulerAngles = true;
                    _orientation = new RawImuQuaternion(tracker.Rotation.X, tracker.Rotation.Y,
                        tracker.Rotation.Z, tracker.Rotation.W);
                    _hasOrientation = true;
                }
                if (tracker.RotationReferenceAdjusted != null)
                {
                    _rotationReferenceAdjusted = ToQuaternion(
                        tracker.RotationReferenceAdjusted);
                    _hasRotationReferenceAdjusted = true;
                }
                if (tracker.RotationIdentityAdjusted != null)
                {
                    _rotationIdentityAdjusted = ToQuaternion(
                        tracker.RotationIdentityAdjusted);
                    _hasRotationIdentityAdjusted = true;
                }
                if (tracker.Tps.HasValue) _ticksPerSecond = tracker.Tps.Value;
                if (battery.HasValue)
                {
                    _batteryPercent = battery.Value;
                    _hasBattery = true;
                }

                _updatedAtUtc = updatedAtUtc;
            }

            public RawImuSensorSample ToSample(DateTime nowUtc)
            {
                bool fresh = _updatedAtUtc != default(DateTime) &&
                             (nowUtc - _updatedAtUtc).TotalSeconds < 3d;
                bool online = fresh && (_status == TrackerStatus.OK || _status == TrackerStatus.BUSY);
                bool useRawAcceleration = _hasRawAcceleration;
                return new RawImuSensorSample
                {
                    sensorId = _sensorId,
                    trackerRole = _bodyPart.ToString(),
                    online = online,
                    deviceUpdatedAtUtc = _updatedAtUtc == default(DateTime)
                        ? string.Empty
                        : _updatedAtUtc.ToString("o"),
                    source = "SlimeVR SolarXR (direct tracker, not VMC)",
                    imuType = _imuType,
                    accelerationKind = useRawAcceleration ? "raw" : "linear",
                    hasAcceleration = useRawAcceleration || _hasLinearAcceleration,
                    hasAngularVelocity = _hasAngularVelocity,
                    hasMagneticField = _hasMagneticField,
                    hasEulerAngles = _hasEulerAngles,
                    hasOrientation = _hasOrientation,
                    hasRotationReferenceAdjusted = _hasRotationReferenceAdjusted,
                    hasRotationIdentityAdjusted = _hasRotationIdentityAdjusted,
                    hasMountingOrientation = _hasMountingOrientation,
                    hasMountingResetOrientation = _hasMountingResetOrientation,
                    hasBattery = _hasBattery,
                    ticksPerSecond = _ticksPerSecond,
                    accelerationG = Clone(useRawAcceleration ? _rawAcceleration : _linearAcceleration),
                    angularVelocityDegPerSec = Clone(_angularVelocity),
                    magneticFieldMicroTesla = Clone(_magneticField),
                    eulerAnglesDeg = Clone(_eulerAngles),
                    orientation = _orientation?.Clone() ?? new RawImuQuaternion(),
                    rotationReferenceAdjusted = _rotationReferenceAdjusted?.Clone() ??
                                                new RawImuQuaternion(),
                    rotationIdentityAdjusted = _rotationIdentityAdjusted?.Clone() ??
                                               new RawImuQuaternion(),
                    mountingOrientation = _mountingOrientation?.Clone() ??
                                          new RawImuQuaternion(),
                    mountingResetOrientation = _mountingResetOrientation?.Clone() ??
                                               new RawImuQuaternion(),
                    batteryPercent = _batteryPercent
                };
            }

            private static RawImuQuaternion ToQuaternion(
                solarxr_protocol.datatypes.math.QuatT value)
            {
                return new RawImuQuaternion(value.X, value.Y, value.Z, value.W);
            }

            private static RawImuVector3 ToAccelerationG(
                solarxr_protocol.datatypes.math.Vec3fT value)
            {
                return new RawImuVector3(value.X / StandardGravity, value.Y / StandardGravity,
                    value.Z / StandardGravity);
            }

            private static RawImuVector3 QuaternionToEulerDegrees(
                solarxr_protocol.datatypes.math.QuatT value)
            {
                double x = value.X;
                double y = value.Y;
                double z = value.Z;
                double w = value.W;
                double roll = Math.Atan2(2d * (w * x + y * z),
                    1d - 2d * (x * x + y * y));
                double pitchInput = 2d * (w * y - z * x);
                double pitch = Math.Asin(Math.Max(-1d, Math.Min(1d, pitchInput)));
                double yaw = Math.Atan2(2d * (w * z + x * y),
                    1d - 2d * (y * y + z * z));
                return new RawImuVector3(roll * RadiansToDegrees,
                    pitch * RadiansToDegrees, yaw * RadiansToDegrees);
            }

            private static RawImuVector3 Clone(RawImuVector3 value)
            {
                return value?.Clone() ?? new RawImuVector3();
            }

            private static string FirstText(params string[] values)
            {
                foreach (string value in values)
                    if (!string.IsNullOrWhiteSpace(value)) return value.Trim();
                return "unknown_sensor";
            }
        }
    }
}
