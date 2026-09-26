using System;
using System.Collections.Generic;
using Assets;
using Assets.Device.Service;

namespace SpineFlow.RawImu
{
    /// <summary>
    /// Read-only adapter over the WitMotion SDK. It copies mutable device state
    /// into recording samples before the Unity frame uses it.
    /// </summary>
    public static class WitRawImuDataSource
    {
        public static bool TryStartUdp(out string error)
        {
            if (!TryResolveServices(out _, out UdpServer udpServer, out error)) return false;

            try
            {
                udpServer.StartReceive();
                return true;
            }
            catch (Exception exception)
            {
                error = "Could not start WitMotion UDP: " + exception.Message;
                return false;
            }
        }

        public static bool TrySendDiscovery(out string error)
        {
            if (!TryResolveServices(out _, out UdpServer udpServer, out error)) return false;

            try
            {
                udpServer.SendLoc();
                return true;
            }
            catch (Exception exception)
            {
                error = "Could not send WitMotion discovery: " + exception.Message;
                return false;
            }
        }

        public static bool TryCopyLatestSamples(List<RawImuSensorSample> destination,
            out string error)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            destination.Clear();

            if (!TryResolveServices(out DeviceService deviceService, out _, out error)) return false;

            for (int attempt = 0; attempt < 3; attempt++)
            {
                try
                {
                    List<DeviceModel> devices = deviceService.GetDeviceList();
                    foreach (DeviceModel device in devices)
                    {
                        if (device == null) continue;
                        destination.Add(CopyDevice(device));
                    }

                    destination.Sort((left, right) =>
                        string.CompareOrdinal(left.sensorId, right.sensorId));
                    error = null;
                    return true;
                }
                catch (InvalidOperationException)
                {
                    destination.Clear();
                }
            }

            error = "WitMotion device list changed while it was being read.";
            return false;
        }

        private static RawImuSensorSample CopyDevice(DeviceModel device)
        {
            // DeviceModel is updated by the UDP receive thread. Primitive reads
            // are copied together as one best-effort sample for this frame.
            DateTime updatedAt = device.LastUpdateTime;
            bool online = updatedAt != DateTime.MinValue &&
                          (DateTime.Now - updatedAt).TotalMilliseconds < 3000d;
            return new RawImuSensorSample
            {
                sensorId = NormalizeSensorId(device.DeivceId),
                online = online,
                source = "WitMotion UDP",
                imuType = "WitMotion",
                accelerationKind = "raw",
                hasAcceleration = true,
                hasAngularVelocity = true,
                hasMagneticField = true,
                hasEulerAngles = true,
                hasBattery = true,
                deviceUpdatedAtUtc = updatedAt == DateTime.MinValue
                    ? string.Empty
                    : updatedAt.ToUniversalTime().ToString("o"),
                accelerationG = new RawImuVector3(device.AccX, device.AccY, device.AccZ),
                angularVelocityDegPerSec = new RawImuVector3(device.AsX, device.AsY, device.AsZ),
                magneticFieldMicroTesla = new RawImuVector3(device.HX, device.HY, device.HZ),
                eulerAnglesDeg = new RawImuVector3(device.AngleX, device.AngleY, device.AngleZ),
                batteryPercent = device.Electricity
            };
        }

        private static bool TryResolveServices(out DeviceService deviceService, out UdpServer udpServer,
            out string error)
        {
            deviceService = null;
            udpServer = null;
            error = null;

            try
            {
                if (WitApplication.Context == null)
                {
                    error = "WitMotion application context is not initialized.";
                    return false;
                }

                deviceService = WitApplication.Context.GetBean<DeviceService>();
                udpServer = WitApplication.Context.GetBean<UdpServer>();
                if (deviceService == null || udpServer == null)
                {
                    error = "WitMotion DeviceService or UdpServer is unavailable.";
                    return false;
                }

                return true;
            }
            catch (Exception exception)
            {
                error = "WitMotion service resolution failed: " + exception.Message;
                return false;
            }
        }

        private static string NormalizeSensorId(string value)
        {
            return string.IsNullOrWhiteSpace(value)
                ? "unknown_sensor"
                : value.Trim('\0', ' ', '\t', '\r', '\n');
        }
    }
}
