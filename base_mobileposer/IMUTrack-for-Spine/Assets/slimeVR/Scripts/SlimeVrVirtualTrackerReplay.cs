using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using Google.FlatBuffers;
using UnityEngine;
using SpineFlow.RawImu;

namespace SpineFlow.Reconstruction
{
    /// <summary>
    /// Replays recorded fused IMU rotations as one multi-sensor SlimeVR UDP device.
    /// The implementation intentionally sends only the documented handshake,
    /// sensor-info and rotation packets; the original recording is never modified.
    /// </summary>
    public sealed class SlimeVrVirtualTrackerReplay : IDisposable
    {
        private const int SlimeVrUdpPort = 6969;
        private const int VirtualLocalUdpPort = 6970;
        private const int ProtocolVersion = 21;
        private const byte SensorStatusOk = 1;
        private const byte RotationDataTypeNormal = 1;
        private const int VirtualHardwareType = 250;
        private const byte VirtualImuType = 16; // ICM45686: marks each sensor as a physical IMU
        private static readonly IPAddress VirtualLoopbackAddress = IPAddress.Parse("127.0.0.42");
        private const string VirtualHardwareIdentifier = "127.0.0.42";
        private static readonly Uri SolarXrEndpoint = new Uri("ws://127.0.0.1:21110");
        private readonly UdpClient _udp;
        private readonly IPEndPoint _server = new IPEndPoint(IPAddress.Loopback, SlimeVrUdpPort);
        private readonly Dictionary<string, byte> _sensorNumbers = new Dictionary<string, byte>(StringComparer.Ordinal);
        private readonly Dictionary<string, byte> _trackerPositions = new Dictionary<string, byte>(StringComparer.Ordinal);
        private long _packetNumber = 1;

        public SlimeVrVirtualTrackerReplay(IReadOnlyList<RawImuSensorSample> sensors)
        {
            if (sensors == null || sensors.Count == 0)
                throw new ArgumentException("At least one sensor is required.", nameof(sensors));

            // Use a dedicated loopback alias. SlimeVR uses the address as the
            // hardware identity when no MAC is supplied; 127.0.0.42 keeps this
            // virtual rig separate from stale 127.0.0.1 tracker assignments.
            // A stable source endpoint lets SlimeVR reuse the same temporary UDP
            // device rather than retaining one disconnected device per replay.
            _udp = new UdpClient(new IPEndPoint(VirtualLoopbackAddress, VirtualLocalUdpPort));
            var usedSensorNumbers = new HashSet<byte>();
            byte fallbackSensorNumber = 100;
            for (int index = 0; index < sensors.Count; index++)
            {
                RawImuSensorSample sensor = sensors[index];
                if (sensor == null || string.IsNullOrWhiteSpace(sensor.sensorId)) continue;
                byte trackerPosition = ResolveTrackerPosition(sensor.trackerRole);
                byte sensorNumber = trackerPosition;
                if (sensorNumber == 0 || usedSensorNumbers.Contains(sensorNumber))
                {
                    while (usedSensorNumbers.Contains(fallbackSensorNumber)) fallbackSensorNumber++;
                    sensorNumber = fallbackSensorNumber++;
                }
                usedSensorNumbers.Add(sensorNumber);
                _sensorNumbers[sensor.sensorId] = sensorNumber;
                _trackerPositions[sensor.sensorId] = trackerPosition;
            }

            if (_sensorNumbers.Count == 0)
                throw new InvalidOperationException("The selected tracker set contains no valid sensors.");

            SendHandshake();
            SendSensorInfo();
        }

        public void SendSensorInfo(bool connected = true)
        {
            foreach (KeyValuePair<string, byte> entry in _sensorNumbers)
            {
                using (var packet = BeginPacket(15))
                {
                    packet.Writer.Write(entry.Value);
                    packet.Writer.Write(connected ? SensorStatusOk : (byte)0);
                    packet.Writer.Write(VirtualImuType);
                    packet.WriteInt16(0); // sensor configuration
                    packet.Writer.Write((byte)1); // rest calibration complete
                    packet.Writer.Write(_trackerPositions[entry.Key]);
                    packet.Writer.Write((byte)0); // tracker data type: rotation
                    Send(packet);
                }
            }
        }

        public void SendFrame(RawImuFrame frame, ISet<string> selectedSensorIds)
        {
            if (frame?.sensors == null) return;
            foreach (RawImuSensorSample sample in frame.sensors)
            {
                // "online" is historical metadata. During an offline reconstruction
                // the recorded orientation remains usable even when an individual
                // source frame was marked offline, so do not suppress its packet.
                if (sample == null ||
                    !selectedSensorIds.Contains(sample.sensorId) ||
                    !_sensorNumbers.TryGetValue(sample.sensorId, out byte sensorNumber))
                    continue;

                GetQuaternion(sample, out float x, out float y, out float z, out float w);
                using (var packet = BeginPacket(17))
                {
                    packet.Writer.Write(sensorNumber);
                    packet.Writer.Write(RotationDataTypeNormal);
                    packet.WriteSingle(x);
                    packet.WriteSingle(y);
                    packet.WriteSingle(z);
                    packet.WriteSingle(w);
                    packet.Writer.Write((byte)0); // calibration quality
                    Send(packet);
                }
            }
        }

        public void Dispose()
        {
            try { SendSensorInfo(false); }
            catch { /* best-effort offline notification */ }
            _udp?.Close();

            bool forgetSucceeded = TryForgetVirtualDevice(out string cleanupError);
            // Always run the persistent-config cleanup. Even if the RPC endpoint
            // has just gone away, a crashed/closed SlimeVR must not leave the
            // temporary loopback tracker in vrconfig.yml.
            if (!TryRestartSlimeVr(out string restartError))
            {
                string rpcDetail = forgetSucceeded ? string.Empty :
                    " ForgetDevice also failed: " + cleanupError + ".";
                Debug.LogWarning("[History Reconstruction] Could not completely purge the " +
                                 "temporary SlimeVR device: " + restartError + "." + rpcDetail);
            }
        }

        /// <summary>
        /// Uses SlimeVR's ForgetDevice RPC after the UDP sender is closed. This
        /// removes the 127.0.0.42 device from the active server instead of leaving
        /// disconnected virtual trackers in later SolarXR capture sessions.
        /// </summary>
        private static bool TryForgetVirtualDevice(out string error)
        {
            error = null;
            try
            {
                byte[] request = BuildForgetDeviceRequest();
                using (var cancellation = new CancellationTokenSource(2000))
                using (var socket = new ClientWebSocket())
                {
                    socket.ConnectAsync(SolarXrEndpoint, cancellation.Token)
                        .GetAwaiter().GetResult();
                    socket.SendAsync(new ArraySegment<byte>(request),
                            WebSocketMessageType.Binary, true, cancellation.Token)
                        .GetAwaiter().GetResult();
                    socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure,
                            "SpineFlow reconstruction complete", cancellation.Token)
                        .GetAwaiter().GetResult();
                }
                return true;
            }
            catch (Exception exception)
            {
                error = exception.Message;
                return false;
            }
        }

        private static byte[] BuildForgetDeviceRequest()
        {
            var builder = new FlatBufferBuilder(256);
            StringOffset hardwareIdentifier = builder.CreateString(VirtualHardwareIdentifier);

            // ForgetDeviceRequest { mac_address:string }.
            builder.StartTable(1);
            builder.AddOffset(0, hardwareIdentifier.Value, 0);
            int forgetRequest = builder.EndTable();

            // RpcMessageHeader { tx_id, message_type, message }. The SlimeVR v21
            // RpcMessage enum value for ForgetDeviceRequest is 57.
            builder.StartTable(3);
            builder.AddByte(1, 57, 0);
            builder.AddOffset(2, forgetRequest, 0);
            int rpcHeader = builder.EndTable();

            builder.StartVector(4, 1, 4);
            builder.AddOffset(rpcHeader);
            VectorOffset rpcMessages = builder.EndVector();

            // MessageBundle field 1 is rpc_msgs (field 0 is data_feed_msgs).
            builder.StartTable(3);
            builder.AddOffset(1, rpcMessages.Value, 0);
            int bundle = builder.EndTable();
            builder.Finish(bundle);
            return builder.SizedByteArray();
        }

        /// <summary>
        /// SlimeVR v21's ForgetDevice RPC disconnects a UDP device but retains its
        /// disconnected DeviceManager object until the server restarts. Restarting
        /// the desktop process after a successful forget is therefore required to
        /// make the temporary tracker disappear from both the UI and SolarXR feed.
        /// The reconstruction workflow already requires all real trackers offline,
        /// and the Unity SolarXR reader reconnects automatically afterwards.
        /// </summary>
        private static bool TryRestartSlimeVr(out string error)
        {
            error = null;
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            try
            {
                System.Diagnostics.Process mainProcess = null;
                string executablePath = null;
                foreach (System.Diagnostics.Process process in
                         System.Diagnostics.Process.GetProcessesByName("SlimeVR"))
                {
                    string candidatePath = null;
                    try
                    {
                        candidatePath = process.MainModule?.FileName;
                    }
                    catch
                    {
                        // MainModule access is denied in some Unity/Mono hosts.
                    }

                    bool hasMainWindow = process.MainWindowHandle != IntPtr.Zero;
                    if (mainProcess != null && !hasMainWindow) continue;
                    mainProcess = process;
                    if (!string.IsNullOrWhiteSpace(candidatePath) &&
                        candidatePath.EndsWith("SlimeVR.exe",
                            StringComparison.OrdinalIgnoreCase))
                        executablePath = candidatePath;
                    if (hasMainWindow) break;
                }

                HashSet<int> desktopProcessIds = GetProcessIdsByImageName("SlimeVR.exe");
                if (mainProcess != null) desktopProcessIds.Add(mainProcess.Id);
                HashSet<int> serverProcessIds = GetTcpListenerProcessIds(21110);
                if (desktopProcessIds.Count == 0 && serverProcessIds.Count == 0)
                {
                    // The application is already stopped, so its configuration can
                    // be edited safely and there is nothing that needs restarting.
                    return TryRemoveVirtualTrackerConfig(out error);
                }

                if (string.IsNullOrWhiteSpace(executablePath))
                    executablePath = FindSlimeVrExecutableFallback();
                if (string.IsNullOrWhiteSpace(executablePath))
                {
                    error = "SlimeVR is running, but its executable path could not be resolved";
                    return false;
                }

                // Capture the Java server PID before stopping the desktop. During
                // shutdown it may release 21110 first and save vrconfig.yml a
                // moment later, making a post-stop port lookup miss the process.
                foreach (int desktopPid in desktopProcessIds)
                {
                    if (!TryStopProcessTree(desktopPid, out string desktopStopError))
                    {
                        error = "SlimeVR desktop process tree did not stop cleanly: " +
                                desktopStopError;
                        return false;
                    }
                }

                // SlimeVR's Java server can outlive the desktop launcher on some
                // installations. Kill the actual owner of SolarXR port 21110 as
                // well; otherwise that old server rewrites the just-removed
                // virtual tracker into vrconfig.yml during the desktop restart.
                serverProcessIds.UnionWith(GetTcpListenerProcessIds(21110));
                foreach (int serverPid in serverProcessIds)
                {
                    if (desktopProcessIds.Contains(serverPid)) continue;
                    if (!TryStopProcessTree(serverPid, out string serverStopError))
                    {
                        error = "SlimeVR SolarXR service did not stop cleanly: " +
                                serverStopError;
                        return false;
                    }
                }

                for (int attempt = 0; attempt < 30 &&
                     GetTcpListenerProcessIds(21110).Count > 0; attempt++)
                    Thread.Sleep(100);
                if (GetTcpListenerProcessIds(21110).Count > 0)
                {
                    error = "SlimeVR SolarXR port 21110 is still in use";
                    return false;
                }

                // SlimeVR persists every discovered UDP sensor under a key derived
                // from its source address. Remove only SpineFlow's reserved
                // loopback entries while the server is stopped so they cannot be
                // restored as offline historical trackers on the next launch.
                bool configCleaned = TryRemoveVirtualTrackerConfig(out string configError);

                System.Diagnostics.Process restarted = System.Diagnostics.Process.Start(
                    new System.Diagnostics.ProcessStartInfo
                {
                    FileName = executablePath,
                    UseShellExecute = true
                });
                if (restarted == null)
                {
                    error = "SlimeVR did not restart";
                    return false;
                }

                if (!configCleaned)
                {
                    error = "SlimeVR restarted, but its temporary tracker config could not be removed: " +
                            configError;
                    return false;
                }

                return true;
            }
            catch (Exception exception)
            {
                error = exception.Message;
                return false;
            }
#else
            error = "automatic SlimeVR restart is only supported on Windows";
            return false;
#endif
        }

        private static string FindSlimeVrExecutableFallback()
        {
            string[] candidates =
            {
                // Project deployment used by the SpineFlow workstation.
                @"D:\slimevr\SlimeVR.exe",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "Programs", "SlimeVR", "SlimeVR.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                    "SlimeVR", "SlimeVR.exe")
            };
            foreach (string candidate in candidates)
                if (!string.IsNullOrWhiteSpace(candidate) && File.Exists(candidate))
                    return candidate;
            return null;
        }

        private static HashSet<int> GetProcessIdsByImageName(string imageName)
        {
            var result = new HashSet<int>();
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            try
            {
                var info = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "tasklist.exe",
                    Arguments = "/FI \"IMAGENAME eq " + imageName + "\" /FO CSV /NH",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true,
                    WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden
                };
                using (System.Diagnostics.Process process =
                       System.Diagnostics.Process.Start(info))
                {
                    if (process == null) return result;
                    string output = process.StandardOutput.ReadToEnd();
                    process.WaitForExit(3000);
                    foreach (string line in output.Split(new[] { '\r', '\n' },
                                 StringSplitOptions.RemoveEmptyEntries))
                    {
                        string[] columns = line.Split(',');
                        if (columns.Length < 2 ||
                            !string.Equals(columns[0].Trim().Trim('"'), imageName,
                                StringComparison.OrdinalIgnoreCase) ||
                            !int.TryParse(columns[1].Trim().Trim('"'), out int processId))
                            continue;
                        result.Add(processId);
                    }
                }
            }
            catch
            {
                // Process.GetProcessesByName remains the primary discovery path.
            }
#endif
            return result;
        }

        private static bool TryStopProcessTree(int processId, out string error)
        {
            error = null;
            try
            {
                try
                {
                    using (System.Diagnostics.Process existing =
                           System.Diagnostics.Process.GetProcessById(processId))
                    {
                        if (existing.HasExited) return true;
                    }
                }
                catch (ArgumentException)
                {
                    return true;
                }

                var stopInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "taskkill.exe",
                    Arguments = "/PID " + processId + " /T /F",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden
                };
                using (System.Diagnostics.Process process =
                       System.Diagnostics.Process.Start(stopInfo))
                {
                    if (process == null)
                    {
                        error = "taskkill did not start";
                        return false;
                    }

                    string output = process.StandardOutput.ReadToEnd();
                    string standardError = process.StandardError.ReadToEnd();
                    if (!process.WaitForExit(5000) || process.ExitCode != 0)
                    {
                        error = string.IsNullOrWhiteSpace(standardError)
                            ? output.Trim()
                            : standardError.Trim();
                        return false;
                    }
                }

                return true;
            }
            catch (Exception exception)
            {
                error = exception.Message;
                return false;
            }
        }

        private static HashSet<int> GetTcpListenerProcessIds(int port)
        {
            var result = new HashSet<int>();
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            try
            {
                var info = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "netstat.exe",
                    Arguments = "-ano -p TCP",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true,
                    WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden
                };
                using (System.Diagnostics.Process process =
                       System.Diagnostics.Process.Start(info))
                {
                    if (process == null) return result;
                    string output = process.StandardOutput.ReadToEnd();
                    process.WaitForExit(3000);
                    string expectedSuffix = ":" + port;
                    foreach (string line in output.Split(new[] { '\r', '\n' },
                                 StringSplitOptions.RemoveEmptyEntries))
                    {
                        string[] columns = line.Split((char[])null,
                            StringSplitOptions.RemoveEmptyEntries);
                        if (columns.Length < 5 ||
                            !string.Equals(columns[0], "TCP",
                                StringComparison.OrdinalIgnoreCase) ||
                            !columns[1].EndsWith(expectedSuffix,
                                StringComparison.OrdinalIgnoreCase) ||
                            !string.Equals(columns[3], "LISTENING",
                                StringComparison.OrdinalIgnoreCase) ||
                            !int.TryParse(columns[4], out int processId)) continue;
                        result.Add(processId);
                    }
                }
            }
            catch
            {
                // The caller treats an empty set as no separate server process.
            }
#endif
            return result;
        }

        private static bool TryRemoveVirtualTrackerConfig(out string error)
        {
            error = null;
            string tempPath = null;
            try
            {
                string appData = Environment.GetFolderPath(
                    Environment.SpecialFolder.ApplicationData);
                string configPath = Path.Combine(appData, "dev.slimevr.SlimeVR", "vrconfig.yml");
                if (!File.Exists(configPath)) return true;

                string[] lines = File.ReadAllLines(configPath);
                var filtered = new List<string>(lines.Length);
                bool skippingVirtualEntry = false;
                bool changed = false;
                foreach (string line in lines)
                {
                    if (IsVirtualTrackerConfigHeader(line))
                    {
                        skippingVirtualEntry = true;
                        changed = true;
                        continue;
                    }

                    if (skippingVirtualEntry)
                    {
                        if (!IsTrackerEntryBoundary(line)) continue;
                        skippingVirtualEntry = false;
                    }

                    filtered.Add(line);
                }

                if (!changed) return true;

                tempPath = configPath + ".spineflow.tmp";
                string backupPath = configPath + ".spineflow.bak";
                File.WriteAllLines(tempPath, filtered, new UTF8Encoding(false));
                File.Copy(configPath, backupPath, true);
                File.Copy(tempPath, configPath, true);
                File.Delete(tempPath);
                return true;
            }
            catch (Exception exception)
            {
                error = exception.Message;
                if (!string.IsNullOrEmpty(tempPath))
                {
                    try
                    {
                        if (File.Exists(tempPath)) File.Delete(tempPath);
                    }
                    catch
                    {
                        // Preserve the original cleanup error.
                    }
                }

                return false;
            }
        }

        private static bool IsVirtualTrackerConfigHeader(string line)
        {
            const string prefix = "  udp://" + VirtualHardwareIdentifier + "/";
            if (string.IsNullOrEmpty(line) ||
                !line.StartsWith(prefix, StringComparison.Ordinal)) return false;

            string suffix = line.Substring(prefix.Length).Trim();
            return suffix.Length > 1 && suffix.EndsWith(":", StringComparison.Ordinal) &&
                   int.TryParse(suffix.Substring(0, suffix.Length - 1), out _);
        }

        private static bool IsTrackerEntryBoundary(string line)
        {
            if (string.IsNullOrWhiteSpace(line)) return false;
            if (!char.IsWhiteSpace(line[0])) return true;
            return line.Length > 2 && line[0] == ' ' && line[1] == ' ' &&
                   !char.IsWhiteSpace(line[2]) &&
                   line.TrimEnd().EndsWith(":", StringComparison.Ordinal);
        }

        private void SendHandshake()
        {
            using (var packet = BeginPacket(3))
            {
                packet.WriteInt32(VirtualHardwareType); // development/reserved board
                packet.WriteInt32(VirtualImuType);
                packet.WriteInt32(VirtualHardwareType); // development/reserved MCU
                packet.WriteInt32(0);
                packet.WriteInt32(0);
                packet.WriteInt32(0);
                packet.WriteInt32(ProtocolVersion);
                byte[] firmware = Encoding.ASCII.GetBytes("SpineFlow Reconstruction");
                packet.Writer.Write((byte)firmware.Length);
                packet.Writer.Write(firmware);
                // An all-zero MAC is decoded as "no MAC" by SlimeVR. This is
                // intentional for a loopback-only temporary device: v21 rejects an
                // unknown non-zero MAC until the user completes hardware pairing.
                packet.Writer.Write(new byte[6]);
                Send(packet);
            }
        }

        private PacketWriter BeginPacket(int packetId)
        {
            var packet = new PacketWriter();
            packet.WriteInt32(packetId);
            packet.WriteInt64(_packetNumber++);
            return packet;
        }

        private void Send(PacketWriter packet)
        {
            byte[] bytes = packet.Stream.ToArray();
            _udp.Send(bytes, bytes.Length, _server);
        }

        private static void GetQuaternion(RawImuSensorSample sample, out float x, out float y,
            out float z, out float w)
        {
            if (sample.hasOrientation && sample.orientation != null)
            {
                x = (float)sample.orientation.x;
                y = (float)sample.orientation.y;
                z = (float)sample.orientation.z;
                w = (float)sample.orientation.w;
                Normalize(ref x, ref y, ref z, ref w);
                return;
            }

            // v1 compatibility: invert the roll/pitch/yaw extraction used by the
            // SolarXR recorder. New recordings use the quaternion path above.
            double roll = DegreesToRadians(sample.eulerAnglesDeg?.x ?? 0d);
            double pitch = DegreesToRadians(sample.eulerAnglesDeg?.y ?? 0d);
            double yaw = DegreesToRadians(sample.eulerAnglesDeg?.z ?? 0d);
            double cr = Math.Cos(roll * 0.5d);
            double sr = Math.Sin(roll * 0.5d);
            double cp = Math.Cos(pitch * 0.5d);
            double sp = Math.Sin(pitch * 0.5d);
            double cy = Math.Cos(yaw * 0.5d);
            double sy = Math.Sin(yaw * 0.5d);
            w = (float)(cr * cp * cy + sr * sp * sy);
            x = (float)(sr * cp * cy - cr * sp * sy);
            y = (float)(cr * sp * cy + sr * cp * sy);
            z = (float)(cr * cp * sy - sr * sp * cy);
            Normalize(ref x, ref y, ref z, ref w);
        }

        private static void Normalize(ref float x, ref float y, ref float z, ref float w)
        {
            double length = Math.Sqrt(x * x + y * y + z * z + w * w);
            if (length < 0.000001d)
            {
                x = y = z = 0f;
                w = 1f;
                return;
            }

            float inverse = (float)(1d / length);
            x *= inverse;
            y *= inverse;
            z *= inverse;
            w *= inverse;
        }

        private static double DegreesToRadians(double value) => value * Math.PI / 180d;

        private static byte ResolveTrackerPosition(string role)
        {
            if (string.IsNullOrWhiteSpace(role)) return 0;
            switch (role.Trim().ToUpperInvariant())
            {
                case "HEAD": return 1;
                case "NECK": return 2;
                case "UPPER_CHEST": return 3;
                case "CHEST": return 4;
                case "WAIST": return 5;
                case "HIP": return 6;
                case "LEFT_UPPER_LEG": return 7;
                case "RIGHT_UPPER_LEG": return 8;
                case "LEFT_LOWER_LEG": return 9;
                case "RIGHT_LOWER_LEG": return 10;
                case "LEFT_FOOT": return 11;
                case "RIGHT_FOOT": return 12;
                case "LEFT_LOWER_ARM": return 13;
                case "RIGHT_LOWER_ARM": return 14;
                case "LEFT_UPPER_ARM": return 15;
                case "RIGHT_UPPER_ARM": return 16;
                case "LEFT_HAND": return 17;
                case "RIGHT_HAND": return 18;
                default: return 0;
            }
        }

        private sealed class PacketWriter : IDisposable
        {
            public readonly MemoryStream Stream = new MemoryStream(64);
            public readonly BinaryWriter Writer;

            public PacketWriter()
            {
                Writer = new BinaryWriter(Stream);
            }

            public void WriteInt16(short value)
            {
                Writer.Write((byte)((value >> 8) & 0xff));
                Writer.Write((byte)(value & 0xff));
            }

            public void WriteInt32(int value)
            {
                Writer.Write((byte)((value >> 24) & 0xff));
                Writer.Write((byte)((value >> 16) & 0xff));
                Writer.Write((byte)((value >> 8) & 0xff));
                Writer.Write((byte)(value & 0xff));
            }

            public void WriteInt64(long value)
            {
                for (int shift = 56; shift >= 0; shift -= 8)
                    Writer.Write((byte)((value >> shift) & 0xff));
            }

            public void WriteSingle(float value)
            {
                byte[] bytes = BitConverter.GetBytes(value);
                if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
                Writer.Write(bytes);
            }

            public void Dispose()
            {
                Writer.Dispose();
                Stream.Dispose();
            }
        }
    }
}
