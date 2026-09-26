using Assets.Library.WitUnitySdk.IOC.Attribute;
using Assets.Service.Device.Events;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace Assets.Device.Service
{
    [Compoment]
    public class UdpServer
    {
        private const int LocalPort = 1399;
        private const int DevicePort = 9250;
        private const int ReceiveLogInterval = 100;

        [Resource]
        public DeviceService DeviceService { get; set; }

        public MsgEvent msgEvent = new MsgEvent();

        private UdpClient udpcRecv = null;
        private IPEndPoint localIpep = new IPEndPoint(IPAddress.Any, LocalPort);
        private volatile bool IsUdpcRecvStart = false;
        private Thread thrRecv;
        private int receivePacketCount = 0;

        ~UdpServer()
        {
            StopReceive();
        }

        [PostConstruct]
        public void Init()
        {
        }

        public void StartReceive()
        {
            if (IsUdpcRecvStart)
            {
                Print($"UDP listen already started. local={localIpep.Address}:{localIpep.Port}");
                return;
            }

            try
            {
                udpcRecv = new UdpClient(localIpep);
                udpcRecv.EnableBroadcast = true;
                thrRecv = new Thread(ReceiveMessage);
                thrRecv.IsBackground = true;
                IsUdpcRecvStart = true;
                receivePacketCount = 0;
                thrRecv.Start();
                Print($"UDP listen started. local={localIpep.Address}:{localIpep.Port}");
                PrintLocalIpDiagnostics();
            }
            catch (SocketException e)
            {
                SafeCloseSocket();
                Print($"UDP listen failed. local={localIpep.Address}:{localIpep.Port}, socketError={e.SocketErrorCode}, native={e.ErrorCode}, message={e.Message}");
            }
            catch (Exception e)
            {
                SafeCloseSocket();
                Print("UDP listen failed. " + e.Message);
            }
        }

        public void StopReceive()
        {
            if (!IsUdpcRecvStart)
            {
                Print("UDP listen is not running.");
                return;
            }

            IsUdpcRecvStart = false;
            SafeCloseSocket();
            if (thrRecv != null && thrRecv.IsAlive)
            {
                thrRecv.Join(500);
            }

            udpcRecv = null;
            thrRecv = null;
            Print("UDP listen stopped.");
        }

        private void SafeCloseSocket()
        {
            try
            {
                if (udpcRecv != null)
                {
                    udpcRecv.Close();
                }
            }
            catch (Exception)
            {
            }
        }

        private void Print(string s)
        {
            string line = $"[WitUdp {DateTime.Now:HH:mm:ss.fff}] {s}";
            Debug.Log(line);
            msgEvent.Invoke(line);
        }

        private void ReceiveMessage()
        {
            IPEndPoint remoteIpep = new IPEndPoint(IPAddress.Any, 0);
            while (IsUdpcRecvStart)
            {
                try
                {
                    byte[] bytRecv = udpcRecv.Receive(ref remoteIpep);
                    if (bytRecv.Length < 1)
                    {
                        continue;
                    }

                    receivePacketCount++;
                    DeviceService.OnReceive(bytRecv);

                    if (receivePacketCount == 1 || receivePacketCount % ReceiveLogInterval == 0)
                    {
                        Print($"UDP RX #{receivePacketCount}: {bytRecv.Length} bytes from {remoteIpep.Address}:{remoteIpep.Port}, firstBytes={GetHexPreview(bytRecv, 8)}, allRotXYZ={GetAllDeviceRotationPreview()}");
                    }
                }
                catch (ObjectDisposedException)
                {
                    return;
                }
                catch (SocketException ex)
                {
                    if (IsUdpcRecvStart)
                    {
                        Print($"UDP receive socket error. socketError={ex.SocketErrorCode}, native={ex.ErrorCode}, message={ex.Message}");
                    }
                }
                catch (Exception ex)
                {
                    if (IsUdpcRecvStart)
                    {
                        Print("UDP receive failed. " + ex.Message);
                    }
                }
            }
        }

        private bool SendMessage(byte[] data, IPAddress address, int port, out string error)
        {
            error = null;
            try
            {
                udpcRecv.Send(data, data.Length, new IPEndPoint(address, port));
                return true;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                return false;
            }
        }

        public void SendLoc()
        {
            if (udpcRecv == null)
            {
                Print("UDP is not initialized. Click Start UDP first.");
                return;
            }

            List<LocalIpInfo> localIps = GetLocalIpInfos();
            if (localIps.Count == 0)
            {
                Print("No usable local IPv4 address found.");
                return;
            }

            try
            {
                Print("SendLoc local IPv4 candidates: " + string.Join("; ", localIps.Select(x => x.ToString()).ToArray()));

                int sentCount = 0;
                int failedCount = 0;
                string firstError = null;
                for (int localIndex = 0; localIndex < localIps.Count; localIndex++)
                {
                    LocalIpInfo localIp = localIps[localIndex];
                    string ip = localIp.Address.ToString();
                    string msg = $"WIT{ip}\r\n";
                    byte[] payload = Encoding.UTF8.GetBytes(msg);
                    List<IPAddress> targets = BuildProbeTargets(localIp, localIndex == 0).ToList();
                    Print($"SendLoc probing {targets.Count} targets on port {DevicePort}. localIp={ip}, interface={localIp.InterfaceName}, payload={msg.Trim()}");

                    foreach (IPAddress target in targets)
                    {
                        string error;
                        if (SendMessage(payload, target, DevicePort, out error))
                        {
                            sentCount++;
                        }
                        else
                        {
                            failedCount++;
                            if (firstError == null)
                            {
                                firstError = $"{target}:{DevicePort} -> {error}";
                            }
                        }
                    }
                }

                Print($"SendLoc finished. sent={sentCount}, failed={failedCount}, devicePort={DevicePort}, listenPort={LocalPort}, firstError={firstError ?? "none"}");
            }
            catch (Exception ex)
            {
                Print("SendLoc failed. " + ex.Message);
            }
        }

        private List<LocalIpInfo> GetLocalIpInfos()
        {
            List<LocalIpInfo> all = new List<LocalIpInfo>();
            try
            {
                foreach (NetworkInterface networkInterface in NetworkInterface.GetAllNetworkInterfaces())
                {
                    if (networkInterface.OperationalStatus != OperationalStatus.Up)
                    {
                        continue;
                    }

                    if (networkInterface.NetworkInterfaceType == NetworkInterfaceType.Loopback ||
                        networkInterface.NetworkInterfaceType == NetworkInterfaceType.Tunnel)
                    {
                        continue;
                    }

                    foreach (UnicastIPAddressInformation addressInfo in networkInterface.GetIPProperties().UnicastAddresses)
                    {
                        if (addressInfo.Address.AddressFamily != AddressFamily.InterNetwork)
                        {
                            continue;
                        }

                        IPAddress mask = null;
                        try
                        {
                            mask = addressInfo.IPv4Mask;
                        }
                        catch (Exception)
                        {
                        }

                        all.Add(new LocalIpInfo
                        {
                            Address = addressInfo.Address,
                            Mask = mask,
                            InterfaceName = networkInterface.Name
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                Print("NetworkInterface scan failed. " + ex.Message);
            }

            if (all.Count == 0)
            {
                AddDnsFallbackIps(all);
            }

            bool hasNonLinkLocal = all.Any(x => !IsLinkLocal(x.Address));
            IEnumerable<LocalIpInfo> usable = hasNonLinkLocal ? all.Where(x => !IsLinkLocal(x.Address)) : all;
            return usable
                .OrderByDescending(x => GetIpPriority(x.Address))
                .ThenBy(x => x.Address.ToString())
                .ToList();
        }

        private void AddDnsFallbackIps(List<LocalIpInfo> all)
        {
            try
            {
                IPHostEntry host = Dns.GetHostEntry(Dns.GetHostName());
                foreach (IPAddress ip in host.AddressList)
                {
                    if (ip.AddressFamily == AddressFamily.InterNetwork && !IPAddress.IsLoopback(ip))
                    {
                        all.Add(new LocalIpInfo
                        {
                            Address = ip,
                            Mask = IPAddress.Parse("255.255.255.0"),
                            InterfaceName = "DnsFallback"
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                Print("DNS local IP fallback failed. " + ex.Message);
            }
        }

        private IEnumerable<IPAddress> BuildProbeTargets(LocalIpInfo localIp, bool includeLimitedBroadcast)
        {
            List<string> emitted = new List<string>();
            Action<IPAddress> add = address =>
            {
                if (address == null)
                {
                    return;
                }

                string key = address.ToString();
                if (!emitted.Contains(key))
                {
                    emitted.Add(key);
                }
            };

            if (includeLimitedBroadcast)
            {
                add(IPAddress.Broadcast);
            }

            add(GetBroadcastAddress(localIp.Address, localIp.Mask));

            byte[] baseAddress = localIp.Address.GetAddressBytes();
            for (int i = 1; i < 255; i++)
            {
                byte[] target = (byte[])baseAddress.Clone();
                target[3] = (byte)i;
                add(new IPAddress(target));
            }

            foreach (string ip in emitted)
            {
                yield return IPAddress.Parse(ip);
            }
        }

        private IPAddress GetBroadcastAddress(IPAddress address, IPAddress mask)
        {
            if (address == null || mask == null)
            {
                return null;
            }

            byte[] addressBytes = address.GetAddressBytes();
            byte[] maskBytes = mask.GetAddressBytes();
            if (addressBytes.Length != maskBytes.Length)
            {
                return null;
            }

            byte[] broadcastBytes = new byte[addressBytes.Length];
            for (int i = 0; i < broadcastBytes.Length; i++)
            {
                broadcastBytes[i] = (byte)(addressBytes[i] | (maskBytes[i] ^ 255));
            }

            return new IPAddress(broadcastBytes);
        }

        private void PrintLocalIpDiagnostics()
        {
            List<LocalIpInfo> localIps = GetLocalIpInfos();
            if (localIps.Count == 0)
            {
                Print("Local IPv4 diagnostics: no usable IPv4 address.");
                return;
            }

            Print("Local IPv4 diagnostics: " + string.Join("; ", localIps.Select(x => x.ToString()).ToArray()));
        }

        private int GetIpPriority(IPAddress ip)
        {
            byte[] bytes = ip.GetAddressBytes();
            if (bytes.Length != 4)
            {
                return 0;
            }

            if (bytes[0] == 192 && bytes[1] == 168)
            {
                return 30;
            }

            if (bytes[0] == 10)
            {
                return 20;
            }

            if (bytes[0] == 172 && bytes[1] >= 16 && bytes[1] <= 31)
            {
                return 20;
            }

            if (IsLinkLocal(ip))
            {
                return 0;
            }

            return 10;
        }

        private bool IsLinkLocal(IPAddress ip)
        {
            byte[] bytes = ip.GetAddressBytes();
            return bytes.Length == 4 && bytes[0] == 169 && bytes[1] == 254;
        }

        private string GetHexPreview(byte[] data, int maxLength)
        {
            int count = Math.Min(data.Length, maxLength);
            List<string> bytes = new List<string>();
            for (int i = 0; i < count; i++)
            {
                bytes.Add(data[i].ToString("X2"));
            }

            return string.Join(" ", bytes.ToArray());
        }

        private string GetAllDeviceRotationPreview()
        {
            if (DeviceService == null)
            {
                return "rotXYZ=unavailable";
            }

            try
            {
                List<DeviceModel> devices = DeviceService.GetDeviceList();
                if (devices == null || devices.Count == 0)
                {
                    return "none";
                }

                return string.Join("; ", devices
                    .OrderBy(device => device.DeivceId)
                    .Select(device => $"{device.DeivceId} rotXYZ=({device.AngleX:F2}, {device.AngleY:F2}, {device.AngleZ:F2})")
                    .ToArray());

            }
            catch (Exception ex)
            {
                return "rotXYZ=unavailable: " + ex.Message;
            }
        }

        private class LocalIpInfo
        {
            public IPAddress Address;
            public IPAddress Mask;
            public string InterfaceName;

            public override string ToString()
            {
                string mask = Mask == null ? "no-mask" : Mask.ToString();
                return $"{Address}/{mask} ({InterfaceName})";
            }
        }
    }
}
