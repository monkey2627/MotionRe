using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using SpineFlow.TrackerRecording;
using UnityEngine;

public sealed class VmcOscDebugBridge : MonoBehaviour
{
    [Header("Receive from SlimeVR")]
    [SerializeField] private int listenPort = 39539;

    [Header("Send to SlimeVR")]
    [SerializeField] private string targetAddress = "127.0.0.1";
    [SerializeField] private int targetPort = 39540;
    [SerializeField] private bool sendOkProbeOnStart = true;
    [SerializeField] private bool repeatOkProbe = false;
    [SerializeField] private float okProbeIntervalSeconds = 5f;

    [Header("Logging")]
    [SerializeField] private bool logRawMessages = true;
    [SerializeField] private bool logBoneMessages = true;
    [SerializeField] private bool combinePoseMessages = true;
    [SerializeField] private bool logTimeMessages = false;
    [SerializeField] private int maxMessagesPerFrame = 128;

    private readonly ConcurrentQueue<OscMessage> receivedMessages = new ConcurrentQueue<OscMessage>();
    private readonly Dictionary<string, TrackerPoseData> latestTrackerPoses =
        new Dictionary<string, TrackerPoseData>(StringComparer.Ordinal);
    private UdpClient receiver;
    private UdpClient sender;
    private Thread receiveThread;
    private volatile bool running;
    private float nextProbeTime;
    private float lastPoseMessageTime = -1000f;

    public bool HasRecentPoseMessages(float maximumAgeSeconds = 1f)
    {
        return Time.unscaledTime - lastPoseMessageTime <= Mathf.Max(0.1f, maximumAgeSeconds);
    }

    private void OnEnable()
    {
        StartReceiver();

        sender = new UdpClient();
        nextProbeTime = Time.unscaledTime + Mathf.Max(0.1f, okProbeIntervalSeconds);

        if (sendOkProbeOnStart)
        {
            SendOkProbe();
        }
    }

    private void Update()
    {
        FlushReceivedMessages();

        if (repeatOkProbe && Time.unscaledTime >= nextProbeTime)
        {
            SendOkProbe();
            nextProbeTime = Time.unscaledTime + Mathf.Max(0.1f, okProbeIntervalSeconds);
        }
    }

    private void OnDisable()
    {
        running = false;

        try
        {
            receiver?.Close();
        }
        catch (Exception exception)
        {
            Debug.LogWarning($"[VMC OSC] Closing receiver failed: {exception.Message}", this);
        }

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(200);
        }

        receiver = null;
        receiveThread = null;

        sender?.Close();
        sender = null;
    }

    [ContextMenu("Send /VMC/Ext/OK Probe")]
    public void SendOkProbe()
    {
        SendOsc("/VMC/Ext/OK", 1);
    }

    [ContextMenu("Send /VMC/Ext/Reset Calibration")]
    public void SendResetCalibration()
    {
        SendOsc("/VMC/Ext/Reset", 1);
    }

    private void StartReceiver()
    {
        try
        {
            receiver = new UdpClient(listenPort);
            running = true;
            receiveThread = new Thread(ReceiveLoop)
            {
                IsBackground = true,
                Name = "VMC OSC UDP Receiver"
            };
            receiveThread.Start();
            Debug.Log($"[VMC OSC] Listening for SlimeVR VMC output on UDP port {listenPort}. In SlimeVR, Port Out should be {listenPort} and Network address should be this Unity machine.", this);
        }
        catch (Exception exception)
        {
            Debug.LogError($"[VMC OSC] Could not listen on UDP port {listenPort}: {exception.Message}", this);
        }
    }

    private void ReceiveLoop()
    {
        var any = new IPEndPoint(IPAddress.Any, listenPort);

        while (running)
        {
            try
            {
                byte[] data = receiver.Receive(ref any);
                if (OscParser.TryParsePacket(data, data.Length, any, receivedMessages))
                {
                    continue;
                }

                receivedMessages.Enqueue(new OscMessage(any, "/<parse-failed>", new object[] { $"bytes={data.Length}" }));
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (SocketException)
            {
                if (running)
                {
                    receivedMessages.Enqueue(new OscMessage(any, "/<socket-error>", Array.Empty<object>()));
                }
            }
            catch (Exception exception)
            {
                receivedMessages.Enqueue(new OscMessage(any, "/<exception>", new object[] { exception.Message }));
            }
        }
    }

    private void FlushReceivedMessages()
    {
        int count = 0;
        var combinedPoses = combinePoseMessages ? new List<string>() : null;

        while (count < maxMessagesPerFrame && receivedMessages.TryDequeue(out OscMessage message))
        {
            count++;

            if (message.Address == "/VMC/Ext/Bone/Pos")
            {
                lastPoseMessageTime = Time.unscaledTime;
                if (combinePoseMessages)
                {
                    AddFormattedPose(combinedPoses, "BONE", message);
                }
                else
                {
                    LogBonePosition(message);
                }
            }
            else if (message.Address == "/VMC/Ext/Root/Pos")
            {
                lastPoseMessageTime = Time.unscaledTime;
                if (combinePoseMessages)
                {
                    AddFormattedPose(combinedPoses, "ROOT", message);
                }
                else
                {
                    LogNamedPose("ROOT", message);
                }
            }
            else if (message.Address == "/VMC/Ext/Tra/Pos" ||
                     message.Address == "/VMC/Ext/Tra/Pos/Local")
            {
                UpdateLatestTrackerPose(message);
                if (combinePoseMessages)
                {
                    AddFormattedPose(combinedPoses, "TRACKER", message);
                }
                else
                {
                    LogNamedPose("TRACKER", message);
                }
            }
            else if (message.Address == "/VMC/Ext/T")
            {
                if (logTimeMessages)
                {
                    Debug.Log($"[VMC OSC TIME] senderRelativeTime={FormatArguments(message.Arguments)}", this);
                }
            }
            else if (logRawMessages)
            {
                Debug.Log($"[VMC OSC IN] {message.RemoteEndPoint} {message.Address} {FormatArguments(message.Arguments)}", this);
            }
        }

        if (combinedPoses != null && combinedPoses.Count > 0)
        {
            Debug.Log($"[VMC OSC POSES] {combinedPoses.Count} pose(s) | {string.Join(" ; ", combinedPoses)}", this);
        }
    }

    public int CopyLatestTrackerPoses(List<TrackerPoseData> destination)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));

        destination.Clear();
        foreach (TrackerPoseData pose in latestTrackerPoses.Values)
        {
            destination.Add(pose.Clone());
        }

        destination.Sort((left, right) => string.CompareOrdinal(left.name, right.name));
        return destination.Count;
    }

    private void UpdateLatestTrackerPose(OscMessage message)
    {
        object[] args = message.Arguments;
        if (args == null || args.Length < 8 || !(args[0] is string trackerName) ||
            string.IsNullOrWhiteSpace(trackerName) ||
            !TryReadFloat(args[1], out float px) || !TryReadFloat(args[2], out float py) ||
            !TryReadFloat(args[3], out float pz) || !TryReadFloat(args[4], out float rx) ||
            !TryReadFloat(args[5], out float ry) || !TryReadFloat(args[6], out float rz) ||
            !TryReadFloat(args[7], out float rw))
        {
            return;
        }

        var rotation = new Quaternion(rx, ry, rz, rw);
        if (Quaternion.Dot(rotation, rotation) > 0.000001f) rotation.Normalize();
        else rotation = Quaternion.identity;

        latestTrackerPoses[trackerName] = new TrackerPoseData
        {
            name = trackerName,
            position = new Vector3(px, py, pz),
            rotation = rotation
        };
    }

    private static bool TryReadFloat(object value, out float result)
    {
        switch (value)
        {
            case float floatValue:
                result = floatValue;
                return true;
            case double doubleValue:
                result = (float)doubleValue;
                return true;
            case int intValue:
                result = intValue;
                return true;
            default:
                result = 0f;
                return false;
        }
    }

    private void LogBonePosition(OscMessage message)
    {
        if (!logBoneMessages)
        {
            return;
        }

        object[] args = message.Arguments;
        if (args.Length >= 8)
        {
            Debug.Log(
                $"[VMC OSC BONE] {args[0]} pos=({F(args[1])}, {F(args[2])}, {F(args[3])}) rot=({F(args[4])}, {F(args[5])}, {F(args[6])}, {F(args[7])})",
                this);
            return;
        }

        Debug.Log($"[VMC OSC BONE] {FormatArguments(args)}", this);
    }

    private void LogNamedPose(string label, OscMessage message)
    {
        object[] args = message.Arguments;
        if (args.Length >= 8)
        {
            Debug.Log(
                $"[VMC OSC {label}] {message.Address} {args[0]} pos=({F(args[1])}, {F(args[2])}, {F(args[3])}) rot=({F(args[4])}, {F(args[5])}, {F(args[6])}, {F(args[7])})",
                this);
            return;
        }

        Debug.Log($"[VMC OSC {label}] {message.Address} {FormatArguments(args)}", this);
    }

    private static void AddFormattedPose(List<string> poses, string label, OscMessage message)
    {
        object[] args = message.Arguments;
        if (args.Length >= 8)
        {
            poses.Add($"{label}:{args[0]} p=({F(args[1])},{F(args[2])},{F(args[3])}) q=({F(args[4])},{F(args[5])},{F(args[6])},{F(args[7])})");
            return;
        }

        poses.Add($"{label}:{message.Address} {FormatArguments(args)}");
    }

    private void SendOsc(string address, params object[] arguments)
    {
        if (sender == null)
        {
            sender = new UdpClient();
        }

        try
        {
            byte[] packet = OscWriter.WriteMessage(address, arguments);
            sender.Send(packet, packet.Length, targetAddress, targetPort);
            Debug.Log($"[VMC OSC OUT] {targetAddress}:{targetPort} {address} {FormatArguments(arguments)}", this);
        }
        catch (Exception exception)
        {
            Debug.LogError($"[VMC OSC] Send failed to {targetAddress}:{targetPort}: {exception.Message}", this);
        }
    }

    private static string FormatArguments(IReadOnlyList<object> arguments)
    {
        if (arguments == null || arguments.Count == 0)
        {
            return "(no args)";
        }

        var builder = new StringBuilder();
        for (int index = 0; index < arguments.Count; index++)
        {
            if (index > 0)
            {
                builder.Append(", ");
            }

            builder.Append(F(arguments[index]));
        }

        return builder.ToString();
    }

    private static string F(object value)
    {
        switch (value)
        {
            case float floatValue:
                return floatValue.ToString("0.###", CultureInfo.InvariantCulture);
            case double doubleValue:
                return doubleValue.ToString("0.###", CultureInfo.InvariantCulture);
            default:
                return value?.ToString() ?? "null";
        }
    }

    private readonly struct OscMessage
    {
        public readonly IPEndPoint RemoteEndPoint;
        public readonly string Address;
        public readonly object[] Arguments;

        public OscMessage(IPEndPoint remoteEndPoint, string address, object[] arguments)
        {
            RemoteEndPoint = remoteEndPoint;
            Address = address;
            Arguments = arguments;
        }
    }

    private static class OscParser
    {
        public static bool TryParsePacket(byte[] data, int length, IPEndPoint remoteEndPoint, ConcurrentQueue<OscMessage> output)
        {
            try
            {
                int offset = 0;
                int parsed = ParsePacket(data, length, remoteEndPoint, output, ref offset);
                return parsed > 0;
            }
            catch
            {
                return false;
            }
        }

        private static int ParsePacket(byte[] data, int length, IPEndPoint remoteEndPoint, ConcurrentQueue<OscMessage> output, ref int offset)
        {
            int startOffset = offset;
            string address = ReadPaddedString(data, length, ref offset);
            if (string.IsNullOrEmpty(address))
            {
                return 0;
            }

            if (address == "#bundle")
            {
                return ParseBundle(data, length, remoteEndPoint, output, ref offset);
            }

            string typeTags = ReadPaddedString(data, length, ref offset);
            if (string.IsNullOrEmpty(typeTags) || typeTags[0] != ',')
            {
                offset = startOffset;
                return 0;
            }

            var arguments = new List<object>();
            for (int index = 1; index < typeTags.Length; index++)
            {
                switch (typeTags[index])
                {
                    case 'i':
                        arguments.Add(ReadInt(data, length, ref offset));
                        break;
                    case 'f':
                        arguments.Add(ReadFloat(data, length, ref offset));
                        break;
                    case 's':
                        arguments.Add(ReadPaddedString(data, length, ref offset));
                        break;
                    case 'T':
                        arguments.Add(true);
                        break;
                    case 'F':
                        arguments.Add(false);
                        break;
                    case 'N':
                        arguments.Add(null);
                        break;
                    default:
                        arguments.Add($"unsupported:{typeTags[index]}");
                        break;
                }
            }

            output.Enqueue(new OscMessage(remoteEndPoint, address, arguments.ToArray()));
            return 1;
        }

        private static int ParseBundle(byte[] data, int length, IPEndPoint remoteEndPoint, ConcurrentQueue<OscMessage> output, ref int offset)
        {
            EnsureAvailable(length, offset, 8);
            offset += 8;

            int parsed = 0;
            while (offset < length)
            {
                int elementSize = ReadInt(data, length, ref offset);
                if (elementSize <= 0)
                {
                    break;
                }

                EnsureAvailable(length, offset, elementSize);
                int elementOffset = offset;
                int elementEnd = offset + elementSize;
                parsed += ParsePacket(data, elementEnd, remoteEndPoint, output, ref elementOffset);
                offset = elementEnd;
            }

            return parsed;
        }

        public static bool TryParse(byte[] data, int length, IPEndPoint remoteEndPoint, out OscMessage message)
        {
            message = default;

            try
            {
                var messages = new ConcurrentQueue<OscMessage>();
                if (!TryParsePacket(data, length, remoteEndPoint, messages))
                {
                    return false;
                }

                return messages.TryDequeue(out message);
            }
            catch
            {
                return false;
            }
        }

        private static string ReadPaddedString(byte[] data, int length, ref int offset)
        {
            int start = offset;
            while (offset < length && data[offset] != 0)
            {
                offset++;
            }

            if (offset >= length)
            {
                throw new FormatException("OSC string is not null terminated.");
            }

            string value = Encoding.UTF8.GetString(data, start, offset - start);
            offset++;
            AlignFour(ref offset);
            return value;
        }

        private static int ReadInt(byte[] data, int length, ref int offset)
        {
            EnsureAvailable(length, offset, 4);
            int value = IPAddress.NetworkToHostOrder(BitConverter.ToInt32(data, offset));
            offset += 4;
            return value;
        }

        private static float ReadFloat(byte[] data, int length, ref int offset)
        {
            EnsureAvailable(length, offset, 4);
            byte[] buffer = new byte[4];
            Buffer.BlockCopy(data, offset, buffer, 0, 4);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(buffer);
            }

            float value = BitConverter.ToSingle(buffer, 0);
            offset += 4;
            return value;
        }

        private static void EnsureAvailable(int length, int offset, int bytes)
        {
            if (offset + bytes > length)
            {
                throw new FormatException("OSC packet ended unexpectedly.");
            }
        }

        private static void AlignFour(ref int offset)
        {
            int remainder = offset % 4;
            if (remainder != 0)
            {
                offset += 4 - remainder;
            }
        }
    }

    private static class OscWriter
    {
        public static byte[] WriteMessage(string address, object[] arguments)
        {
            var bytes = new List<byte>();
            WritePaddedString(bytes, address);

            var typeTags = new StringBuilder(",");
            foreach (object argument in arguments)
            {
                switch (argument)
                {
                    case int _:
                        typeTags.Append('i');
                        break;
                    case float _:
                    case double _:
                        typeTags.Append('f');
                        break;
                    case string _:
                        typeTags.Append('s');
                        break;
                    case bool boolValue:
                        typeTags.Append(boolValue ? 'T' : 'F');
                        break;
                    default:
                        throw new NotSupportedException($"OSC argument type is not supported: {argument?.GetType().Name ?? "null"}");
                }
            }

            WritePaddedString(bytes, typeTags.ToString());

            foreach (object argument in arguments)
            {
                switch (argument)
                {
                    case int intValue:
                        WriteInt(bytes, intValue);
                        break;
                    case float floatValue:
                        WriteFloat(bytes, floatValue);
                        break;
                    case double doubleValue:
                        WriteFloat(bytes, (float)doubleValue);
                        break;
                    case string stringValue:
                        WritePaddedString(bytes, stringValue);
                        break;
                    case bool _:
                        break;
                }
            }

            return bytes.ToArray();
        }

        private static void WritePaddedString(List<byte> bytes, string value)
        {
            bytes.AddRange(Encoding.UTF8.GetBytes(value ?? string.Empty));
            bytes.Add(0);
            while (bytes.Count % 4 != 0)
            {
                bytes.Add(0);
            }
        }

        private static void WriteInt(List<byte> bytes, int value)
        {
            byte[] buffer = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(value));
            bytes.AddRange(buffer);
        }

        private static void WriteFloat(List<byte> bytes, float value)
        {
            byte[] buffer = BitConverter.GetBytes(value);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(buffer);
            }

            bytes.AddRange(buffer);
        }
    }
}
