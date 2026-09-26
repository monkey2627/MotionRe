using System.Globalization;
using EVMC4U;
using UnityEngine;

public sealed class VmcRootPositionMonitor : MonoBehaviour, IExternalReceiver
{
    [SerializeField] private bool showOverlay = true;
    [SerializeField] private bool logWhenRootChanges = true;
    [SerializeField] private float logPositionDelta = 0.01f;

    private bool hasRootPacket;
    private Vector3 lastRootPosition;
    private Quaternion lastRootRotation = Quaternion.identity;
    private int rootPacketCount;
    private float lastPacketTime;
    private Vector3 lastLoggedPosition;

    public void MessageDaisyChain(ref uOSC.Message message, int callCount)
    {
        if (message.address != "/VMC/Ext/Root/Pos" || message.values == null || message.values.Length < 8)
        {
            return;
        }

        if (!(message.values[1] is float x)
            || !(message.values[2] is float y)
            || !(message.values[3] is float z)
            || !(message.values[4] is float rx)
            || !(message.values[5] is float ry)
            || !(message.values[6] is float rz)
            || !(message.values[7] is float rw))
        {
            return;
        }

        lastRootPosition = new Vector3(x, y, z);
        lastRootRotation = new Quaternion(rx, ry, rz, rw);
        rootPacketCount++;
        lastPacketTime = Time.unscaledTime;

        if (!hasRootPacket)
        {
            hasRootPacket = true;
            lastLoggedPosition = lastRootPosition;
            Debug.Log($"[VMC Root Monitor] First root packet position={Format(lastRootPosition)} rotation={Format(lastRootRotation)}", this);
            return;
        }

        if (logWhenRootChanges && Vector3.Distance(lastLoggedPosition, lastRootPosition) >= logPositionDelta)
        {
            lastLoggedPosition = lastRootPosition;
            Debug.Log($"[VMC Root Monitor] Root position changed to {Format(lastRootPosition)}", this);
        }
    }

    public void UpdateDaisyChain()
    {
    }

    private void OnGUI()
    {
        if (!showOverlay)
        {
            return;
        }

        const int width = 520;
        const int height = 76;
        GUILayout.BeginArea(new Rect(12, 12, width, height), GUI.skin.box);
        GUILayout.Label(hasRootPacket
            ? $"VMC Root packets: {rootPacketCount} | pos {Format(lastRootPosition)} | age {(Time.unscaledTime - lastPacketTime).ToString("0.00", CultureInfo.InvariantCulture)}s"
            : "VMC Root packets: none received");
        GUILayout.Label($"Root rot {Format(lastRootRotation)}");
        GUILayout.EndArea();
    }

    private static string Format(Vector3 value)
    {
        return $"({value.x.ToString("0.###", CultureInfo.InvariantCulture)}, {value.y.ToString("0.###", CultureInfo.InvariantCulture)}, {value.z.ToString("0.###", CultureInfo.InvariantCulture)})";
    }

    private static string Format(Quaternion value)
    {
        return $"({value.x.ToString("0.###", CultureInfo.InvariantCulture)}, {value.y.ToString("0.###", CultureInfo.InvariantCulture)}, {value.z.ToString("0.###", CultureInfo.InvariantCulture)}, {value.w.ToString("0.###", CultureInfo.InvariantCulture)})";
    }
}
