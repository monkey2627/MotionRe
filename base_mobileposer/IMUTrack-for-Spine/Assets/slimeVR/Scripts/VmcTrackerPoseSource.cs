using System;
using System.Collections.Generic;
using EVMC4U;
using SpineFlow.TrackerRecording;
using UnityEngine;

/// <summary>
/// Captures every VMC tracker pose from EVMC4U's existing message chain.
/// Attach it to an object already referenced by ExternalReceiver.NextReceivers;
/// no additional UDP listener is needed.
/// </summary>
public sealed class VmcTrackerPoseSource : MonoBehaviour, IExternalReceiver
{
    private readonly Dictionary<string, TrackerPoseData> latestTrackerPoses =
        new Dictionary<string, TrackerPoseData>(StringComparer.Ordinal);
    private readonly Dictionary<string, TrackerPoseData> latestBonePoses =
        new Dictionary<string, TrackerPoseData>(StringComparer.Ordinal);

    public string CurrentSourceDescription => latestTrackerPoses.Count > 0
        ? "SlimeVR VMC tracker poses (/VMC/Ext/Tra/Pos)"
        : "SlimeVR VMC bone pose fallback (/VMC/Ext/Bone/Pos)";

    public void MessageDaisyChain(ref uOSC.Message message, int callCount)
    {
        bool isTrackerPose = message.address == "/VMC/Ext/Tra/Pos" ||
                             message.address == "/VMC/Ext/Tra/Pos/Local";
        bool isBonePose = message.address == "/VMC/Ext/Bone/Pos";
        if (!isTrackerPose && !isBonePose)
            return;

        object[] values = message.values;
        if (values == null || values.Length < 8 || !(values[0] is string trackerName) ||
            string.IsNullOrWhiteSpace(trackerName) ||
            !TryReadFloat(values[1], out float px) || !TryReadFloat(values[2], out float py) ||
            !TryReadFloat(values[3], out float pz) || !TryReadFloat(values[4], out float rx) ||
            !TryReadFloat(values[5], out float ry) || !TryReadFloat(values[6], out float rz) ||
            !TryReadFloat(values[7], out float rw))
            return;

        var rotation = new Quaternion(rx, ry, rz, rw);
        if (Quaternion.Dot(rotation, rotation) > 0.000001f) rotation.Normalize();
        else rotation = Quaternion.identity;

        string poseName = isBonePose ? "bone:" + trackerName : trackerName;
        var pose = new TrackerPoseData
        {
            name = poseName,
            position = new Vector3(px, py, pz),
            rotation = rotation
        };

        if (isTrackerPose)
            latestTrackerPoses[poseName] = pose;
        else
            latestBonePoses[poseName] = pose;
    }

    public void UpdateDaisyChain()
    {
    }

    public int CopyLatestTrackerPoses(List<TrackerPoseData> destination)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));

        destination.Clear();
        Dictionary<string, TrackerPoseData> source = latestTrackerPoses.Count > 0
            ? latestTrackerPoses
            : latestBonePoses;
        foreach (TrackerPoseData pose in source.Values)
            destination.Add(pose.Clone());
        destination.Sort((left, right) => string.CompareOrdinal(left.name, right.name));
        return destination.Count;
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
}
