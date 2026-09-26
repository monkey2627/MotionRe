using System;
using UnityEngine;

[Serializable]
public sealed class RecordedCoachMotionPackage
{
    public int formatVersion = 1;
    public string actionId;
    public string displayName;
    public float fps = 30f;
    public string sourceClipName;
    public int setCount = 1;
    public RecordedCoachMotionSegment[] segments = Array.Empty<RecordedCoachMotionSegment>();
}

[Serializable]
public sealed class RecordedCoachMotionSegment
{
    public string label;
    public string sourceStateName;
    public int repeatCount = 1;
    public float scoreLeniency = 1f;
    public float scoringDuration = 6f;
    public string[] bodyParts = Array.Empty<string>();
    public float holdSeconds;
    public bool applyFormalActionScoreMapping = true;
    public string voiceClipName;
    public RecordedCoachMotionFrame[] frames = Array.Empty<RecordedCoachMotionFrame>();
}

[Serializable]
public sealed class RecordedCoachMotionFrame
{
    public const int MuscleCount = 95;

    public float time;
    public Vector3 bodyPosition;
    public Quaternion bodyRotation;
    public float[] muscles = new float[MuscleCount];

    public static RecordedCoachMotionFrame FromHumanPose(float time, HumanPose pose)
    {
        var frame = new RecordedCoachMotionFrame
        {
            time = time,
            bodyPosition = pose.bodyPosition,
            bodyRotation = pose.bodyRotation,
            muscles = new float[MuscleCount]
        };

        int count = Mathf.Min(MuscleCount, pose.muscles == null ? 0 : pose.muscles.Length);
        for (int i = 0; i < count; i++) frame.muscles[i] = pose.muscles[i];
        return frame;
    }
}
