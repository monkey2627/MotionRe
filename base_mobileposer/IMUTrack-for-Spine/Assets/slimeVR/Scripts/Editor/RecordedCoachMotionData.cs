#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using UnityEngine;

public static class RecordedCoachMotionSampler
{
    public static RecordedCoachMotionFrame[] SampleClip(
        AnimationClip clip,
        GameObject sampleTarget,
        Animator sampleAnimator,
        float fps)
    {
        if (clip == null || sampleTarget == null || sampleAnimator == null ||
            sampleAnimator.avatar == null || !sampleAnimator.avatar.isHuman)
        {
            return Array.Empty<RecordedCoachMotionFrame>();
        }

        var frames = new List<RecordedCoachMotionFrame>();
        var handler = new HumanPoseHandler(sampleAnimator.avatar, sampleAnimator.transform);
        var pose = new HumanPose();

        try
        {
            float safeFps = Mathf.Max(1f, fps);
            int finalFrame = Mathf.Max(1, Mathf.CeilToInt(clip.length * safeFps));
            for (int i = 0; i <= finalFrame; i++)
            {
                float time = Mathf.Min(i / safeFps, clip.length);
                clip.SampleAnimation(sampleTarget, time);
                handler.GetHumanPose(ref pose);
                frames.Add(RecordedCoachMotionFrame.FromHumanPose(time, pose));

                if (time >= clip.length) break;
            }
        }
        finally
        {
            handler.Dispose();
        }

        return frames.ToArray();
    }
}
#endif
