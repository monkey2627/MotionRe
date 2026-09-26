using System;

namespace SpineFlow.MotionEditing
{
    [Serializable]
    public sealed class RecordedMotionEditDefinition
    {
        public int formatVersion = 1;
        public string sourceClipAssetPath;
        public string sourceClipName;
        public float sourceClipLengthSeconds;
        public float sourceFrameRate;
        public int totalFrames;
        public string actionId;
        public string displayName;
        public int sampleFps = 30;
        public int setCount = 1;
        public string savedAtUtc;
        public RecordedMotionKeyframeDefinition[] keyframes = Array.Empty<RecordedMotionKeyframeDefinition>();
    }

    [Serializable]
    public sealed class RecordedMotionKeyframeDefinition
    {
        public string name;
        public int frame;
        public string[] bodyParts = Array.Empty<string>();
        public float holdSeconds;
        public string followingSegmentName;
        public float followingSegmentDurationSeconds;
    }
}
