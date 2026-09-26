using System;
using System.Collections.Generic;
using System.IO;
using SpineFlow.PoseRecording;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[DefaultExecutionOrder(20000)]
public sealed class PoseFbxRecorder : MonoBehaviour
{
    [SerializeField] private Transform recordRoot;
    [SerializeField] private Animator avatarAnimator;
    [SerializeField] public string takeName = "SlimeVR_Take";
    [SerializeField] private int sampleRate = 30;
    [SerializeField] public string outputFolder = "Assets/Recordings";
    [SerializeField] private KeyCode startKey = KeyCode.F9;
    [SerializeField] private KeyCode stopKey = KeyCode.F10;
    [SerializeField] private bool showControls = true;
    [SerializeField] private bool exportWhenStopped = true;

    private readonly List<TransformTrack> tracks = new List<TransformTrack>();
    private readonly List<RecordedCoachMotionFrame> runtimeHumanoidFrames =
        new List<RecordedCoachMotionFrame>();
    private bool isRecording;
    private float recordingStartTime;
    private float nextSampleTime;
    private int recordingIndex = 1;
    private string lastExportPath = string.Empty;
    private AnimationClip lastExportedClip;
    private HumanPoseHandler humanPoseHandler;
    private HumanPose humanPose;
    private RootMotion.BakerMuscle[] humanoidMuscles;
    private RootMotion.BakerHumanoidQT humanoidRoot;
    private RootMotion.BakerHumanoidQT leftFootIK;
    private RootMotion.BakerHumanoidQT rightFootIK;
    private RootMotion.BakerHumanoidQT leftHandIK;
    private RootMotion.BakerHumanoidQT rightHandIK;
    private Quaternion lastBodyRotation = Quaternion.identity;
    private bool recordsHumanoidPose;
    private string runtimeActionId = "recorded_motion";
    private string runtimeDisplayName = "Recorded Motion";
    private string runtimeTrajectoryName;
    private string lastRuntimeRecordingPath = string.Empty;
    private RuntimePoseRecordingPackage lastRuntimeRecording;

    public Animator AvatarAnimator => avatarAnimator;
    public string LastExportPath => lastExportPath;
    public AnimationClip LastExportedClip => lastExportedClip;
    public string LastRuntimeRecordingPath => lastRuntimeRecordingPath;
    public RuntimePoseRecordingPackage LastRuntimeRecording => lastRuntimeRecording;
    public bool IsRecording => isRecording;

    /// <summary>
    /// Rebinds the recorder after the live preview avatar is replaced.
    /// Switching while recording is deliberately rejected so one take can never
    /// contain transform tracks from two different skeletons.
    /// </summary>
    public bool TrySetAvatarAnimator(Animator animator, out string error)
    {
        error = null;
        if (isRecording)
        {
            error = "录制过程中不能切换人物模型";
            return false;
        }

        if (animator == null || animator.avatar == null || !animator.isHuman)
        {
            error = "所选模型没有有效的 Humanoid Animator";
            return false;
        }

        DisposeHumanoidRecorder();
        tracks.Clear();
        runtimeHumanoidFrames.Clear();
        avatarAnimator = animator;
        return true;
    }

    public void ConfigureRuntimeMetadata(string actionId, string displayName, string trajectoryName)
    {
        runtimeActionId = string.IsNullOrWhiteSpace(actionId) ? "recorded_motion" : actionId.Trim();
        runtimeDisplayName = string.IsNullOrWhiteSpace(displayName) ? runtimeActionId : displayName.Trim();
        runtimeTrajectoryName = string.IsNullOrWhiteSpace(trajectoryName)
            ? DateTime.Now.ToString("yyyyMMdd_HHmmss")
            : trajectoryName.Trim();
    }

    private void Awake()
    {
        if (recordRoot == null)
        {
            recordRoot = transform;
        }

        if (avatarAnimator == null)
        {
            avatarAnimator = recordRoot.GetComponentInChildren<Animator>();
        }
    }

    private void Update()
    {
        if (Input.GetKeyDown(startKey))
        {
            StartRecording();
        }

        if (Input.GetKeyDown(stopKey))
        {
            StopRecording();
        }
    }

    private void LateUpdate()
    {
        if (!isRecording || Time.unscaledTime + 0.0001f < nextSampleTime)
        {
            return;
        }

        Sample(Time.unscaledTime - recordingStartTime);
        nextSampleTime += 1f / Mathf.Max(1, sampleRate);
    }

    private void OnGUI()
    {
        if (!showControls)
        {
            return;
        }

        GUILayout.BeginArea(new Rect(12, 98, 360, 140), GUI.skin.box);
        GUILayout.Label(isRecording
            ? $"Recording {tracks.Count} transform(s) | {GetSampleCount()} frame(s)"
            : "Pose FBX Recorder");

        GUILayout.BeginHorizontal();
        GUI.enabled = !isRecording;
        if (GUILayout.Button($"Start ({startKey})", GUILayout.Height(32)))
        {
            StartRecording();
        }

        GUI.enabled = isRecording;
        if (GUILayout.Button($"Stop And Export ({stopKey})", GUILayout.Height(32)))
        {
            StopRecording();
        }

        GUI.enabled = true;
        GUILayout.EndHorizontal();

        if (!string.IsNullOrEmpty(lastExportPath))
        {
            GUILayout.Label($"Last export: {lastExportPath}");
        }

        GUILayout.EndArea();
    }

    [ContextMenu("Start Recording")]
    public void StartRecording()
    {
        if (isRecording)
        {
            return;
        }

        if (avatarAnimator == null)
        {
            Debug.LogError("[Pose FBX Recorder] No avatar Animator is assigned.", this);
            return;
        }

        tracks.Clear();
        runtimeHumanoidFrames.Clear();
        lastRuntimeRecording = null;
        lastRuntimeRecordingPath = string.Empty;
        DisposeHumanoidRecorder();

        Transform clipRoot = avatarAnimator.transform;
        Transform[] transforms = clipRoot.GetComponentsInChildren<Transform>(true);
        for (int i = 0; i < transforms.Length; i++)
        {
            tracks.Add(new TransformTrack(transforms[i], GetRelativePath(transforms[i], clipRoot)));
        }

        InitializeHumanoidRecorder();
        recordingStartTime = Time.unscaledTime;
        isRecording = true;

        Sample(0f);
        nextSampleTime = Time.unscaledTime + 1f / Mathf.Max(1, sampleRate);
        Debug.Log($"[Pose FBX Recorder] Started recording {tracks.Count} transform(s).", this);
    }

    [ContextMenu("Stop Recording And Export")]
    public void StopRecording()
    {
        if (!isRecording)
        {
            return;
        }

        Sample(Time.unscaledTime - recordingStartTime);
        isRecording = false;

        if (exportWhenStopped)
        {
            ExportRecording();
        }
        else if (!SaveRuntimePoseRecording(out string runtimeError))
        {
            Debug.LogError("[Pose FBX Recorder] " + runtimeError, this);
        }

        recordingIndex++;
        DisposeHumanoidRecorder();
    }

    [ContextMenu("Export Current Recording")]
    public void ExportRecording()
    {
        if (avatarAnimator == null || tracks.Count == 0 || GetSampleCount() == 0)
        {
            Debug.LogWarning("[Pose FBX Recorder] Nothing to export.", this);
            return;
        }

        if (!SaveRuntimePoseRecording(out string runtimeError))
        {
            Debug.LogError("[Pose FBX Recorder] " + runtimeError, this);
            return;
        }

        lastExportPath = lastRuntimeRecordingPath;
        Debug.Log($"[Pose FBX Recorder] Saved runtime pose recording: {lastRuntimeRecordingPath}", this);

#if UNITY_EDITOR
        string directory = GetOutputDirectory();
        Directory.CreateDirectory(directory);

        string exportName = string.IsNullOrWhiteSpace(runtimeTrajectoryName)
            ? takeName
            : runtimeTrajectoryName;
        string basePath = Path.Combine(directory, MakeSafeFileName(exportName));
        string animAssetPath = ToAssetPath(basePath + ".anim");
        if (string.IsNullOrEmpty(animAssetPath))
        {
            Debug.LogError("[Pose FBX Recorder] Output folder must be inside this Unity project so the recorded AnimationClip can be saved as an asset.", this);
            return;
        }

        if (!recordsHumanoidPose)
        {
            Debug.LogError("[Pose FBX Recorder] The assigned Animator is not a valid Humanoid avatar, so no Humanoid AnimationClip was exported.", this);
            return;
        }

        AnimationClip humanoidClip = SaveHumanoidAnimationClip(animAssetPath);
        lastExportedClip = humanoidClip;
        lastExportPath = AssetDatabase.GetAssetPath(humanoidClip);
        Debug.Log($"[Pose FBX Recorder] Exported Humanoid AnimationClip: {lastExportPath}\n" +
                  $"Runtime pose: {lastRuntimeRecordingPath}", this);

        AssetDatabase.Refresh();
#endif
    }

    private void Sample(float time)
    {
        for (int i = 0; i < tracks.Count; i++)
        {
            Transform transformToRecord = tracks[i].Transform;
            Vector3 localPosition = transformToRecord.localPosition;
            Quaternion localRotation = transformToRecord.localRotation;

            if (transformToRecord == avatarAnimator.transform && recordRoot != null && recordRoot != avatarAnimator.transform)
            {
                localPosition += recordRoot.localPosition;
                localRotation = recordRoot.localRotation * localRotation;
            }

            tracks[i].Samples.Add(new TransformSample(
                time,
                localPosition,
                localRotation,
                transformToRecord.localScale));
        }

        SampleHumanoid(time);
    }

    private void InitializeHumanoidRecorder()
    {
        recordsHumanoidPose = avatarAnimator != null && avatarAnimator.isHuman && avatarAnimator.avatar != null && avatarAnimator.avatar.isValid;
        if (!recordsHumanoidPose)
        {
            return;
        }

        humanPoseHandler = new HumanPoseHandler(avatarAnimator.avatar, avatarAnimator.transform);
        humanoidMuscles = new RootMotion.BakerMuscle[HumanTrait.MuscleCount];
        for (int i = 0; i < humanoidMuscles.Length; i++)
        {
            humanoidMuscles[i] = new RootMotion.BakerMuscle(i);
        }

        humanoidRoot = new RootMotion.BakerHumanoidQT("Root");
        leftFootIK = CreateIKTrack(HumanBodyBones.LeftFoot, AvatarIKGoal.LeftFoot, "LeftFoot");
        rightFootIK = CreateIKTrack(HumanBodyBones.RightFoot, AvatarIKGoal.RightFoot, "RightFoot");
        leftHandIK = CreateIKTrack(HumanBodyBones.LeftHand, AvatarIKGoal.LeftHand, "LeftHand");
        rightHandIK = CreateIKTrack(HumanBodyBones.RightHand, AvatarIKGoal.RightHand, "RightHand");
        lastBodyRotation = Quaternion.identity;
    }

    private RootMotion.BakerHumanoidQT CreateIKTrack(HumanBodyBones bone, AvatarIKGoal goal, string trackName)
    {
        Transform boneTransform = avatarAnimator.GetBoneTransform(bone);
        return boneTransform == null ? null : new RootMotion.BakerHumanoidQT(boneTransform, goal, trackName);
    }

    private void SampleHumanoid(float time)
    {
        if (!recordsHumanoidPose || humanPoseHandler == null)
        {
            return;
        }

        humanPoseHandler.GetHumanPose(ref humanPose);
        Quaternion bodyRotation = RootMotion.BakerUtilities.EnsureQuaternionContinuity(lastBodyRotation, humanPose.bodyRotation);
        lastBodyRotation = bodyRotation;

        var runtimePose = new HumanPose
        {
            bodyPosition = humanPose.bodyPosition,
            bodyRotation = bodyRotation,
            muscles = humanPose.muscles
        };
        runtimeHumanoidFrames.Add(RecordedCoachMotionFrame.FromHumanPose(time, runtimePose));

        for (int i = 0; i < humanoidMuscles.Length; i++)
        {
            humanoidMuscles[i].SetKeyframe(time, humanPose.muscles);
        }

        humanoidRoot.SetKeyframes(time, humanPose.bodyPosition, bodyRotation);

        Vector3 bodyPositionScaled = humanPose.bodyPosition * avatarAnimator.humanScale;
        leftFootIK?.SetIKKeyframes(time, avatarAnimator.avatar, avatarAnimator.transform, avatarAnimator.humanScale, bodyPositionScaled, bodyRotation);
        rightFootIK?.SetIKKeyframes(time, avatarAnimator.avatar, avatarAnimator.transform, avatarAnimator.humanScale, bodyPositionScaled, bodyRotation);
        leftHandIK?.SetIKKeyframes(time, avatarAnimator.avatar, avatarAnimator.transform, avatarAnimator.humanScale, bodyPositionScaled, bodyRotation);
        rightHandIK?.SetIKKeyframes(time, avatarAnimator.avatar, avatarAnimator.transform, avatarAnimator.humanScale, bodyPositionScaled, bodyRotation);
    }

    private void DisposeHumanoidRecorder()
    {
        humanPoseHandler?.Dispose();
        humanPoseHandler = null;
        humanoidMuscles = null;
        humanoidRoot = null;
        leftFootIK = null;
        rightFootIK = null;
        leftHandIK = null;
        rightHandIK = null;
        recordsHumanoidPose = false;
    }

    private int GetSampleCount()
    {
        return tracks.Count == 0 ? 0 : tracks[0].Samples.Count;
    }

    private bool SaveRuntimePoseRecording(out string error)
    {
        error = null;
        if (runtimeHumanoidFrames.Count == 0)
        {
            error = "The assigned Animator is not a valid Humanoid avatar, so no runtime pose was saved.";
            return false;
        }

        int frameCount = runtimeHumanoidFrames.Count;
        for (int trackIndex = 0; trackIndex < tracks.Count; trackIndex++)
            frameCount = Mathf.Min(frameCount, tracks[trackIndex].Samples.Count);
        if (frameCount == 0)
        {
            error = "The recording contains no complete pose frames.";
            return false;
        }

        var package = new RuntimePoseRecordingPackage
        {
            formatVersion = RuntimePoseRecordingPackage.CurrentFormatVersion,
            actionId = runtimeActionId,
            displayName = runtimeDisplayName,
            trajectoryName = string.IsNullOrWhiteSpace(runtimeTrajectoryName)
                ? DateTime.Now.ToString("yyyyMMdd_HHmmss")
                : runtimeTrajectoryName,
            recordedAtUtc = DateTime.UtcNow.ToString("o"),
            avatarName = avatarAnimator == null ? string.Empty : avatarAnimator.gameObject.name,
            sampleRate = Mathf.Max(1, sampleRate),
            frameCount = frameCount,
            transformPaths = new string[tracks.Count],
            frames = new RuntimePoseFrame[frameCount]
        };

        for (int trackIndex = 0; trackIndex < tracks.Count; trackIndex++)
            package.transformPaths[trackIndex] = tracks[trackIndex].Path;

        for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
        {
            RecordedCoachMotionFrame humanoidFrame = runtimeHumanoidFrames[frameIndex];
            var frame = new RuntimePoseFrame
            {
                timeSeconds = humanoidFrame.time,
                bodyPosition = humanoidFrame.bodyPosition,
                bodyRotation = humanoidFrame.bodyRotation,
                muscles = (float[])humanoidFrame.muscles.Clone(),
                transforms = new RuntimeTransformPose[tracks.Count]
            };

            for (int trackIndex = 0; trackIndex < tracks.Count; trackIndex++)
            {
                TransformSample sample = tracks[trackIndex].Samples[frameIndex];
                frame.transforms[trackIndex] = new RuntimeTransformPose
                {
                    localPosition = sample.LocalPosition,
                    localRotation = sample.LocalRotation,
                    localScale = sample.LocalScale
                };
            }

            package.frames[frameIndex] = frame;
        }

        package.durationSeconds = package.frames[frameCount - 1].timeSeconds;
        if (!RuntimePoseRecordingStorage.TrySave(package, out lastRuntimeRecordingPath, out error))
            return false;

        lastRuntimeRecording = package;
        return true;
    }

    private string GetOutputDirectory()
    {
        if (Path.IsPathRooted(outputFolder))
        {
            return outputFolder;
        }

        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", outputFolder));
    }

    private static string GetRelativePath(Transform transform, Transform root)
    {
        if (transform == root)
        {
            return string.Empty;
        }

        var names = new Stack<string>();
        Transform current = transform;
        while (current != null && current != root)
        {
            names.Push(current.name);
            current = current.parent;
        }

        return string.Join("/", names.ToArray());
    }

    private static string MakeSafeFileName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return "RecordedPose";
        }

        foreach (char invalid in Path.GetInvalidFileNameChars())
        {
            value = value.Replace(invalid, '_');
        }

        return value;
    }

#if UNITY_EDITOR
    private AnimationClip SaveHumanoidAnimationClip(string assetPath)
    {
        var clip = new AnimationClip
        {
            name = Path.GetFileNameWithoutExtension(assetPath),
            frameRate = Mathf.Max(1, sampleRate),
            legacy = false
        };

        for (int i = 0; i < humanoidMuscles.Length; i++)
        {
            humanoidMuscles[i].SetCurves(ref clip, 0f, 1f);
        }

        humanoidRoot.SetCurves(ref clip, 0f, 1f);
        leftFootIK?.SetCurves(ref clip, 0f, 1f);
        rightFootIK?.SetCurves(ref clip, 0f, 1f);
        leftHandIK?.SetCurves(ref clip, 0f, 1f);
        rightHandIK?.SetCurves(ref clip, 0f, 1f);

        clip.EnsureQuaternionContinuity();
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(Path.Combine(Application.dataPath, "..", assetPath))) ?? Application.dataPath);

        string uniqueAssetPath = AssetDatabase.GenerateUniqueAssetPath(assetPath);
        AssetDatabase.CreateAsset(clip, uniqueAssetPath);
        AssetDatabase.SaveAssets();
        return clip;
    }

    private static string ToAssetPath(string absolutePath)
    {
        string projectPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..")).Replace("\\", "/");
        string normalized = Path.GetFullPath(absolutePath).Replace("\\", "/");
        if (!normalized.StartsWith(projectPath + "/", StringComparison.OrdinalIgnoreCase))
        {
            return string.Empty;
        }

        return normalized.Substring(projectPath.Length + 1);
    }

#endif

    private sealed class TransformTrack
    {
        public readonly Transform Transform;
        public readonly string Path;
        public readonly List<TransformSample> Samples = new List<TransformSample>();

        public TransformTrack(Transform transform, string path)
        {
            Transform = transform;
            Path = path;
        }
    }

    private readonly struct TransformSample
    {
        public readonly float Time;
        public readonly Vector3 LocalPosition;
        public readonly Quaternion LocalRotation;
        public readonly Vector3 LocalScale;

        public TransformSample(float time, Vector3 localPosition, Quaternion localRotation, Vector3 localScale)
        {
            Time = time;
            LocalPosition = localPosition;
            LocalRotation = localRotation;
            LocalScale = localScale;
        }
    }
}
