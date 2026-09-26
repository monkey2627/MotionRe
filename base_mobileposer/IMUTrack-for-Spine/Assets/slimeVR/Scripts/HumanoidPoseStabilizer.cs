using System;
using UnityEngine;

/// <summary>
/// Converts the pose written by VMC back through Unity's Humanoid muscle system,
/// then applies anatomical limits and time-continuous filtering before recording.
/// This keeps source/target VRM bone-axis differences and isolated IMU spikes from
/// being displayed as impossible joint rotations.
/// </summary>
[DefaultExecutionOrder(10000)]
[DisallowMultipleComponent]
public sealed class HumanoidPoseStabilizer : MonoBehaviour
{
    [SerializeField] private Animator avatarAnimator;

    [Header("Anatomical limits")]
    [SerializeField, Range(0.5f, 1f)] private float maximumMuscleValue = 0.98f;
    [SerializeField, Range(0.35f, 1f)] private float torsoMuscleLimit = 0.82f;
    [SerializeField, Range(0.35f, 1f)] private float neckAndHeadMuscleLimit = 0.88f;

    [Header("Lower-body plausibility")]
    [SerializeField, Range(0.2f, 0.8f)] private float hipSideLimit = 0.46f;
    [SerializeField, Range(0.15f, 0.7f)] private float hipTwistLimit = 0.38f;
    [SerializeField, Range(0.1f, 0.5f)] private float bentHipSideLimit = 0.30f;
    [SerializeField, Range(0.1f, 0.5f)] private float bentHipTwistLimit = 0.25f;
    [SerializeField, Range(0.05f, 0.4f)] private float kneeTwistLimit = 0.16f;
    [SerializeField, Range(0.15f, 0.7f)] private float footTwistLimit = 0.38f;
    [SerializeField, Range(0.6f, 1f)] private float kneeExtensionLimit = 0.88f;

    [Header("Temporal stability")]
    [SerializeField, Min(0f)] private float response = 22f;
    [SerializeField, Min(0.1f)] private float maximumMuscleSpeed = 9f;
    [SerializeField, Min(1f)] private float maximumBodyAngularSpeed = 720f;
    [SerializeField, Min(0.1f)] private float maximumBodySpeed = 4f;
    [SerializeField] private bool lockHorizontalBodyPosition = true;
    [SerializeField] private bool lockVerticalBodyPosition = true;

    private HumanPoseHandler poseHandler;
    private HumanPose sampledPose;
    private HumanPose stabilizedPose;
    private float[] muscleMinimums;
    private float[] muscleMaximums;
    private int leftHipFrontBackIndex = -1;
    private int leftHipSideIndex = -1;
    private int leftHipTwistIndex = -1;
    private int rightHipFrontBackIndex = -1;
    private int rightHipSideIndex = -1;
    private int rightHipTwistIndex = -1;
    private bool hasStablePose;
    private Vector3 bodyPositionAnchor;
    private int boundAvatarInstanceId;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void InstallForRecorderScene()
    {
        PoseFbxRecorder[] recorders = FindObjectsOfType<PoseFbxRecorder>();
        for (int index = 0; index < recorders.Length; index++)
        {
            PoseFbxRecorder recorder = recorders[index];
            if (recorder == null)
            {
                continue;
            }

            HumanoidPoseStabilizer stabilizer =
                recorder.GetComponent<HumanoidPoseStabilizer>();
            if (stabilizer == null)
            {
                stabilizer = recorder.gameObject.AddComponent<HumanoidPoseStabilizer>();
            }

            stabilizer.Configure(recorder.AvatarAnimator);
        }

        EVMC4U.ExternalReceiver[] receivers = FindObjectsOfType<EVMC4U.ExternalReceiver>();
        for (int index = 0; index < receivers.Length; index++)
        {
            EVMC4U.ExternalReceiver receiver = receivers[index];
            if (receiver == null || receiver.Model == null)
            {
                continue;
            }

            // VMC remains the SlimeVR pose source. The low-pass filter removes
            // transport jitter before the Humanoid anatomical pass below.
            receiver.BoneRotationFilterEnable = true;
            receiver.BoneFilter = 0.65f;
        }
    }

    private void Awake()
    {
        ResolveAnimator();
        EnsurePoseHandler();
    }

    private void OnEnable()
    {
        hasStablePose = false;
    }

    private void LateUpdate()
    {
        if (!EnsurePoseHandler())
        {
            return;
        }

        try
        {
            poseHandler.GetHumanPose(ref sampledPose);
        }
        catch (Exception exception)
        {
            Debug.LogWarning("[Humanoid Pose Stabilizer] Failed to read the avatar pose: " +
                             exception.Message, this);
            RecreatePoseHandler();
            return;
        }

        if (sampledPose.muscles == null ||
            sampledPose.muscles.Length != HumanTrait.MuscleCount)
        {
            return;
        }

        if (!hasStablePose)
        {
            SeedFromCurrentPose();
        }
        else
        {
            StabilizeCurrentPose();
        }

        try
        {
            // A Get/Set round trip is intentional: Unity converts the received local
            // rotations into this avatar's Humanoid definition and reapplies them using
            // this avatar's own bind axes and muscle ranges.
            poseHandler.SetHumanPose(ref stabilizedPose);
        }
        catch (Exception exception)
        {
            Debug.LogWarning("[Humanoid Pose Stabilizer] Failed to apply the stabilized pose: " +
                             exception.Message, this);
            RecreatePoseHandler();
        }
    }

    /// <summary>
    /// Re-seeds the temporal filter on the next frame. This is useful after a
    /// SlimeVR reset or when changing avatars.
    /// </summary>
    public void ResetStabilizer()
    {
        hasStablePose = false;
    }

    public void Configure(Animator animator)
    {
        if (avatarAnimator == animator)
        {
            return;
        }

        avatarAnimator = animator;
        RecreatePoseHandler();
        EnsurePoseHandler();
    }

    private void SeedFromCurrentPose()
    {
        stabilizedPose.bodyPosition = IsFinite(sampledPose.bodyPosition)
            ? sampledPose.bodyPosition
            : Vector3.zero;
        bodyPositionAnchor = stabilizedPose.bodyPosition;
        stabilizedPose.bodyRotation = NormalizeOrFallback(sampledPose.bodyRotation,
            Quaternion.identity);

        for (int index = 0; index < HumanTrait.MuscleCount; index++)
        {
            float value = IsFinite(sampledPose.muscles[index])
                ? sampledPose.muscles[index]
                : 0f;
            stabilizedPose.muscles[index] = ClampMuscle(index, value);
        }

        ConstrainBentLegs(stabilizedPose.muscles);

        hasStablePose = true;
    }

    private void StabilizeCurrentPose()
    {
        float deltaTime = Mathf.Clamp(Time.unscaledDeltaTime, 1f / 240f, 0.1f);
        float blend = response <= 0f ? 1f : 1f - Mathf.Exp(-response * deltaTime);

        Vector3 targetBodyPosition = IsFinite(sampledPose.bodyPosition)
            ? sampledPose.bodyPosition
            : stabilizedPose.bodyPosition;
        if (lockHorizontalBodyPosition)
        {
            targetBodyPosition.x = bodyPositionAnchor.x;
            targetBodyPosition.z = bodyPositionAnchor.z;
        }
        if (lockVerticalBodyPosition)
        {
            targetBodyPosition.y = bodyPositionAnchor.y;
        }
        Vector3 smoothedBodyPosition = Vector3.Lerp(stabilizedPose.bodyPosition,
            targetBodyPosition, blend);
        stabilizedPose.bodyPosition = Vector3.MoveTowards(stabilizedPose.bodyPosition,
            smoothedBodyPosition, maximumBodySpeed * deltaTime);

        Quaternion targetBodyRotation = NormalizeOrFallback(sampledPose.bodyRotation,
            stabilizedPose.bodyRotation);
        Quaternion rateLimitedBodyRotation = Quaternion.RotateTowards(
            stabilizedPose.bodyRotation,
            targetBodyRotation,
            maximumBodyAngularSpeed * deltaTime);
        stabilizedPose.bodyRotation = Quaternion.Slerp(stabilizedPose.bodyRotation,
            rateLimitedBodyRotation, blend);

        float maximumMuscleStep = maximumMuscleSpeed * deltaTime;
        for (int index = 0; index < HumanTrait.MuscleCount; index++)
        {
            float previous = stabilizedPose.muscles[index];
            float target = IsFinite(sampledPose.muscles[index])
                ? sampledPose.muscles[index]
                : previous;
            target = ClampMuscle(index, target);

            float smoothed = Mathf.Lerp(previous, target, blend);
            stabilizedPose.muscles[index] = Mathf.MoveTowards(previous, smoothed,
                maximumMuscleStep);
        }


        // Hip abduction and axial rotation that is acceptable while standing can
        // make the knees cross when both thighs are flexed for sitting or squatting.
        // Tighten only those channels as the hip bends, preserving ordinary walking
        // and allowing a lying pose without rotating the knee as a ball joint.
        ConstrainBentLegs(stabilizedPose.muscles);
    }

    private bool EnsurePoseHandler()
    {
        ResolveAnimator();
        if (!IsValidHumanoid(avatarAnimator))
        {
            DisposePoseHandler();
            return false;
        }

        int avatarInstanceId = avatarAnimator.avatar.GetInstanceID();
        if (poseHandler != null && boundAvatarInstanceId == avatarInstanceId)
        {
            return true;
        }

        DisposePoseHandler();
        try
        {
            poseHandler = new HumanPoseHandler(avatarAnimator.avatar,
                avatarAnimator.transform);
            sampledPose = CreatePose();
            stabilizedPose = CreatePose();
            BuildMuscleLimits();
            boundAvatarInstanceId = avatarInstanceId;
            hasStablePose = false;
            return true;
        }
        catch (Exception exception)
        {
            Debug.LogWarning("[Humanoid Pose Stabilizer] Unable to initialize: " +
                             exception.Message, this);
            DisposePoseHandler();
            return false;
        }
    }

    private void ResolveAnimator()
    {
        if (avatarAnimator == null)
        {
            avatarAnimator = GetComponentInChildren<Animator>();
        }
    }

    private void RecreatePoseHandler()
    {
        DisposePoseHandler();
        hasStablePose = false;
    }

    private void OnDisable()
    {
        DisposePoseHandler();
    }

    private void OnDestroy()
    {
        DisposePoseHandler();
    }

    private void DisposePoseHandler()
    {
        poseHandler?.Dispose();
        poseHandler = null;
        boundAvatarInstanceId = 0;
    }

    private void BuildMuscleLimits()
    {
        muscleMinimums = new float[HumanTrait.MuscleCount];
        muscleMaximums = new float[HumanTrait.MuscleCount];
        leftHipFrontBackIndex = -1;
        leftHipSideIndex = -1;
        leftHipTwistIndex = -1;
        rightHipFrontBackIndex = -1;
        rightHipSideIndex = -1;
        rightHipTwistIndex = -1;

        for (int index = 0; index < muscleMinimums.Length; index++)
        {
            string muscleName = HumanTrait.MuscleName[index] ?? string.Empty;
            float limit = maximumMuscleValue;
            if (ContainsIgnoreCase(muscleName, "Spine") ||
                ContainsIgnoreCase(muscleName, "Chest"))
            {
                limit = Mathf.Min(limit, torsoMuscleLimit);
            }
            else if (ContainsIgnoreCase(muscleName, "Neck") ||
                     ContainsIgnoreCase(muscleName, "Head"))
            {
                limit = Mathf.Min(limit, neckAndHeadMuscleLimit);
            }

            if (ContainsIgnoreCase(muscleName, "Upper Leg In-Out"))
            {
                limit = Mathf.Min(limit, hipSideLimit);
                SetHipChannelIndex(muscleName, index, ref leftHipSideIndex,
                    ref rightHipSideIndex);
            }
            else if (ContainsIgnoreCase(muscleName, "Upper Leg Twist In-Out"))
            {
                limit = Mathf.Min(limit, hipTwistLimit);
                SetHipChannelIndex(muscleName, index, ref leftHipTwistIndex,
                    ref rightHipTwistIndex);
            }
            else if (ContainsIgnoreCase(muscleName, "Upper Leg Front-Back"))
            {
                SetHipChannelIndex(muscleName, index, ref leftHipFrontBackIndex,
                    ref rightHipFrontBackIndex);
            }
            else if (ContainsIgnoreCase(muscleName, "Lower Leg Twist In-Out"))
            {
                limit = Mathf.Min(limit, kneeTwistLimit);
            }
            else if (ContainsIgnoreCase(muscleName, "Foot Twist In-Out"))
            {
                limit = Mathf.Min(limit, footTwistLimit);
            }

            muscleMinimums[index] = -limit;
            muscleMaximums[index] = limit;

            // In Unity's Humanoid convention negative Lower Leg Stretch is knee
            // flexion and positive is extension. Keep the full useful bend range,
            // but leave a margin before the avatar's configured hyperextension end.
            if (ContainsIgnoreCase(muscleName, "Lower Leg Stretch"))
            {
                muscleMaximums[index] = Mathf.Min(muscleMaximums[index],
                    kneeExtensionLimit);
            }
        }
    }

    private float ClampMuscle(int index, float value)
    {
        return Mathf.Clamp(value, muscleMinimums[index], muscleMaximums[index]);
    }

    private void ConstrainBentLegs(float[] muscles)
    {
        ConstrainBentLeg(muscles, leftHipFrontBackIndex, leftHipSideIndex,
            leftHipTwistIndex);
        ConstrainBentLeg(muscles, rightHipFrontBackIndex, rightHipSideIndex,
            rightHipTwistIndex);
    }

    private void ConstrainBentLeg(float[] muscles, int frontBackIndex, int sideIndex,
        int twistIndex)
    {
        if (!IsMuscleIndex(frontBackIndex) || !IsMuscleIndex(sideIndex) ||
            !IsMuscleIndex(twistIndex))
        {
            return;
        }

        // A negative Upper Leg Front-Back value is forward hip flexion. The blend
        // begins before a deep squat and reaches full strength at a seated thigh.
        float bentAmount = Mathf.InverseLerp(0.45f, 0.9f,
            -muscles[frontBackIndex]);
        float sideLimit = Mathf.Lerp(hipSideLimit, bentHipSideLimit, bentAmount);
        float twistLimit = Mathf.Lerp(hipTwistLimit, bentHipTwistLimit, bentAmount);
        muscles[sideIndex] = Mathf.Clamp(muscles[sideIndex], -sideLimit, sideLimit);
        muscles[twistIndex] = Mathf.Clamp(muscles[twistIndex], -twistLimit,
            twistLimit);
    }

    private static void SetHipChannelIndex(string muscleName, int index,
        ref int leftIndex, ref int rightIndex)
    {
        if (ContainsIgnoreCase(muscleName, "Left"))
        {
            leftIndex = index;
        }
        else if (ContainsIgnoreCase(muscleName, "Right"))
        {
            rightIndex = index;
        }
    }

    private static bool IsMuscleIndex(int index)
    {
        return index >= 0 && index < HumanTrait.MuscleCount;
    }

    private static HumanPose CreatePose()
    {
        return new HumanPose
        {
            bodyRotation = Quaternion.identity,
            muscles = new float[HumanTrait.MuscleCount]
        };
    }

    private static bool IsValidHumanoid(Animator animator)
    {
        return animator != null && animator.avatar != null && animator.avatar.isValid &&
               animator.avatar.isHuman;
    }

    private static Quaternion NormalizeOrFallback(Quaternion value, Quaternion fallback)
    {
        if (!IsFinite(value))
        {
            return fallback;
        }

        float magnitudeSquared = value.x * value.x + value.y * value.y +
                                 value.z * value.z + value.w * value.w;
        if (magnitudeSquared < 0.000001f)
        {
            return fallback;
        }

        float inverseMagnitude = 1f / Mathf.Sqrt(magnitudeSquared);
        return new Quaternion(value.x * inverseMagnitude, value.y * inverseMagnitude,
            value.z * inverseMagnitude, value.w * inverseMagnitude);
    }

    private static bool ContainsIgnoreCase(string value, string text)
    {
        return value.IndexOf(text, StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static bool IsFinite(Vector3 value)
    {
        return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
    }

    private static bool IsFinite(Quaternion value)
    {
        return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z) &&
               IsFinite(value.w);
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        maximumMuscleValue = Mathf.Clamp(maximumMuscleValue, 0.5f, 1f);
        torsoMuscleLimit = Mathf.Clamp(torsoMuscleLimit, 0.35f, maximumMuscleValue);
        neckAndHeadMuscleLimit = Mathf.Clamp(neckAndHeadMuscleLimit, 0.35f,
            maximumMuscleValue);
        hipSideLimit = Mathf.Clamp(hipSideLimit, 0.2f, maximumMuscleValue);
        hipTwistLimit = Mathf.Clamp(hipTwistLimit, 0.15f, maximumMuscleValue);
        bentHipSideLimit = Mathf.Clamp(bentHipSideLimit, 0.1f, hipSideLimit);
        bentHipTwistLimit = Mathf.Clamp(bentHipTwistLimit, 0.1f, hipTwistLimit);
        kneeTwistLimit = Mathf.Clamp(kneeTwistLimit, 0.05f, 0.4f);
        footTwistLimit = Mathf.Clamp(footTwistLimit, 0.15f, 0.7f);
        kneeExtensionLimit = Mathf.Clamp(kneeExtensionLimit, 0.6f,
            maximumMuscleValue);
        response = Mathf.Max(0f, response);
        maximumMuscleSpeed = Mathf.Max(0.1f, maximumMuscleSpeed);
        maximumBodyAngularSpeed = Mathf.Max(1f, maximumBodyAngularSpeed);
        maximumBodySpeed = Mathf.Max(0.1f, maximumBodySpeed);
        if (Application.isPlaying && muscleMinimums != null)
        {
            BuildMuscleLimits();
        }
    }
#endif
}
