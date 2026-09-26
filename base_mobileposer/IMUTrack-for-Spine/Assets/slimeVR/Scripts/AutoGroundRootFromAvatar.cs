using UnityEngine;

[DefaultExecutionOrder(10002)]
public sealed class AutoGroundRootFromAvatar : MonoBehaviour
{
    [SerializeField] private Transform targetRoot;
    [SerializeField] private Transform avatarRoot;
    [SerializeField] private Animator avatarAnimator;
    [SerializeField] private float floorY = 0f;
    [SerializeField] private float groundOffset = 0.01f;
    [SerializeField] private float smoothing = 18f;
    [SerializeField] private float maxCorrectionPerSecond = 3f;
    [SerializeField] private bool useHumanoidBones = true;
    [Tooltip("Use only as a fallback for avatars without valid Humanoid foot bones. " +
             "Animated skinned-renderer bounds can move when hands or other limbs move " +
             "and must not drive the recorder preview root every frame.")]
    [SerializeField] private bool useRendererBounds = false;
    [SerializeField] private bool includeInactiveRenderers = false;
    [SerializeField] private bool keepExactGroundHeight = true;
    [SerializeField, Min(0.1f)] private float maximumGroundOffsetFromInitial = 2f;
    [SerializeField, Min(0.1f)] private float maximumRendererDepthBelowFeet = 0.75f;
    [SerializeField] private bool smoothContactTransitions = true;
    [SerializeField, Min(0.1f)] private float contactTransitionSpeed = 1.2f;
    [SerializeField, Min(0f)] private float contactTransitionResponse = 10f;
    [SerializeField, Min(0f)] private float groundSnapTolerance = 0.012f;
    [SerializeField] private bool holdPreviewHeightWhenLying = true;
    [SerializeField, Range(0.2f, 0.8f)] private float lyingEnterVerticalRatio = 0.62f;
    [SerializeField, Range(0.3f, 0.95f)] private float lyingExitVerticalRatio = 0.75f;
    [SerializeField] private bool lockHorizontalPosition = true;
    [SerializeField] private bool lockRootRotation = true;

    private Renderer[] renderers;
    private Transform[] groundBones;
    private Transform hipsBone;
    private Transform headBone;
    private Vector3 initialRootPosition;
    private Quaternion initialRootRotation;
    private bool hasGroundSample;
    private bool isLyingHeightLocked;
    private float lyingRootY;

    public void RebindAvatar(Transform newAvatarRoot, Animator newAnimator)
    {
        avatarRoot = newAvatarRoot;
        avatarAnimator = newAnimator;
        hasGroundSample = false;
        isLyingHeightLocked = false;
        CacheHumanoidBones();
        RefreshRenderers();
    }

    private void Awake()
    {
        if (targetRoot == null)
        {
            targetRoot = transform;
        }

        if (avatarRoot == null)
        {
            avatarRoot = targetRoot;
        }

        if (avatarAnimator == null)
        {
            avatarAnimator = avatarRoot.GetComponentInChildren<Animator>();
        }

        CacheHumanoidBones();
        RefreshRenderers();

        initialRootPosition = targetRoot.position;
        initialRootRotation = targetRoot.rotation;
    }

    private void LateUpdate()
    {
        if (targetRoot == null || avatarRoot == null)
        {
            return;
        }

        if (useHumanoidBones && (groundBones == null || groundBones.Length == 0))
        {
            CacheHumanoidBones();
        }

        if (useRendererBounds && (renderers == null || renderers.Length == 0))
        {
            RefreshRenderers();
        }

        if (UpdateLyingHeightLock())
        {
            Vector3 lyingPosition = targetRoot.position;
            lyingPosition.y = lyingRootY;
            targetRoot.position = lyingPosition;
            ApplyHorizontalAndRotationLocks();
            return;
        }

        if (!TryGetLowestY(out float lowestY))
        {
            return;
        }

        float desiredDelta = floorY + groundOffset - lowestY;
        float deltaTime = Mathf.Max(Time.deltaTime, 0.0001f);
        float limitedDelta;
        if (!hasGroundSample)
        {
            // Establish the common floor immediately once. Later contact changes
            // are continuous so a foot-to-back transition cannot teleport the rig.
            limitedDelta = desiredDelta;
            hasGroundSample = true;
        }
        else if (keepExactGroundHeight &&
                 (!smoothContactTransitions ||
                  Mathf.Abs(desiredDelta) <= groundSnapTolerance))
        {
            limitedDelta = desiredDelta;
        }
        else if (smoothContactTransitions)
        {
            float blend = contactTransitionResponse <= 0f
                ? 1f
                : 1f - Mathf.Exp(-contactTransitionResponse * deltaTime);
            float blendedDelta = Mathf.Lerp(0f, desiredDelta, blend);
            limitedDelta = Mathf.Clamp(blendedDelta,
                -contactTransitionSpeed * deltaTime,
                contactTransitionSpeed * deltaTime);
        }
        else
        {
            float smoothedDelta = Mathf.Lerp(0f, desiredDelta,
                1f - Mathf.Exp(-smoothing * deltaTime));
            limitedDelta = Mathf.Clamp(
                smoothedDelta,
                -maxCorrectionPerSecond * deltaTime,
                maxCorrectionPerSecond * deltaTime);
        }

        // The correction must be symmetric. The old lower bound allowed the root
        // to move only 10 cm downward, so a squat shortened the leg chain while the
        // feet remained visibly suspended above the fixed viewport floor. A lying
        // pose has the same requirement when another body surface becomes lowest.
        // The symmetric safety range still prevents corrupt bounds from launching
        // the avatar out of the preview.
        float desiredRootY = Mathf.Clamp(
            targetRoot.position.y + limitedDelta,
            initialRootPosition.y - maximumGroundOffsetFromInitial,
            initialRootPosition.y + maximumGroundOffsetFromInitial);
        limitedDelta = desiredRootY - targetRoot.position.y;

        targetRoot.position += Vector3.up * limitedDelta;

        ApplyHorizontalAndRotationLocks();
    }

    private bool UpdateLyingHeightLock()
    {
        if (!holdPreviewHeightWhenLying || hipsBone == null || headBone == null)
        {
            isLyingHeightLocked = false;
            return false;
        }

        Vector3 torsoAxis = headBone.position - hipsBone.position;
        if (torsoAxis.sqrMagnitude < 0.01f)
        {
            return isLyingHeightLocked;
        }

        float verticalRatio = Mathf.Abs(Vector3.Dot(torsoAxis.normalized, Vector3.up));
        if (!isLyingHeightLocked && verticalRatio <= lyingEnterVerticalRatio)
        {
            // Preserve the already-correct sitting/crouching preview height at the
            // moment the torso changes into a lying orientation. This is a viewer
            // framing rule; it deliberately avoids lowering the avatar until its
            // back touches the bottom edge of the viewport.
            lyingRootY = targetRoot.position.y;
            isLyingHeightLocked = true;
        }
        else if (isLyingHeightLocked && verticalRatio >= lyingExitVerticalRatio)
        {
            isLyingHeightLocked = false;
        }

        return isLyingHeightLocked;
    }

    private void ApplyHorizontalAndRotationLocks()
    {
        if (lockHorizontalPosition)
        {
            Vector3 position = targetRoot.position;
            position.x = initialRootPosition.x;
            position.z = initialRootPosition.z;
            targetRoot.position = position;
        }

        if (lockRootRotation)
        {
            targetRoot.rotation = initialRootRotation;
        }
    }

    [ContextMenu("Refresh Renderers")]
    private void RefreshRenderers()
    {
        if (avatarRoot == null)
        {
            renderers = new Renderer[0];
            return;
        }

        renderers = avatarRoot.GetComponentsInChildren<Renderer>(includeInactiveRenderers);
    }

    private bool TryGetLowestY(out float lowestY)
    {
        float lowestBoneY = float.PositiveInfinity;
        float lowestRendererY = float.PositiveInfinity;
        bool foundBone = false;
        bool foundRenderer = false;

        if (useHumanoidBones && groundBones != null)
        {
            for (int i = 0; i < groundBones.Length; i++)
            {
                Transform bone = groundBones[i];
                if (bone == null)
                {
                    continue;
                }

                lowestBoneY = Mathf.Min(lowestBoneY, bone.position.y);
                foundBone = true;
            }
        }

        if (!useRendererBounds)
        {
            lowestY = lowestBoneY;
            return foundBone;
        }

        for (int i = 0; i < renderers.Length; i++)
        {
            Renderer renderer = renderers[i];
            if (renderer == null || !renderer.enabled)
            {
                continue;
            }

            float rendererY = renderer.bounds.min.y;
            if (float.IsNaN(rendererY) || float.IsInfinity(rendererY))
            {
                continue;
            }

            lowestRendererY = Mathf.Min(lowestRendererY, rendererY);
            foundRenderer = true;
        }

        if (!foundRenderer)
        {
            lowestY = lowestBoneY;
            return foundBone;
        }

        if (!foundBone)
        {
            lowestY = lowestRendererY;
            return true;
        }

        // Mesh bounds supply the sole/body-surface thickness missing from bone
        // positions and make lying poses rest on the visible body. Reject only an
        // implausibly deep bound, which usually means a stale imported VRM bound.
        lowestY = lowestRendererY >= lowestBoneY - maximumRendererDepthBelowFeet
            ? Mathf.Min(lowestBoneY, lowestRendererY)
            : lowestBoneY;
        return true;
    }

#if UNITY_EDITOR
    private void OnValidate()
    {
        maximumGroundOffsetFromInitial = Mathf.Max(0.1f,
            maximumGroundOffsetFromInitial);
        maximumRendererDepthBelowFeet = Mathf.Max(0.1f,
            maximumRendererDepthBelowFeet);
        contactTransitionSpeed = Mathf.Max(0.1f, contactTransitionSpeed);
        contactTransitionResponse = Mathf.Max(0f, contactTransitionResponse);
        groundSnapTolerance = Mathf.Max(0f, groundSnapTolerance);
        lyingEnterVerticalRatio = Mathf.Clamp(lyingEnterVerticalRatio, 0.2f, 0.8f);
        lyingExitVerticalRatio = Mathf.Clamp(lyingExitVerticalRatio,
            lyingEnterVerticalRatio + 0.05f, 0.95f);
        groundOffset = Mathf.Max(0f, groundOffset);
        smoothing = Mathf.Max(0f, smoothing);
        maxCorrectionPerSecond = Mathf.Max(0f, maxCorrectionPerSecond);
    }
#endif

    private void CacheHumanoidBones()
    {
        if (avatarAnimator == null || !avatarAnimator.isHuman)
        {
            groundBones = new Transform[0];
            return;
        }

        groundBones = new[]
        {
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftFoot),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightFoot),
            avatarAnimator.GetBoneTransform(HumanBodyBones.LeftToes),
            avatarAnimator.GetBoneTransform(HumanBodyBones.RightToes),
        };
        hipsBone = avatarAnimator.GetBoneTransform(HumanBodyBones.Hips);
        headBone = avatarAnimator.GetBoneTransform(HumanBodyBones.Head);
    }
}
