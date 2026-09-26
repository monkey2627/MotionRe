using UnityEngine;

[DefaultExecutionOrder(10001)]
public sealed class AutoRootMotionFromPose : MonoBehaviour
{
    [SerializeField] private Transform targetRoot;
    [SerializeField] private Animator avatarAnimator;
    [SerializeField] private Transform directionReference;
    [SerializeField] private bool useDirectionReference = false;
    [SerializeField] private float movementGain = 0.75f;
    [SerializeField] private float minimumFootSpeed = 0.08f;
    [SerializeField] private float maximumSpeed = 1.6f;
    [SerializeField] private float smoothing = 8f;
    [SerializeField] private bool lockVerticalPosition = true;

    private Transform leftFoot;
    private Transform rightFoot;
    private Vector3 previousLeftFootLocal;
    private Vector3 previousRightFootLocal;
    private bool hasPreviousSample;
    private float currentSpeed;
    private float initialY;
    private AutoGroundRootFromAvatar groundController;

    private void Awake()
    {
        if (targetRoot == null)
        {
            targetRoot = transform;
        }

        if (avatarAnimator == null)
        {
            avatarAnimator = GetComponentInChildren<Animator>();
        }

        if (directionReference == null)
        {
            directionReference = targetRoot;
        }

        initialY = targetRoot.position.y;
        groundController = GetComponent<AutoGroundRootFromAvatar>();
        CacheBones();
    }

    private void LateUpdate()
    {
        if (targetRoot == null || avatarAnimator == null)
        {
            return;
        }

        if (leftFoot == null || rightFoot == null)
        {
            CacheBones();
            if (leftFoot == null || rightFoot == null)
            {
                return;
            }
        }

        Vector3 leftFootLocal = targetRoot.InverseTransformPoint(leftFoot.position);
        Vector3 rightFootLocal = targetRoot.InverseTransformPoint(rightFoot.position);

        if (!hasPreviousSample)
        {
            previousLeftFootLocal = leftFootLocal;
            previousRightFootLocal = rightFootLocal;
            hasPreviousSample = true;
            return;
        }

        float deltaTime = Mathf.Max(Time.deltaTime, 0.0001f);
        float leftSpeed = Vector3.Distance(leftFootLocal, previousLeftFootLocal) / deltaTime;
        float rightSpeed = Vector3.Distance(rightFootLocal, previousRightFootLocal) / deltaTime;
        float footMotion = Mathf.Max(leftSpeed, rightSpeed);
        float targetSpeed = Mathf.Clamp((footMotion - minimumFootSpeed) * movementGain, 0f, maximumSpeed);

        currentSpeed = Mathf.Lerp(currentSpeed, targetSpeed, 1f - Mathf.Exp(-smoothing * deltaTime));

        Vector3 forward = GetForwardDirection();
        targetRoot.position += forward * currentSpeed * deltaTime;

        // AutoGroundRootFromAvatar owns the vertical channel when present. Resetting
        // Y here on every frame used to erase its accumulated contact correction,
        // forcing the ground component to snap the complete offset again in the same
        // frame whenever contact changed from feet to the body.
        if (lockVerticalPosition && groundController == null)
        {
            Vector3 position = targetRoot.position;
            position.y = initialY;
            targetRoot.position = position;
        }

        previousLeftFootLocal = leftFootLocal;
        previousRightFootLocal = rightFootLocal;
    }

    private void CacheBones()
    {
        if (avatarAnimator == null || !avatarAnimator.isHuman)
        {
            return;
        }

        leftFoot = avatarAnimator.GetBoneTransform(HumanBodyBones.LeftFoot);
        rightFoot = avatarAnimator.GetBoneTransform(HumanBodyBones.RightFoot);
    }

    private Vector3 GetForwardDirection()
    {
        Transform reference = useDirectionReference && directionReference != null ? directionReference : targetRoot;
        Vector3 forward = Vector3.ProjectOnPlane(reference.forward, Vector3.up);

        if (forward.sqrMagnitude < 0.001f)
        {
            return Vector3.forward;
        }

        return forward.normalized;
    }
}
