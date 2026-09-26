using UnityEngine;

[DefaultExecutionOrder(10000)]
public sealed class ManualRootMotionCompensator : MonoBehaviour
{
    [SerializeField] private Transform targetRoot;
    [SerializeField] private Transform referenceCamera;
    [SerializeField] private bool cameraRelative = true;
    [SerializeField] private float moveSpeed = 1.2f;
    [SerializeField] private float fastMultiplier = 2.5f;
    [SerializeField] private float rotateSpeed = 90f;
    [SerializeField] private KeyCode fastKey = KeyCode.LeftShift;
    [SerializeField] private KeyCode upKey = KeyCode.E;
    [SerializeField] private KeyCode downKey = KeyCode.Q;
    [SerializeField] private KeyCode turnLeftKey = KeyCode.Z;
    [SerializeField] private KeyCode turnRightKey = KeyCode.C;
    [SerializeField] private KeyCode resetKey = KeyCode.R;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    private void Awake()
    {
        if (targetRoot == null)
        {
            targetRoot = transform;
        }

        if (referenceCamera == null && Camera.main != null)
        {
            referenceCamera = Camera.main.transform;
        }

        initialPosition = targetRoot.position;
        initialRotation = targetRoot.rotation;
    }

    private void LateUpdate()
    {
        if (targetRoot == null)
        {
            return;
        }

        if (Input.GetKeyDown(resetKey))
        {
            targetRoot.SetPositionAndRotation(initialPosition, initialRotation);
            return;
        }

        float horizontal = Input.GetAxisRaw("Horizontal");
        float forward = Input.GetAxisRaw("Vertical");
        float vertical = 0f;

        if (Input.GetKey(upKey))
        {
            vertical += 1f;
        }

        if (Input.GetKey(downKey))
        {
            vertical -= 1f;
        }

        Vector3 move = GetMoveDirection(horizontal, forward);
        move += Vector3.up * vertical;

        if (move.sqrMagnitude > 1f)
        {
            move.Normalize();
        }

        float speed = moveSpeed * (Input.GetKey(fastKey) ? fastMultiplier : 1f);
        targetRoot.position += move * speed * Time.deltaTime;

        float turn = 0f;
        if (Input.GetKey(turnLeftKey))
        {
            turn -= 1f;
        }

        if (Input.GetKey(turnRightKey))
        {
            turn += 1f;
        }

        if (!Mathf.Approximately(turn, 0f))
        {
            targetRoot.Rotate(Vector3.up, turn * rotateSpeed * Time.deltaTime, Space.World);
        }
    }

    private Vector3 GetMoveDirection(float horizontal, float forward)
    {
        if (!cameraRelative || referenceCamera == null)
        {
            return new Vector3(horizontal, 0f, forward);
        }

        Vector3 cameraForward = Vector3.ProjectOnPlane(referenceCamera.forward, Vector3.up).normalized;
        Vector3 cameraRight = Vector3.ProjectOnPlane(referenceCamera.right, Vector3.up).normalized;

        if (cameraForward.sqrMagnitude < 0.001f)
        {
            cameraForward = Vector3.forward;
        }

        if (cameraRight.sqrMagnitude < 0.001f)
        {
            cameraRight = Vector3.right;
        }

        return cameraRight * horizontal + cameraForward * forward;
    }
}
