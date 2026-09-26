using EVMC4U;
using UnityEngine;

[DefaultExecutionOrder(10003)]
public sealed class HipsRootHeightFromVmc : MonoBehaviour, IExternalReceiver
{
    [SerializeField] private Transform targetRoot;
    [SerializeField] private bool applyY = true;
    [SerializeField] private bool applyXZ = false;
    [SerializeField] private float heightScale = 1f;
    [SerializeField] private float smoothing = 16f;
    [SerializeField] private float maxCorrectionPerSecond = 3f;
    [SerializeField] private bool calibrateOnFirstHipsPacket = true;
    [SerializeField] private KeyCode recalibrateKey = KeyCode.H;

    private bool hasHipsPacket;
    private bool hasCalibration;
    private Vector3 latestHipsPosition;
    private Vector3 calibrationHipsPosition;
    private Vector3 calibrationRootPosition;

    public void MessageDaisyChain(ref uOSC.Message message, int callCount)
    {
        if (message.address != "/VMC/Ext/Bone/Pos"
            || message.values == null
            || message.values.Length < 4
            || !(message.values[0] is string boneName)
            || boneName != "Hips"
            || !(message.values[1] is float x)
            || !(message.values[2] is float y)
            || !(message.values[3] is float z))
        {
            return;
        }

        latestHipsPosition = new Vector3(x, y, z);
        hasHipsPacket = true;

        if (calibrateOnFirstHipsPacket && !hasCalibration)
        {
            Calibrate();
        }
    }

    public void UpdateDaisyChain()
    {
    }

    private void Awake()
    {
        if (targetRoot == null)
        {
            targetRoot = transform;
        }
    }

    private void LateUpdate()
    {
        if (targetRoot == null || !hasHipsPacket)
        {
            return;
        }

        if (Input.GetKeyDown(recalibrateKey))
        {
            Calibrate();
        }

        if (!hasCalibration)
        {
            return;
        }

        Vector3 hipsDelta = (latestHipsPosition - calibrationHipsPosition) * heightScale;
        Vector3 targetPosition = calibrationRootPosition;

        if (applyY)
        {
            targetPosition.y += hipsDelta.y;
        }

        if (applyXZ)
        {
            targetPosition.x += hipsDelta.x;
            targetPosition.z += hipsDelta.z;
        }

        float deltaTime = Mathf.Max(Time.deltaTime, 0.0001f);
        float maxDistance = maxCorrectionPerSecond * deltaTime;
        float lerp = 1f - Mathf.Exp(-smoothing * deltaTime);
        Vector3 nextPosition = Vector3.Lerp(targetRoot.position, targetPosition, lerp);
        targetRoot.position = Vector3.MoveTowards(targetRoot.position, nextPosition, maxDistance);
    }

    [ContextMenu("Calibrate Current Hips Height")]
    private void Calibrate()
    {
        if (targetRoot == null || !hasHipsPacket)
        {
            return;
        }

        calibrationHipsPosition = latestHipsPosition;
        calibrationRootPosition = targetRoot.position;
        hasCalibration = true;
        Debug.Log($"[Hips Root Height] Calibrated. hips={calibrationHipsPosition}, root={calibrationRootPosition}", this);
    }
}
