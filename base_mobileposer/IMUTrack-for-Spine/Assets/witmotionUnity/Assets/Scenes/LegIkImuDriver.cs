using Assets;
using Assets.Device.Service;
using RootMotion.FinalIK;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LegIkImuDriver : MonoBehaviour
{
    private const float CalibrationSeconds = 1f;
    private const float QuaternionEpsilon = 0.0001f;

    public LegIK legIK;
    public ImuJointBinding thigh = new ImuJointBinding("Thigh");
    public ImuJointBinding calf = new ImuJointBinding("Calf/Knee");

    public bool autoStartUdp = true;
    public bool sendLocOnStart = true;
    public bool repeatSendLocUntilDevicesFound = true;
    public float sendLocRepeatSeconds = 3f;
    public bool autoAssignEmptyDeviceIds = true;
    public bool autoCalibratePoseOnImuConnected = true;

    private readonly Queue<string> pendingDeviceIds = new Queue<string>();
    private readonly HashSet<string> discoveredDeviceIds = new HashSet<string>();
    private readonly object deviceQueueLock = new object();

    private DeviceService deviceService;
    private UdpServer udpServer;
    private Coroutine sendLocCoroutine;
    private Coroutine calibratePoseCoroutine;
    private bool subscribedToDeviceEvent;
    private bool initialPoseCaptured;
    private bool poseCalibrationInProgress;
    private bool autoCalibratePoseTriggered;

    private void OnValidate()
    {
        EnsureBindingDefaults();
    }

    private void Awake()
    {
        EnsureBindingDefaults();
        EnsureReferences();
        CaptureInitialPose();
        ApplyTargetsToLegIK();
    }

    private void OnEnable()
    {
        ResolveWitServices();
        SubscribeToDeviceEvent();
    }

    private void Start()
    {
        if (autoStartUdp)
        {
            StartUdp();
        }

        if (sendLocOnStart)
        {
            SendLoc();
        }

        if (repeatSendLocUntilDevicesFound && sendLocRepeatSeconds > 0f)
        {
            sendLocCoroutine = StartCoroutine(SendLocUntilDevicesFoundCoroutine());
        }
    }

    private void Update()
    {
        ResolveWitServices();
        SubscribeToDeviceEvent();
        FlushDiscoveredDevices();
        TryAutoCalibratePose();
        UpdateTargetsFromImu();
    }

    private void OnDisable()
    {
        if (deviceService != null && subscribedToDeviceEvent)
        {
            deviceService.putDeviceEvent.RemoveListener(OnFindDevice);
            subscribedToDeviceEvent = false;
        }

        if (sendLocCoroutine != null)
        {
            StopCoroutine(sendLocCoroutine);
            sendLocCoroutine = null;
        }

        if (calibratePoseCoroutine != null)
        {
            StopCoroutine(calibratePoseCoroutine);
            calibratePoseCoroutine = null;
        }

        poseCalibrationInProgress = false;
        autoCalibratePoseTriggered = false;
    }

    private void OnDestroy()
    {
        StopUdp();
    }

    [ContextMenu("Start UDP")]
    public void StartUdp()
    {
        ResolveWitServices();
        if (udpServer == null)
        {
            Debug.LogWarning("[LegIkImu] Cannot start UDP because UdpServer is unavailable.");
            return;
        }

        udpServer.StartReceive();
    }

    [ContextMenu("Send Loc")]
    public void SendLoc()
    {
        ResolveWitServices();
        if (udpServer == null)
        {
            Debug.LogWarning("[LegIkImu] Cannot send loc because UdpServer is unavailable.");
            return;
        }

        udpServer.SendLoc();
    }

    [ContextMenu("Stop UDP")]
    public void StopUdp()
    {
        if (udpServer != null)
        {
            udpServer.StopReceive();
        }
    }

    [ContextMenu("Calibrate Pose")]
    public void CalibratePose()
    {
        if (!Application.isPlaying)
        {
            EnsureReferences();
            CaptureInitialPose();
            ApplyTargetsToLegIK();
            Debug.Log("[LegIkImu] Captured edit-mode reference pose. Enter Play Mode to run Calibrate Pose.");
            return;
        }

        if (calibratePoseCoroutine != null)
        {
            StopCoroutine(calibratePoseCoroutine);
            calibratePoseCoroutine = null;
            poseCalibrationInProgress = false;
        }

        calibratePoseCoroutine = StartCoroutine(CalibratePoseCoroutine());
    }

    public bool HasBothDeviceIds()
    {
        return !string.IsNullOrEmpty(thigh.deviceId) && !string.IsNullOrEmpty(calf.deviceId);
    }

    private void TryAutoCalibratePose()
    {
        if (!autoCalibratePoseOnImuConnected || autoCalibratePoseTriggered || poseCalibrationInProgress)
        {
            return;
        }

        if (!HasBothDeviceIds() || !HasFreshDeviceData(thigh) || !HasFreshDeviceData(calf))
        {
            return;
        }

        autoCalibratePoseTriggered = true;
        Debug.Log("[LegIkImu] IMUs connected. Auto Calibrate Pose will start.");
        CalibratePose();
    }

    private bool HasFreshDeviceData(ImuJointBinding binding)
    {
        DeviceModel deviceModel = GetDevice(binding != null ? binding.deviceId : null);
        return deviceModel != null && deviceModel.IsOnline;
    }

    private void ResolveWitServices()
    {
        if (WitApplication.Context == null)
        {
            return;
        }

        if (udpServer == null)
        {
            udpServer = WitApplication.Context.GetBean<UdpServer>();
        }

        if (deviceService == null)
        {
            deviceService = WitApplication.Context.GetBean<DeviceService>();
        }
    }

    private void SubscribeToDeviceEvent()
    {
        if (deviceService != null && !subscribedToDeviceEvent)
        {
            deviceService.putDeviceEvent.AddListener(OnFindDevice);
            subscribedToDeviceEvent = true;
        }
    }

    private void EnsureReferences()
    {
        EnsureBindingDefaults();

        if (legIK == null)
        {
            legIK = GetComponent<LegIK>();
        }

        if (legIK == null)
        {
            legIK = FindObjectOfType<LegIK>();
        }

        if (legIK != null)
        {
            if (thigh.target == null)
            {
                thigh.target = legIK.solver.thighTarget.target;
            }

            if (calf.target == null)
            {
                calf.target = legIK.solver.calfTarget.target;
            }
        }
    }

    private void EnsureBindingDefaults()
    {
        if (thigh == null)
        {
            thigh = new ImuJointBinding("Thigh");
        }

        if (calf == null)
        {
            calf = new ImuJointBinding("Calf/Knee");
        }

        thigh.EnsureDefaults("Thigh");
        calf.EnsureDefaults("Calf/Knee");
    }

    private void ApplyTargetsToLegIK()
    {
        if (legIK == null)
        {
            return;
        }

        if (thigh.target != null)
        {
            legIK.solver.thighTarget.target = thigh.target;
            legIK.solver.thighTarget.rotationWeight = 1f;
        }

        if (calf.target != null)
        {
            legIK.solver.calfTarget.target = calf.target;
            legIK.solver.calfTarget.rotationWeight = 1f;
        }
    }

    private void CaptureInitialPose()
    {
        CaptureInitialPose(thigh, legIK != null ? legIK.solver.thigh.transform : null);
        CaptureInitialPose(calf, legIK != null ? legIK.solver.calf.transform : null);
        initialPoseCaptured = true;
    }

    private void CaptureInitialPose(ImuJointBinding binding, Transform sourceBone)
    {
        if (binding == null)
        {
            return;
        }

        Transform source = sourceBone != null ? sourceBone : binding.target;
        if (source == null)
        {
            return;
        }

        binding.initialRotation = NormalizeQuaternion(source.rotation);
        if (!binding.isCalibrated)
        {
            binding.calibrationTargetRotation = binding.initialRotation;
        }

        if (binding.target != null)
        {
            binding.target.position = source.position;
            binding.target.rotation = binding.initialRotation;
        }
    }

    private void FlushDiscoveredDevices()
    {
        while (true)
        {
            string deviceId = null;
            lock (deviceQueueLock)
            {
                if (pendingDeviceIds.Count > 0)
                {
                    deviceId = pendingDeviceIds.Dequeue();
                }
            }

            if (string.IsNullOrEmpty(deviceId))
            {
                break;
            }

            RegisterDiscoveredDevice(deviceId);
        }

        if (autoAssignEmptyDeviceIds && discoveredDeviceIds.Count < 2)
        {
            RegisterExistingDevices();
        }
    }

    private void RegisterExistingDevices()
    {
        if (deviceService == null)
        {
            return;
        }

        try
        {
            List<DeviceModel> devices = deviceService.GetDeviceList();
            for (int i = 0; i < devices.Count; i++)
            {
                if (devices[i] != null)
                {
                    RegisterDiscoveredDevice(devices[i].DeivceId);
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[LegIkImu] Device list scan failed: " + ex.Message);
        }
    }

    private void RegisterDiscoveredDevice(string deviceId)
    {
        if (string.IsNullOrEmpty(deviceId) || !discoveredDeviceIds.Add(deviceId))
        {
            return;
        }

        Debug.Log("[LegIkImu] IMU discovered: " + deviceId);

        if (!autoAssignEmptyDeviceIds)
        {
            return;
        }

        if (string.IsNullOrEmpty(thigh.deviceId))
        {
            thigh.deviceId = deviceId;
            Debug.Log("[LegIkImu] Assigned " + deviceId + " to Thigh.");
            return;
        }

        if (string.IsNullOrEmpty(calf.deviceId) && thigh.deviceId != deviceId)
        {
            calf.deviceId = deviceId;
            Debug.Log("[LegIkImu] Assigned " + deviceId + " to Calf/Knee.");
        }
    }

    private void UpdateTargetsFromImu()
    {
        UpdateTargetFromImu(thigh);
        UpdateTargetFromImu(calf);
    }

    private void UpdateTargetFromImu(ImuJointBinding binding)
    {
        if (binding == null || binding.target == null)
        {
            return;
        }

        if (poseCalibrationInProgress)
        {
            binding.target.rotation = binding.calibrationTargetRotation;
            return;
        }

        if (!binding.isCalibrated)
        {
            binding.target.rotation = binding.initialRotation;
            return;
        }

        DeviceModel deviceModel = GetDevice(binding.deviceId);
        if (deviceModel == null)
        {
            return;
        }

        Vector3 sensorEuler = GetSensorEuler(deviceModel);
        Quaternion sensorRotation = GetMappedSensorRotation(binding, sensorEuler);
        binding.lastSensorEuler = sensorEuler;

        // OpenSim-style mounting offset: sensor = target * mountingOffset.
        // This is equivalent to (sensorNow * inverse(sensorAtCalibration)) * targetAtCalibration.
        Quaternion targetRotation = NormalizeQuaternion(sensorRotation * Quaternion.Inverse(binding.mountingOffset));
        binding.target.rotation = ApplyFixedAxes(binding, targetRotation);
    }

    private DeviceModel GetDevice(string deviceId)
    {
        if (deviceService == null || string.IsNullOrEmpty(deviceId))
        {
            return null;
        }

        try
        {
            return deviceService.GetDevice(deviceId);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[LegIkImu] Device lookup failed for " + deviceId + ": " + ex.Message);
            return null;
        }
    }

    private IEnumerator SendLocUntilDevicesFoundCoroutine()
    {
        while (repeatSendLocUntilDevicesFound && !HasBothDeviceIds())
        {
            yield return new WaitForSeconds(sendLocRepeatSeconds);

            if (!HasBothDeviceIds())
            {
                SendLoc();
            }
        }

        sendLocCoroutine = null;
    }

    private IEnumerator CalibratePoseCoroutine()
    {
        ResolveWitServices();
        EnsureReferences();
        if (!initialPoseCaptured)
        {
            CaptureInitialPose();
        }

        ApplyTargetsToLegIK();
        PrepareBindingForPoseCalibration(thigh);
        PrepareBindingForPoseCalibration(calf);
        poseCalibrationInProgress = true;

        Dictionary<ImuJointBinding, RotationSampleAccumulator> samples =
            new Dictionary<ImuJointBinding, RotationSampleAccumulator>();
        float startedAt = Time.time;

        Debug.Log("[LegIkImu] Calibrate Pose started. Stand still for 1 second.");

        while (Time.time - startedAt < CalibrationSeconds)
        {
            SampleBinding(thigh, samples);
            SampleBinding(calf, samples);
            yield return null;
        }

        ApplyPoseCalibration(thigh, samples);
        ApplyPoseCalibration(calf, samples);

        poseCalibrationInProgress = false;
        calibratePoseCoroutine = null;
        Debug.Log("[LegIkImu] Calibrate Pose finished.");
    }

    private void PrepareBindingForPoseCalibration(ImuJointBinding binding)
    {
        if (binding == null || binding.target == null)
        {
            return;
        }

        binding.isCalibrated = false;
        binding.calibrationTargetRotation = binding.initialRotation;
        binding.initialRotation = binding.calibrationTargetRotation;
        binding.target.rotation = binding.calibrationTargetRotation;
    }

    private void SampleBinding(
        ImuJointBinding binding,
        Dictionary<ImuJointBinding, RotationSampleAccumulator> samples)
    {
        if (binding == null)
        {
            return;
        }

        DeviceModel deviceModel = GetDevice(binding.deviceId);
        if (deviceModel == null)
        {
            return;
        }

        if (!samples.ContainsKey(binding))
        {
            samples[binding] = new RotationSampleAccumulator();
        }

        Vector3 sensorEuler = GetSensorEuler(deviceModel);
        samples[binding].Add(GetMappedSensorRotation(binding, sensorEuler), sensorEuler);
    }

    private void ApplyPoseCalibration(
        ImuJointBinding binding,
        Dictionary<ImuJointBinding, RotationSampleAccumulator> samples)
    {
        if (binding == null || binding.target == null)
        {
            return;
        }

        if (!samples.ContainsKey(binding) || samples[binding].Count == 0)
        {
            Debug.LogWarning("[LegIkImu] Calibrate Pose skipped for " + binding.label + ". No IMU samples for deviceId=" + binding.deviceId);
            return;
        }

        Quaternion sensorRotation = samples[binding].GetAverage();
        binding.calibrationSensorRotation = sensorRotation;
        binding.mountingOffset = NormalizeQuaternion(Quaternion.Inverse(binding.calibrationTargetRotation) * sensorRotation);
        binding.lastSensorEuler = samples[binding].GetAverageEuler();
        binding.isCalibrated = true;
        binding.target.rotation = binding.calibrationTargetRotation;

        Debug.Log("[LegIkImu] " + binding.label +
                  " calibrated. deviceId=" + binding.deviceId +
                  ", IMUAxis2JointAxis=" + binding.IMUAxis2JointAxis +
                  ", targetRotation=" + binding.calibrationTargetRotation.eulerAngles +
                  ", sensorRotation=" + binding.calibrationSensorRotation.eulerAngles +
                  ", mountingOffset=" + binding.mountingOffset.eulerAngles +
                  ", samples=" + samples[binding].Count);
    }

    private void OnFindDevice(DeviceModel deviceModel)
    {
        if (deviceModel == null)
        {
            return;
        }

        lock (deviceQueueLock)
        {
            pendingDeviceIds.Enqueue(deviceModel.DeivceId);
        }
    }

    private static Vector3 GetSensorEuler(DeviceModel deviceModel)
    {
        return new Vector3((float)deviceModel.AngleX, (float)deviceModel.AngleY, (float)deviceModel.AngleZ);
    }

    private static Quaternion GetMappedSensorRotation(ImuJointBinding binding, Vector3 sensorEuler)
    {
        Quaternion sensorRotation = Quaternion.Euler(sensorEuler.x, sensorEuler.y, sensorEuler.z);
        if (binding == null)
        {
            return sensorRotation;
        }

        binding.EnsureDefaults(null);
        if (!IsAxisMappingValid(binding.IMUAxis2JointAxis))
        {
            return sensorRotation;
        }

        Matrix4x4 axisMap = BuildAxisMap(binding.IMUAxis2JointAxis);
        Matrix4x4 sensorMatrix = Matrix4x4.Rotate(sensorRotation);
        Matrix4x4 mappedMatrix = axisMap * sensorMatrix * axisMap.inverse;
        return QuaternionFromMatrix(mappedMatrix);
    }

    private static Quaternion ApplyFixedAxes(ImuJointBinding binding, Quaternion targetRotation)
    {
        if (binding == null || binding.fixedJointAxes == null || !binding.fixedJointAxes.HasAnyFixedAxis)
        {
            return targetRotation;
        }

        Quaternion delta = NormalizeQuaternion(Quaternion.Inverse(binding.calibrationTargetRotation) * targetRotation);
        Vector3 deltaEuler = NormalizeEuler(delta.eulerAngles);

        if (binding.fixedJointAxes.fixX)
        {
            deltaEuler.x = 0f;
        }

        if (binding.fixedJointAxes.fixY)
        {
            deltaEuler.y = 0f;
        }

        if (binding.fixedJointAxes.fixZ)
        {
            deltaEuler.z = 0f;
        }

        Quaternion filteredDelta = Quaternion.Euler(deltaEuler.x, deltaEuler.y, deltaEuler.z);
        return NormalizeQuaternion(binding.calibrationTargetRotation * filteredDelta);
    }

    private static Vector3 NormalizeEuler(Vector3 euler)
    {
        return new Vector3(
            NormalizeEulerAngle(euler.x),
            NormalizeEulerAngle(euler.y),
            NormalizeEulerAngle(euler.z));
    }

    private static float NormalizeEulerAngle(float angle)
    {
        while (angle > 180f)
        {
            angle -= 360f;
        }

        while (angle < -180f)
        {
            angle += 360f;
        }

        return angle;
    }

    private static bool IsAxisMappingValid(ImuAxis2JointAxis axisMap)
    {
        if (axisMap == null)
        {
            return false;
        }

        int x = GetAxisIndex(axisMap.imuX);
        int y = GetAxisIndex(axisMap.imuY);
        int z = GetAxisIndex(axisMap.imuZ);
        if (x == y || x == z || y == z)
        {
            return false;
        }

        return true;
    }

    private static Matrix4x4 BuildAxisMap(ImuAxis2JointAxis axisMap)
    {
        Matrix4x4 matrix = Matrix4x4.identity;
        SetAxisMapColumn(ref matrix, 0, GetAxisVector(axisMap.imuX));
        SetAxisMapColumn(ref matrix, 1, GetAxisVector(axisMap.imuY));
        SetAxisMapColumn(ref matrix, 2, GetAxisVector(axisMap.imuZ));
        return matrix;
    }

    private static void SetAxisMapColumn(ref Matrix4x4 matrix, int column, Vector3 axis)
    {
        matrix.SetColumn(column, new Vector4(axis.x, axis.y, axis.z, column == 3 ? 1f : 0f));
    }

    private static Vector3 GetAxisVector(SignedJointAxis axis)
    {
        switch (axis)
        {
            case SignedJointAxis.MinusX:
                return -Vector3.right;
            case SignedJointAxis.PlusY:
                return Vector3.up;
            case SignedJointAxis.MinusY:
                return -Vector3.up;
            case SignedJointAxis.PlusZ:
                return Vector3.forward;
            case SignedJointAxis.MinusZ:
                return -Vector3.forward;
            case SignedJointAxis.PlusX:
            default:
                return Vector3.right;
        }
    }

    private static int GetAxisIndex(SignedJointAxis axis)
    {
        switch (axis)
        {
            case SignedJointAxis.PlusY:
            case SignedJointAxis.MinusY:
                return 1;
            case SignedJointAxis.PlusZ:
            case SignedJointAxis.MinusZ:
                return 2;
            case SignedJointAxis.PlusX:
            case SignedJointAxis.MinusX:
            default:
                return 0;
        }
    }

    private static Quaternion QuaternionFromMatrix(Matrix4x4 matrix)
    {
        Vector4 forwardColumn = matrix.GetColumn(2);
        Vector4 upColumn = matrix.GetColumn(1);
        Vector3 forward = new Vector3(forwardColumn.x, forwardColumn.y, forwardColumn.z);
        Vector3 up = new Vector3(upColumn.x, upColumn.y, upColumn.z);

        if (forward.sqrMagnitude < QuaternionEpsilon || up.sqrMagnitude < QuaternionEpsilon)
        {
            return Quaternion.identity;
        }

        return NormalizeQuaternion(Quaternion.LookRotation(forward.normalized, up.normalized));
    }

    private static Quaternion NormalizeQuaternion(Quaternion quaternion)
    {
        float magnitude = Mathf.Sqrt(
            quaternion.x * quaternion.x +
            quaternion.y * quaternion.y +
            quaternion.z * quaternion.z +
            quaternion.w * quaternion.w);

        if (magnitude < QuaternionEpsilon)
        {
            return Quaternion.identity;
        }

        return new Quaternion(
            quaternion.x / magnitude,
            quaternion.y / magnitude,
            quaternion.z / magnitude,
            quaternion.w / magnitude);
    }

    public enum SignedJointAxis
    {
        PlusX,
        MinusX,
        PlusY,
        MinusY,
        PlusZ,
        MinusZ
    }

    [Serializable]
    public class ImuAxis2JointAxis
    {
        [Tooltip("The joint/local bone axis that matches the IMU local +X axis.")]
        public SignedJointAxis imuX = SignedJointAxis.PlusX;
        [Tooltip("The joint/local bone axis that matches the IMU local +Y axis.")]
        public SignedJointAxis imuY = SignedJointAxis.PlusY;
        [Tooltip("The joint/local bone axis that matches the IMU local +Z axis.")]
        public SignedJointAxis imuZ = SignedJointAxis.PlusZ;

        public override string ToString()
        {
            return "IMU X->" + imuX + ", IMU Y->" + imuY + ", IMU Z->" + imuZ;
        }
    }

    [Serializable]
    public class FixedJointAxes
    {
        [Tooltip("Keep the calibrated joint X rotation instead of driving it from the IMU.")]
        public bool fixX;
        [Tooltip("Keep the calibrated joint Y rotation instead of driving it from the IMU.")]
        public bool fixY;
        [Tooltip("Keep the calibrated joint Z rotation instead of driving it from the IMU.")]
        public bool fixZ;

        public bool HasAnyFixedAxis
        {
            get { return fixX || fixY || fixZ; }
        }
    }

    [Serializable]
    public class ImuJointBinding
    {
        public string label;
        public string deviceId;
        public Transform target;
        public ImuAxis2JointAxis IMUAxis2JointAxis = new ImuAxis2JointAxis();
        public FixedJointAxes fixedJointAxes = new FixedJointAxes();
        public Quaternion initialRotation = Quaternion.identity;
        public Quaternion calibrationSensorRotation = Quaternion.identity;
        public Quaternion calibrationTargetRotation = Quaternion.identity;
        public Quaternion mountingOffset = Quaternion.identity;
        public Vector3 lastSensorEuler;
        public bool isCalibrated;

        public ImuJointBinding()
        {
        }

        public ImuJointBinding(string label)
        {
            this.label = label;
            EnsureDefaults(label);
        }

        public void EnsureDefaults(string fallbackLabel)
        {
            if (string.IsNullOrEmpty(label) && !string.IsNullOrEmpty(fallbackLabel))
            {
                label = fallbackLabel;
            }

            if (IMUAxis2JointAxis == null)
            {
                IMUAxis2JointAxis = new ImuAxis2JointAxis();
            }

            if (fixedJointAxes == null)
            {
                fixedJointAxes = new FixedJointAxes();
            }
        }
    }

    private class RotationSampleAccumulator
    {
        private Quaternion referenceRotation = Quaternion.identity;
        private Vector4 rotationSum;
        private Vector3 lastEuler;

        public int Count { get; private set; }

        public void Add(Quaternion rotation, Vector3 euler)
        {
            rotation = NormalizeQuaternion(rotation);

            if (Count == 0)
            {
                referenceRotation = rotation;
            }
            else if (Quaternion.Dot(referenceRotation, rotation) < 0f)
            {
                rotation = new Quaternion(-rotation.x, -rotation.y, -rotation.z, -rotation.w);
            }

            rotationSum.x += rotation.x;
            rotationSum.y += rotation.y;
            rotationSum.z += rotation.z;
            rotationSum.w += rotation.w;
            lastEuler = euler;
            Count++;
        }

        public Quaternion GetAverage()
        {
            if (Count == 0)
            {
                return Quaternion.identity;
            }

            return NormalizeQuaternion(new Quaternion(
                rotationSum.x / Count,
                rotationSum.y / Count,
                rotationSum.z / Count,
                rotationSum.w / Count));
        }

        public Vector3 GetAverageEuler()
        {
            if (Count == 0)
            {
                return lastEuler;
            }

            return GetAverage().eulerAngles;
        }
    }
}
