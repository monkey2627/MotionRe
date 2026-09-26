using Assets;
using Assets.Device.Service;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Demo : MonoBehaviour
{
    private const float CalibrationSeconds = 1f;
    private static readonly Color[] CubeColors =
    {
        new Color(0.22f, 0.62f, 1f),
        new Color(1f, 0.55f, 0.18f),
        new Color(0.3f, 0.85f, 0.45f),
        new Color(0.9f, 0.35f, 0.75f),
        new Color(0.95f, 0.8f, 0.22f)
    };

    public GameObject deviceScanResultProto;
    public GameObject deviceDataResultProto;
    public Text TextMsg;

    private Transform scanResultRoot;
    private Transform dataResultRoot;
    private DeviceService deviceService;
    private UdpServer udpServer;
    private readonly List<GameObject> deviceModels = new List<GameObject>();
    private readonly HashSet<string> displayedDeviceIds = new HashSet<string>();
    private readonly Dictionary<string, ImuCubeBinding> cubeBindings = new Dictionary<string, ImuCubeBinding>();
    private readonly Queue<DeviceModel> pendingFoundDevices = new Queue<DeviceModel>();
    private readonly Queue<string> pendingMessages = new Queue<string>();
    private readonly object deviceQueueLock = new object();
    private readonly object messageQueueLock = new object();
    private Button calibrateButton;
    private Text calibrateButtonText;
    private bool isCalibrating;

    void Start()
    {
        udpServer = WitApplication.Context.GetBean<UdpServer>();
        udpServer.msgEvent.AddListener(OnMsg);

        deviceService = WitApplication.Context.GetBean<DeviceService>();
        deviceService.putDeviceEvent.AddListener(OnFindDevice);

        scanResultRoot = deviceScanResultProto.transform.parent;
        deviceScanResultProto.transform.SetParent(null);
        dataResultRoot = deviceDataResultProto.transform.parent;
        deviceDataResultProto.transform.SetParent(null);

        EnsureCalibrationButton();
        Debug.Log("[WitDemo] Demo initialized. Click Start UDP, then Send Loc. Expected device port is 9250, local listen port is 1399.");
    }

    void Update()
    {
        if (deviceService == null)
        {
            return;
        }

        FlushMessages();
        FlushFoundDevices();
        SyncKnownDevices();
        UpdateDeviceDataRows();
        UpdateImuCubes();
    }

    private void FlushMessages()
    {
        string msg = null;
        lock (messageQueueLock)
        {
            while (pendingMessages.Count > 0)
            {
                msg = pendingMessages.Dequeue();
            }
        }

        if (msg != null && TextMsg != null)
        {
            TextMsg.text = msg;
        }
    }

    private void FlushFoundDevices()
    {
        while (true)
        {
            DeviceModel deviceModel = null;
            lock (deviceQueueLock)
            {
                if (pendingFoundDevices.Count > 0)
                {
                    deviceModel = pendingFoundDevices.Dequeue();
                }
            }

            if (deviceModel == null)
            {
                return;
            }

            AddDeviceRows(deviceModel);
        }
    }

    private void AddDeviceRows(DeviceModel deviceModel)
    {
        if (deviceModel == null)
        {
            return;
        }

        if (displayedDeviceIds.Contains(deviceModel.DeivceId))
        {
            EnsureImuCube(deviceModel);
            return;
        }

        displayedDeviceIds.Add(deviceModel.DeivceId);
        Debug.Log($"[WitDemo] Device discovered: {deviceModel.DeivceId}");

        GameObject scanRow = Instantiate(deviceScanResultProto, scanResultRoot);
        scanRow.name = deviceModel.DeivceId;
        scanRow.transform.GetChild(0).GetComponent<Text>().text = deviceModel.DeivceId;

        GameObject dataRow = Instantiate(deviceDataResultProto, dataResultRoot);
        dataRow.name = deviceModel.DeivceId;
        dataRow.transform.GetChild(0).GetComponent<Text>().text = deviceModel.DeivceId;
        dataRow.transform.GetChild(1).GetComponent<Text>().text = GetDeviceData(deviceModel);
        deviceModels.Add(dataRow);

        EnsureImuCube(deviceModel);
    }

    private void SyncKnownDevices()
    {
        if (deviceService == null)
        {
            return;
        }

        List<DeviceModel> devices = deviceService.GetDeviceList();
        for (int i = 0; i < devices.Count; i++)
        {
            AddDeviceRows(devices[i]);
        }
    }

    private void UpdateDeviceDataRows()
    {
        for (int i = 0; i < deviceModels.Count; i++)
        {
            GameObject dataRow = deviceModels[i];
            dataRow.transform.GetChild(1).GetComponent<Text>().text = GetDeviceData(deviceService.GetDevice(dataRow.name));
        }
    }

    private void UpdateImuCubes()
    {
        foreach (ImuCubeBinding binding in cubeBindings.Values)
        {
            DeviceModel deviceModel = deviceService.GetDevice(binding.DeviceId);
            if (deviceModel == null)
            {
                continue;
            }

            Quaternion sensorRotation = GetSensorRotation(deviceModel);
            binding.Cube.transform.rotation = Quaternion.Inverse(binding.CalibrationRotation) * sensorRotation;
        }
    }

    private void EnsureImuCube(DeviceModel deviceModel)
    {
        if (deviceModel == null || cubeBindings.ContainsKey(deviceModel.DeivceId))
        {
            return;
        }

        int index = cubeBindings.Count;
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = $"IMU_Cube_{deviceModel.DeivceId}";
        cube.transform.position = GetCubePosition(index);
        cube.transform.localScale = new Vector3(1.3f, 0.45f, 0.8f);

        Renderer renderer = cube.GetComponent<Renderer>();
        if (renderer != null)
        {
            Shader shader = Shader.Find("Standard");
            if (shader != null)
            {
                renderer.material = new Material(shader);
            }

            renderer.material.color = CubeColors[index % CubeColors.Length];
        }

        GameObject label = new GameObject($"IMU_Label_{deviceModel.DeivceId}");
        label.transform.SetParent(cube.transform);
        label.transform.localPosition = new Vector3(0f, 0.55f, 0f);
        TextMesh textMesh = label.AddComponent<TextMesh>();
        textMesh.text = deviceModel.DeivceId;
        textMesh.anchor = TextAnchor.MiddleCenter;
        textMesh.alignment = TextAlignment.Center;
        textMesh.characterSize = 0.18f;
        textMesh.fontSize = 48;
        textMesh.color = Color.black;

        cubeBindings[deviceModel.DeivceId] = new ImuCubeBinding
        {
            DeviceId = deviceModel.DeivceId,
            Cube = cube,
            CalibrationRotation = Quaternion.identity
        };

        Debug.Log($"[WitDemo] Created cube for {deviceModel.DeivceId} at {cube.transform.position}.");
    }

    private Vector3 GetCubePosition(int index)
    {
        int column = index % 4;
        int row = index / 4;
        float x = -4.5f + column * 3f;
        float y = 2.7f - row * 1.8f;
        return new Vector3(x, y, 0f);
    }

    private Quaternion GetSensorRotation(DeviceModel deviceModel)
    {
        float x = (float)deviceModel.AngleX;
        float y = (float)deviceModel.AngleY;
        float z = (float)deviceModel.AngleZ;
        return Quaternion.Euler(x, y, z);
    }

    private void EnsureCalibrationButton()
    {
        if (calibrateButton != null)
        {
            return;
        }

        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null)
        {
            Debug.LogWarning("[WitDemo] Cannot create calibration button because no Canvas was found.");
            return;
        }

        GameObject buttonObject = new GameObject("BtnCalibrateRotation");
        buttonObject.transform.SetParent(canvas.transform, false);

        RectTransform rectTransform = buttonObject.AddComponent<RectTransform>();
        rectTransform.anchorMin = new Vector2(0.5f, 0.5f);
        rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
        rectTransform.anchoredPosition = new Vector2(190f, 169f);
        rectTransform.sizeDelta = new Vector2(190f, 30f);

        Image image = buttonObject.AddComponent<Image>();
        image.color = Color.white;

        calibrateButton = buttonObject.AddComponent<Button>();
        calibrateButton.targetGraphic = image;
        calibrateButton.onClick.AddListener(CalibrateInitialRotation);

        GameObject textObject = new GameObject("Text");
        textObject.transform.SetParent(buttonObject.transform, false);

        RectTransform textRect = textObject.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;

        calibrateButtonText = textObject.AddComponent<Text>();
        calibrateButtonText.font = TextMsg != null && TextMsg.font != null
            ? TextMsg.font
            : Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        calibrateButtonText.text = "Calibrate Rotation";
        calibrateButtonText.alignment = TextAnchor.MiddleCenter;
        calibrateButtonText.color = new Color(0.35f, 0.35f, 0.35f);
        calibrateButtonText.raycastTarget = false;
    }

    public void CalibrateInitialRotation()
    {
        if (isCalibrating)
        {
            return;
        }

        StartCoroutine(CalibrateInitialRotationCoroutine());
    }

    private IEnumerator CalibrateInitialRotationCoroutine()
    {
        if (cubeBindings.Count == 0)
        {
            Debug.Log("[WitDemo] Calibration skipped: no IMU cubes have been created yet.");
            SetCalibrationButtonText("No IMU Found");
            yield return new WaitForSeconds(0.8f);
            SetCalibrationButtonText("Calibrate Rotation");
            yield break;
        }

        isCalibrating = true;
        if (calibrateButton != null)
        {
            calibrateButton.interactable = false;
        }

        Dictionary<string, RotationSampleAccumulator> angleSamples = new Dictionary<string, RotationSampleAccumulator>();
        float startedAt = Time.time;
        SetCalibrationButtonText("Calibrating...");

        while (Time.time - startedAt < CalibrationSeconds)
        {
            foreach (ImuCubeBinding binding in cubeBindings.Values)
            {
                DeviceModel deviceModel = deviceService.GetDevice(binding.DeviceId);
                if (deviceModel == null)
                {
                    continue;
                }

                Vector3 angle = new Vector3((float)deviceModel.AngleX, (float)deviceModel.AngleY, (float)deviceModel.AngleZ);
                if (!angleSamples.ContainsKey(binding.DeviceId))
                {
                    angleSamples[binding.DeviceId] = new RotationSampleAccumulator();
                }

                angleSamples[binding.DeviceId].Add(angle);
            }

            yield return null;
        }

        foreach (ImuCubeBinding binding in cubeBindings.Values)
        {
            if (!angleSamples.ContainsKey(binding.DeviceId) || angleSamples[binding.DeviceId].Count == 0)
            {
                DeviceModel deviceModel = deviceService.GetDevice(binding.DeviceId);
                binding.CalibrationRotation = deviceModel == null ? Quaternion.identity : GetSensorRotation(deviceModel);
                continue;
            }

            Vector3 averageAngle = angleSamples[binding.DeviceId].GetAverage();
            binding.CalibrationRotation = Quaternion.Euler(averageAngle.x, averageAngle.y, averageAngle.z);
            binding.Cube.transform.rotation = Quaternion.identity;
            Debug.Log($"[WitDemo] Calibrated {binding.DeviceId}. baseline={averageAngle}, samples={angleSamples[binding.DeviceId].Count}.");
        }

        isCalibrating = false;
        if (calibrateButton != null)
        {
            calibrateButton.interactable = true;
        }

        SetCalibrationButtonText("Calibrated");
        yield return new WaitForSeconds(0.8f);
        SetCalibrationButtonText("Calibrate Rotation");
    }

    private void SetCalibrationButtonText(string text)
    {
        if (calibrateButtonText != null)
        {
            calibrateButtonText.text = text;
        }
    }

    private string GetDeviceData(DeviceModel deviceModel)
    {
        if (deviceModel == null)
        {
            return "No data";
        }

        string acc = $"AccX:{deviceModel.AccX}g\t\tAccY:{deviceModel.AccY}g\t\tAccZ:{deviceModel.AccZ}g\r\n";
        string angularSpeed = $"AsX:{deviceModel.AsX}deg/s\t\tAsY:{deviceModel.AsY}deg/s\t\tAsZ:{deviceModel.AsZ}deg/s\r\n";
        string angle = $"AngleX:{deviceModel.AngleX}deg\t\tAngleY:{deviceModel.AngleY}deg\t\tAngleZ:{deviceModel.AngleZ}deg\r\n";
        string mag = $"HX:{deviceModel.HX}ut\t\tHY:{deviceModel.HY}ut\t\tHZ:{deviceModel.HZ}ut\r\n";
        string electricity = $"Electricity:{deviceModel.Electricity}%";
        return acc + angularSpeed + angle + mag + electricity;
    }

    public void StartUDP()
    {
        try
        {
            Debug.Log("[WitDemo] Start UDP clicked.");
            udpServer.StartReceive();
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
        }
    }

    public void SendLoc()
    {
        try
        {
            Debug.Log("[WitDemo] Send Loc clicked.");
            udpServer.SendLoc();
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
        }
    }

    public void StopUDP()
    {
        try
        {
            Debug.Log("[WitDemo] Stop UDP clicked.");
            udpServer.StopReceive();
        }
        catch (Exception ex)
        {
            Debug.LogException(ex);
        }
    }

    private void OnFindDevice(DeviceModel deviceModel)
    {
        lock (deviceQueueLock)
        {
            pendingFoundDevices.Enqueue(deviceModel);
        }
    }

    private void OnMsg(string msg)
    {
        lock (messageQueueLock)
        {
            pendingMessages.Enqueue(msg);
        }
    }

    private class ImuCubeBinding
    {
        public string DeviceId;
        public GameObject Cube;
        public Quaternion CalibrationRotation;
    }

    private class RotationSampleAccumulator
    {
        private Vector3 sinSum;
        private Vector3 cosSum;

        public int Count { get; private set; }

        public void Add(Vector3 degrees)
        {
            AddAxis(degrees.x, ref sinSum.x, ref cosSum.x);
            AddAxis(degrees.y, ref sinSum.y, ref cosSum.y);
            AddAxis(degrees.z, ref sinSum.z, ref cosSum.z);
            Count++;
        }

        public Vector3 GetAverage()
        {
            if (Count == 0)
            {
                return Vector3.zero;
            }

            return new Vector3(
                GetAverageAxis(sinSum.x, cosSum.x),
                GetAverageAxis(sinSum.y, cosSum.y),
                GetAverageAxis(sinSum.z, cosSum.z));
        }

        private void AddAxis(float degrees, ref float sin, ref float cos)
        {
            float radians = degrees * Mathf.Deg2Rad;
            sin += Mathf.Sin(radians);
            cos += Mathf.Cos(radians);
        }

        private float GetAverageAxis(float sin, float cos)
        {
            return Mathf.Atan2(sin, cos) * Mathf.Rad2Deg;
        }
    }
}
