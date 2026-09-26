using System;
using System.Collections.Generic;
using System.Globalization;
using SpineFlow.RawImu;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Displays one independent UI block per physical IMU tracker. SlimeVR data is
/// read through SolarXR directly and never reconstructed from VMC skeleton poses.
/// </summary>
public sealed class RawImuTrackerPreview : MonoBehaviour
{
    [Header("Manual UI bindings")]
    [Tooltip("Assign one legacy UI Text per physical IMU tracker, in display order.")]
    public List<Text> sensorTexts = new List<Text>();

    [Header("Preview")]
    [SerializeField, Range(1f, 30f)] private float refreshRate = 10f;
    [SerializeField] private bool startUdpAutomatically = true;

    private readonly List<RawImuSensorSample> _samples = new List<RawImuSensorSample>();
    private float _nextRefreshTime;
    private string _lastError;
    private bool _playbackFrameActive;
    private Dropdown _compactTrackerDropdown;
    private Text _compactDetailsText;
    private readonly List<string> _compactSensorIds = new List<string>();
    private readonly List<string> _compactOptionLabels = new List<string>();
    private string _selectedSensorId;

    private void Start()
    {
        if (startUdpAutomatically)
        {
            RawImuDataSource.TryStart(out _);
        }
        RefreshNow();
    }

    private void Update()
    {
        if (_playbackFrameActive) return;
        if (Time.unscaledTime < _nextRefreshTime) return;
        _nextRefreshTime = Time.unscaledTime + 1f / Mathf.Max(1f, refreshRate);
        RefreshNow();
    }

    public void RefreshNow()
    {
        if (_playbackFrameActive) return;
        if (!RawImuDataSource.TryCopyLatestSamples(_samples, out string error))
        {
            _lastError = error;
            SetEmptySlots(string.IsNullOrWhiteSpace(error) ? "Waiting for raw IMU data..." : error);
            RenderCompactView(_samples, error);
            return;
        }

        _lastError = null;
        RenderSamples(_samples);
    }

    public void ShowPlaybackFrame(RawImuFrame frame)
    {
        _playbackFrameActive = true;
        _lastError = null;
        RenderSamples(frame?.sensors ?? Array.Empty<RawImuSensorSample>());
    }

    public void ReturnToLive()
    {
        _playbackFrameActive = false;
        _nextRefreshTime = 0f;
        RefreshNow();
    }

    public bool IsShowingPlaybackFrame()
    {
        return _playbackFrameActive;
    }

    private void RenderSamples(IList<RawImuSensorSample> samples)
    {
        for (int index = 0; index < sensorTexts.Count; index++)
        {
            Text text = sensorTexts[index];
            if (text == null) continue;

            if (samples == null || index >= samples.Count)
            {
                text.text = $"IMU {index + 1}: waiting";
                continue;
            }

            text.text = FormatSample(samples[index]);
        }

        RenderCompactView(samples, null);
    }

    /// <summary>
    /// Connects the compact single-tracker monitor created by the commercial UI.
    /// The dropdown stores sensor IDs rather than indexes so the selection remains
    /// stable when trackers connect, disconnect, or change sort order.
    /// </summary>
    public void BindCompactView(Dropdown trackerDropdown, Text detailsText)
    {
        if (_compactTrackerDropdown != null)
            _compactTrackerDropdown.onValueChanged.RemoveListener(OnCompactTrackerChanged);

        _compactTrackerDropdown = trackerDropdown;
        _compactDetailsText = detailsText;
        _compactSensorIds.Clear();
        _compactOptionLabels.Clear();
        _selectedSensorId = null;

        if (_compactTrackerDropdown != null)
            _compactTrackerDropdown.onValueChanged.AddListener(OnCompactTrackerChanged);

        RenderCompactView(_samples, _lastError);
    }

    private void OnDestroy()
    {
        if (_compactTrackerDropdown != null)
            _compactTrackerDropdown.onValueChanged.RemoveListener(OnCompactTrackerChanged);
    }

    private void OnCompactTrackerChanged(int index)
    {
        if (index >= 0 && index < _compactSensorIds.Count)
            _selectedSensorId = _compactSensorIds[index];
        RenderCompactDetails(_samples, _lastError);
    }

    private void RenderCompactView(IList<RawImuSensorSample> samples, string error)
    {
        if (_compactTrackerDropdown == null && _compactDetailsText == null) return;

        var onlineSamples = new List<RawImuSensorSample>();
        if (samples != null)
        {
            foreach (RawImuSensorSample sample in samples)
                if (sample != null && sample.online && !string.IsNullOrWhiteSpace(sample.sensorId))
                    onlineSamples.Add(sample);
        }

        bool optionsChanged = onlineSamples.Count != _compactSensorIds.Count ||
                              onlineSamples.Count != _compactOptionLabels.Count;
        if (!optionsChanged)
        {
            for (int index = 0; index < onlineSamples.Count; index++)
            {
                RawImuSensorSample sample = onlineSamples[index];
                if (string.Equals(sample.sensorId, _compactSensorIds[index],
                        StringComparison.Ordinal) &&
                    string.Equals(BuildCompactOptionLabel(sample), _compactOptionLabels[index],
                        StringComparison.Ordinal)) continue;
                optionsChanged = true;
                break;
            }
        }

        if (optionsChanged) RebuildCompactOptions(onlineSamples);
        RenderCompactDetails(samples, error);
    }

    private void RebuildCompactOptions(IList<RawImuSensorSample> onlineSamples)
    {
        _compactSensorIds.Clear();
        _compactOptionLabels.Clear();
        var options = new List<Dropdown.OptionData>();
        if (onlineSamples != null)
        {
            foreach (RawImuSensorSample sample in onlineSamples)
            {
                _compactSensorIds.Add(sample.sensorId);
                string label = BuildCompactOptionLabel(sample);
                _compactOptionLabels.Add(label);
                options.Add(new Dropdown.OptionData(label));
            }
        }

        int selectedIndex = _compactSensorIds.FindIndex(id =>
            string.Equals(id, _selectedSensorId, StringComparison.Ordinal));
        if (_compactSensorIds.Count == 0)
        {
            _selectedSensorId = null;
            options.Add(new Dropdown.OptionData("等待 Tracker 连接..."));
            selectedIndex = 0;
        }
        else
        {
            if (selectedIndex < 0) selectedIndex = 0;
            _selectedSensorId = _compactSensorIds[selectedIndex];
        }

        if (_compactTrackerDropdown == null) return;
        _compactTrackerDropdown.ClearOptions();
        _compactTrackerDropdown.AddOptions(options);
        _compactTrackerDropdown.SetValueWithoutNotify(selectedIndex);
        _compactTrackerDropdown.interactable = _compactSensorIds.Count > 0;
        _compactTrackerDropdown.RefreshShownValue();
    }

    private void RenderCompactDetails(IList<RawImuSensorSample> samples, string error)
    {
        if (_compactDetailsText == null) return;

        RawImuSensorSample selected = null;
        if (samples != null && !string.IsNullOrWhiteSpace(_selectedSensorId))
        {
            foreach (RawImuSensorSample sample in samples)
            {
                if (sample == null || !sample.online ||
                    !string.Equals(sample.sensorId, _selectedSensorId, StringComparison.Ordinal)) continue;
                selected = sample;
                break;
            }
        }

        if (selected == null)
        {
            _compactDetailsText.text = "<b>等待 Tracker 连接</b>\n\n" +
                                       "连接后可在上方下拉框选择设备，\n这里将实时显示对应物理 IMU 数据。" +
                                       (string.IsNullOrWhiteSpace(error)
                                           ? string.Empty
                                           : "\n\n<color=#91A4BD>" + EscapeRichText(error) + "</color>");
            return;
        }

        string accelerationLabel = string.Equals(selected.accelerationKind, "raw",
            StringComparison.OrdinalIgnoreCase) ? "原始加速度" : "线性加速度";
        string imuType = string.IsNullOrWhiteSpace(selected.imuType) ? "IMU" : selected.imuType.Trim();
        string battery = selected.hasBattery
            ? Mathf.Clamp(selected.batteryPercent, 0, 100) + "%"
            : "N/A";
        string tps = selected.ticksPerSecond > 0
            ? selected.ticksPerSecond.ToString(CultureInfo.InvariantCulture) + " TPS"
            : "N/A";

        _compactDetailsText.text =
            $"<b>{EscapeRichText(GetTrackerDisplayName(selected))}</b>  <color=#35D49A>● 在线</color>\n" +
            $"类型  {EscapeRichText(imuType)}    电量  {battery}    频率  {tps}\n" +
            $"{accelerationLabel} (g)  {FormatCompactVector(selected.accelerationG, selected.hasAcceleration)}\n" +
            $"角速度 (°/s)     {FormatCompactVector(selected.angularVelocityDegPerSec, selected.hasAngularVelocity)}\n" +
            $"磁场 (µT)         {FormatCompactVector(selected.magneticFieldMicroTesla, selected.hasMagneticField)}\n" +
            $"融合姿态角 (°)   {FormatCompactVector(selected.eulerAnglesDeg, selected.hasEulerAngles)}";
    }

    private static string FormatCompactVector(RawImuVector3 value, bool available)
    {
        if (!available || value == null) return "N/A";
        return string.Format(CultureInfo.InvariantCulture,
            "X {0,7:0.00}  Y {1,7:0.00}  Z {2,7:0.00}", value.x, value.y, value.z);
    }

    private static string EscapeRichText(string value)
    {
        return (value ?? string.Empty).Replace("&", "&amp;").Replace("<", "&lt;")
            .Replace(">", "&gt;");
    }

    public int GetOnlineSensorCount()
    {
        int count = 0;
        foreach (RawImuSensorSample sample in _samples)
            if (sample != null && sample.online) count++;
        return count;
    }

    public string GetLastError()
    {
        return _lastError;
    }

    private void SetEmptySlots(string message)
    {
        for (int index = 0; index < sensorTexts.Count; index++)
        {
            if (sensorTexts[index] != null)
                sensorTexts[index].text = index == 0 ? message : $"IMU {index + 1}: waiting";
        }
    }

    private static string FormatSample(RawImuSensorSample sample)
    {
        string state = sample.online ? "ONLINE" : "OFFLINE";
        return string.Format(
            "{0} [{1}] {2}\n{3}\n{4} Acc(g) {5}\nGyro {6} | Mag {7}\nEuler {8} | Battery {9} | {10} TPS",
            GetTrackerDisplayName(sample),
            state,
            string.IsNullOrWhiteSpace(sample.imuType) ? "IMU" : sample.imuType,
            string.IsNullOrWhiteSpace(sample.source) ? "Direct sensor" : sample.source,
            string.Equals(sample.accelerationKind, "raw", StringComparison.OrdinalIgnoreCase)
                ? "Raw"
                : "Linear",
            FormatOptionalVector(sample.accelerationG, sample.hasAcceleration),
            FormatOptionalVector(sample.angularVelocityDegPerSec, sample.hasAngularVelocity),
            FormatOptionalVector(sample.magneticFieldMicroTesla, sample.hasMagneticField),
            FormatOptionalVector(sample.eulerAnglesDeg, sample.hasEulerAngles),
            sample.hasBattery ? Mathf.Clamp(sample.batteryPercent, 0, 100) + "%" : "N/A",
            sample.ticksPerSecond);
    }

    private static string BuildCompactOptionLabel(RawImuSensorSample sample)
    {
        string displayName = GetTrackerDisplayName(sample);
        string type = string.IsNullOrWhiteSpace(sample?.imuType) ? null : sample.imuType.Trim();
        return string.IsNullOrWhiteSpace(type) ? displayName : displayName + "  ·  " + type;
    }

    private static string GetTrackerDisplayName(RawImuSensorSample sample)
    {
        string roleName = GetTrackerRoleName(sample?.trackerRole);
        if (!string.IsNullOrWhiteSpace(roleName)) return roleName;
        return string.IsNullOrWhiteSpace(sample?.sensorId) ? "未分配 Tracker" : sample.sensorId.Trim();
    }

    private static string GetTrackerRoleName(string trackerRole)
    {
        switch ((trackerRole ?? string.Empty).Trim().ToUpperInvariant())
        {
            case "HEAD": return "头部";
            case "NECK": return "颈部";
            case "UPPER_CHEST": return "上胸部";
            case "CHEST": return "胸部";
            case "WAIST": return "腰部";
            case "HIP": return "髋部";
            case "LEFT_HIP": return "左髋";
            case "RIGHT_HIP": return "右髋";
            case "LEFT_SHOULDER": return "左肩";
            case "RIGHT_SHOULDER": return "右肩";
            case "LEFT_UPPER_ARM": return "左上臂";
            case "RIGHT_UPPER_ARM": return "右上臂";
            case "LEFT_LOWER_ARM": return "左前臂";
            case "RIGHT_LOWER_ARM": return "右前臂";
            case "LEFT_HAND": return "左手";
            case "RIGHT_HAND": return "右手";
            case "LEFT_UPPER_LEG": return "左大腿";
            case "RIGHT_UPPER_LEG": return "右大腿";
            case "LEFT_LOWER_LEG": return "左小腿";
            case "RIGHT_LOWER_LEG": return "右小腿";
            case "LEFT_FOOT": return "左脚";
            case "RIGHT_FOOT": return "右脚";
            case "LEFT_THUMB_METACARPAL": return "左拇指掌骨";
            case "LEFT_THUMB_PROXIMAL": return "左拇指近节";
            case "LEFT_THUMB_DISTAL": return "左拇指远节";
            case "LEFT_INDEX_PROXIMAL": return "左食指近节";
            case "LEFT_INDEX_INTERMEDIATE": return "左食指中节";
            case "LEFT_INDEX_DISTAL": return "左食指远节";
            case "LEFT_MIDDLE_PROXIMAL": return "左中指近节";
            case "LEFT_MIDDLE_INTERMEDIATE": return "左中指中节";
            case "LEFT_MIDDLE_DISTAL": return "左中指远节";
            case "LEFT_RING_PROXIMAL": return "左无名指近节";
            case "LEFT_RING_INTERMEDIATE": return "左无名指中节";
            case "LEFT_RING_DISTAL": return "左无名指远节";
            case "LEFT_LITTLE_PROXIMAL": return "左小指近节";
            case "LEFT_LITTLE_INTERMEDIATE": return "左小指中节";
            case "LEFT_LITTLE_DISTAL": return "左小指远节";
            case "RIGHT_THUMB_METACARPAL": return "右拇指掌骨";
            case "RIGHT_THUMB_PROXIMAL": return "右拇指近节";
            case "RIGHT_THUMB_DISTAL": return "右拇指远节";
            case "RIGHT_INDEX_PROXIMAL": return "右食指近节";
            case "RIGHT_INDEX_INTERMEDIATE": return "右食指中节";
            case "RIGHT_INDEX_DISTAL": return "右食指远节";
            case "RIGHT_MIDDLE_PROXIMAL": return "右中指近节";
            case "RIGHT_MIDDLE_INTERMEDIATE": return "右中指中节";
            case "RIGHT_MIDDLE_DISTAL": return "右中指远节";
            case "RIGHT_RING_PROXIMAL": return "右无名指近节";
            case "RIGHT_RING_INTERMEDIATE": return "右无名指中节";
            case "RIGHT_RING_DISTAL": return "右无名指远节";
            case "RIGHT_LITTLE_PROXIMAL": return "右小指近节";
            case "RIGHT_LITTLE_INTERMEDIATE": return "右小指中节";
            case "RIGHT_LITTLE_DISTAL": return "右小指远节";
            default: return null;
        }
    }

    private static string FormatOptionalVector(RawImuVector3 value, bool available)
    {
        return available ? FormatVector(value) : "N/A";
    }

    private static string FormatVector(RawImuVector3 value)
    {
        if (value == null) return "(n/a)";
        return string.Format("({0:F2}, {1:F2}, {2:F2})", value.x, value.y, value.z);
    }
}
