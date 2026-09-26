using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using EVMC4U;
using Assets.Library.WitUnitySdk.Utils;
using SpineFlow.MotionPackages;
using SpineFlow.PoseRecording;
using SpineFlow.RawImu;
using SpineFlow.Reconstruction;
using SpineFlow.RuntimeExport;
using SpineFlow.TrackerRecording;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public sealed class HistoryReconstructionSensor
{
    public string sensorId;
    public string trackerRole;
    public bool hasQuaternion;
}

[Serializable]
public sealed class HistoryReconstructionCandidate
{
    public string metadataPath;
    public string name;
    public string recordedAt;
    public float duration;
    public int rawFormatVersion;
    public HistoryReconstructionSensor[] sensors = Array.Empty<HistoryReconstructionSensor>();
}

/// <summary>
/// 录制控制面板。不修改 IMUTrack 任何已有代码，
/// 通过 PoseFbxRecorder 公开的 StartRecording/StopRecording 控制录制，
/// 同时管理可在 Editor 与 Standalone 共用的 Humanoid 姿态、Tracker 与 Raw IMU 数据。
/// </summary>
[DefaultExecutionOrder(21000)]
public class RecordingController : MonoBehaviour
{
    [Header("UI")]
    public Text txtStatus;
    public Text txtFileName;
    public Button btnCalibrate;
    public Button btnRecord;
    public Button btnPlayback;
    public Text txtRecordLabel;

    [Header("Coach Profile (optional until UI is assigned)")]
    public InputField inputCoachHeightCm;
    public InputField inputCoachWeightKg;
    public Dropdown dropdownCoachSex;

    [Header("Recorder")]
    public PoseFbxRecorder poseRecorder;

    [Header("Tracker JSON")]
    [Tooltip("Preferred source: receives tracker poses through the existing EVMC4U message chain.")]
    public VmcTrackerPoseSource trackerPoseSource;
    [Tooltip("Receives SlimeVR VMC tracker poses. Auto-detected when empty.")]
    public VmcOscDebugBridge vmcBridge;
    [Tooltip("Existing EVMC4U tracker receiver. Auto-detected and preferred when empty.")]
    public EVMC4U.DeviceReceiver trackerDeviceReceiver;
    [Tooltip("Optional path field for importing an existing .tracker.json. Empty means load the latest recording.")]
    public InputField inputTrackerJsonPath;
    [Tooltip("Optional explicit preview bindings. When empty, EVMC4U DeviceReceiver serial/transform bindings are used.")]
    public List<TrackerPreviewBinding> trackerPreviewBindings = new List<TrackerPreviewBinding>();
    [SerializeField, Range(1f, 120f)] private float trackerSampleRate = 30f;

    [Header("Raw IMU JSON")]
    [Tooltip("Capture direct per-tracker IMU data. SlimeVR SolarXR is preferred; WitMotion is retained as a fallback.")]
    [SerializeField] private bool enableRawImuCapture = true;
    [Tooltip("Connect to the direct physical tracker data source automatically when this controller starts.")]
    [SerializeField] private bool autoStartRawImuUdp = true;
    [SerializeField, Range(1f, 120f)] private float rawImuSampleRate = 30f;
    [SerializeField, Min(1)] private int minimumRawImuSensors = 2;
    [Tooltip("Optional path field for importing an existing .raw-imu.json. Empty means load the latest recording.")]
    public InputField inputRawImuJsonPath;
    [Tooltip("The manually configured ten-slot raw IMU preview panel.")]
    public RawImuTrackerPreview rawImuTrackerPreview;

    [Header("Motion Package")]
    [Tooltip("Optional. Falls back to dead_bug_demo when it is not assigned or left blank.")]
    public InputField inputActionId;
    [Tooltip("Optional. Falls back to Dead Bug Demo when it is not assigned or left blank.")]
    public InputField inputActionName;
    [SerializeField] private string defaultActionId = "dead_bug_demo";
    [SerializeField] private string defaultActionName = "Dead Bug Demo";
    [SerializeField, Min(1f)] private float operationStatusSeconds = 8f;

    [Header("Trajectory List")]
    public Transform trajectoryListContainer;
    public GameObject trajectoryItemPrefab;
    public event Action TrajectoryListRefreshed;
    public event Action TrajectoryExportSelectionChanged;
    public event Action<bool> HistoryReconstructionStateChanged;
    [SerializeField, Min(2f)] private float deleteConfirmationSeconds = 8f;

    [Header("Playback")]
    public Animator playbackAnimator;

    public bool IsRecording => _isRecording || (poseRecorder != null && poseRecorder.IsRecording);
    public bool IsHistoryReconstructing => _isHistoryReconstructing;
    public string SelectedExportMetadataPath => _selectedExportMetaPath;

    public bool TryPrepareForAvatarSwitch(out string error)
    {
        error = null;
        if (IsRecording)
        {
            error = "请先停止录制，再切换人物模型";
            return false;
        }

        if (_isPlayingBack) EndPlayback();
        return true;
    }

    public void RebindPlaybackAnimator(Animator animator)
    {
        _posePlaybackHandler?.Dispose();
        _posePlaybackHandler = null;
        _loadedPoseRecording = null;
        _currentPlaybackClip = null;
        _isPlayingBack = false;
        playbackAnimator = animator;
    }

    private bool _isRecording;
    private bool _showCalibMsg;
    private string _currentTrajectoryName;
    private string _recordingActionId;
    private string _recordingDisplayName;
    private string _metaDataPath;
    private float _recordStartTime;
    private HashSet<string> _preRecordFiles;
    private bool _isPlayingBack;
    private float _playbackTime;
    private AnimationClip _currentPlaybackClip;
    private RuntimePoseRecordingPackage _loadedPoseRecording;
    private HumanPoseHandler _posePlaybackHandler;
    private HumanPose _posePlaybackPose;
    private float _posePlaybackTime;
    private int _posePlaybackFrameIndex;
    private float _operationStatusUntil;
    private readonly List<TrackerPoseData> _trackerPoseBuffer = new List<TrackerPoseData>();
    private readonly List<TrackerRecordingFrame> _trackerFrames = new List<TrackerRecordingFrame>();
    private TrackerRecordingPackage _loadedTrackerRecording;
    private float _trackerPlaybackTime;
    private int _trackerPlaybackFrameIndex;
    private string _lastTrackerJsonPath;
    private float _nextTrackerSampleTime;
    private string _trackerDataSource;
    private readonly List<RawImuSensorSample> _rawImuSampleBuffer = new List<RawImuSensorSample>();
    private readonly List<RawImuFrame> _rawImuFrames = new List<RawImuFrame>();
    private readonly List<RawImuCalibration> _rawImuCalibrations = new List<RawImuCalibration>();
    private string _lastRawImuJsonPath;
    private float _nextRawImuSampleTime;
    private bool _rawImuUdpStarted;
    private RawImuRecordingPackage _loadedRawImuRecording;
    private bool _isRawImuPlayback;
    private bool _isRawImuPlaybackPaused;
    private float _rawImuPlaybackTime;
    private int _rawImuPlaybackFrameIndex;
    private string _lastImportedPoseRecordingPath;
    private string _lastImportedMotionSourcePath;
    private RuntimePoseRecordingPackage _pendingImportedPoseRecording;
    private string _pendingDeleteMetaPath;
    private Button _pendingDeleteButton;
    private string _pendingDeleteOriginalLabel;
    private float _pendingDeleteDeadline;
    private bool _isHistoryReconstructing;
    private string _selectedExportMetaPath;
    private readonly Dictionary<int, string> _trajectoryItemMetadataPaths =
        new Dictionary<int, string>();

    [System.Serializable]
    private sealed class TrajectoryMetadata
    {
        public string name;
        public string actionId;
        public string displayName;
        public string recordedAt;
        public float duration;
        public string poseRecordingFile;
        public string trackerJsonFile;
        public string rawImuJsonFile;
        public bool imported;
        public string importLabel;
        public string importSourcePath;
        public bool reconstructed;
        public string reconstructionSourceName;
        public string reconstructionSourceMetadata;
        public string reconstructionSourceRawImuFile;
        public int reconstructionTrackerCount;
        public string[] reconstructionSensorIds = System.Array.Empty<string>();
        public string[] reconstructionTrackerRoles = System.Array.Empty<string>();
        public string[] files = System.Array.Empty<string>();
    }

    // PoseFbxRecorder 的 outputFolder 默认值（相对于项目根目录）
    private static string OutputFolder => "Recordings";

    private void Start()
    {
        _metaDataPath = Path.Combine(Application.persistentDataPath, "TrajectoriesMeta");
        Directory.CreateDirectory(_metaDataPath);

        if (poseRecorder == null)
            poseRecorder = FindObjectOfType<PoseFbxRecorder>();
        if (vmcBridge == null)
            vmcBridge = FindObjectOfType<VmcOscDebugBridge>();
        if (trackerPoseSource == null)
            trackerPoseSource = FindObjectOfType<VmcTrackerPoseSource>();
        if (trackerDeviceReceiver == null)
            trackerDeviceReceiver = FindObjectOfType<EVMC4U.DeviceReceiver>();
        if (rawImuTrackerPreview == null)
            rawImuTrackerPreview = FindObjectOfType<RawImuTrackerPreview>();

        if (enableRawImuCapture && autoStartRawImuUdp)
            TryStartRawImuUdp(false, out _);

        RefreshTrajectoryList();
    }

    private void Update()
    {
        if (!string.IsNullOrWhiteSpace(_pendingDeleteMetaPath) &&
            Time.unscaledTime > _pendingDeleteDeadline)
        {
            ClearPendingDeleteConfirmation(true);
        }

        if (_isRecording)
        {
            CaptureTrackerFrame();
            CaptureRawImuFrame();
        }

        if (_isRawImuPlayback)
        {
            UpdateRawImuPlayback();
            return;
        }

        // Runtime humanoid pose playback is applied from LateUpdate so it wins
        // over Animator and live receiver updates made earlier in the frame.
        if (_isPlayingBack && _loadedPoseRecording != null) return;

        if (txtStatus == null) return;

        if (_isPlayingBack && _loadedTrackerRecording != null)
        {
            UpdateTrackerPlayback();
            return;
        }

        // 回放中逐帧采样
        if (_isPlayingBack && _currentPlaybackClip != null && playbackAnimator != null)
        {
            _playbackTime += Time.deltaTime;
            if (_playbackTime >= _currentPlaybackClip.length)
            {
                EndPlayback();
            }
            else
            {
                _currentPlaybackClip.SampleAnimation(playbackAnimator.gameObject, _playbackTime);
                txtStatus.text = $"回放中... {_playbackTime:F1}s / {_currentPlaybackClip.length:F1}s";
            }
            return;
        }

        if (_showCalibMsg) return;
        if (_isRecording)
            txtStatus.text = $"正在录制... {(Time.unscaledTime - _recordStartTime):F3}秒";
        else if (Time.unscaledTime >= _operationStatusUntil)
            txtStatus.text = "按「开始录制」记录动作";
    }

    private void LateUpdate()
    {
        if (_isPlayingBack && _loadedPoseRecording != null)
            UpdatePosePlayback();
    }

    // ===== 校准 =====
    public void OnCalibrate()
    {
        _showCalibMsg = true;
        bool rawImuCalibrated = TryCalibrateRawImu(out string rawImuMessage);
        var bridge = FindObjectOfType<VmcOscDebugBridge>();
        if (bridge != null)
        {
            bridge.SendResetCalibration();
            if (txtStatus != null)
                txtStatus.text = rawImuCalibrated
                    ? rawImuMessage + "; SlimeVR calibration command sent."
                    : "SlimeVR calibration command sent; " + rawImuMessage;
        }
        else
        {
            if (txtStatus != null)
                txtStatus.text = rawImuCalibrated
                    ? rawImuMessage
                    : rawImuMessage + "; calibrate SlimeVR manually.";
        }

        // Do not blend the pre-calibration pose into the first calibrated frame.
        // The stabilizer will re-seed from the next valid Humanoid pose.
        var poseStabilizer = FindObjectOfType<HumanoidPoseStabilizer>();
        poseStabilizer?.ResetStabilizer();
        Invoke(nameof(ClearCalibMsg), 3f);
    }

    private void ClearCalibMsg()
    {
        _showCalibMsg = false;
    }

    // ===== 录制 =====
    public void OnToggleRecord()
    {
        if (poseRecorder == null)
        {
            txtStatus.text = "录制器未就绪";
            return;
        }

        if (_isRecording)
            StopRecording();
        else
            StartRecording();
    }

    private void StartRecording()
    {
        if (!TryReadCoachProfile(out _, out string profileError))
        {
            ShowOperationStatus(profileError);
            return;
        }

        if (_isPlayingBack) EndPlayback();

        string sensorNotice = null;
        if (enableRawImuCapture)
            sensorNotice = GetRawImuSensorCountNotice();

        if (trackerPoseSource == null)
            trackerPoseSource = FindObjectOfType<VmcTrackerPoseSource>();
        if (trackerDeviceReceiver == null)
            trackerDeviceReceiver = FindObjectOfType<EVMC4U.DeviceReceiver>();
        if (vmcBridge == null && trackerPoseSource == null && trackerDeviceReceiver == null)
            vmcBridge = FindObjectOfType<VmcOscDebugBridge>();
        _trackerFrames.Clear();
        _lastTrackerJsonPath = null;
        _trackerDataSource = null;
        _nextTrackerSampleTime = 0f;
        _rawImuFrames.Clear();
        _lastRawImuJsonPath = null;
        _nextRawImuSampleTime = 0f;
        if (enableRawImuCapture && autoStartRawImuUdp)
            TryStartRawImuUdp(false, out _);
        _isRecording = true;
        _recordStartTime = Time.unscaledTime;
        _recordingActionId = GetMotionPackageActionId();
        _recordingDisplayName = GetMotionPackageDisplayName();
        _currentTrajectoryName = BuildRecordingTrajectoryName(
            _recordingActionId, _recordingDisplayName);

        // 快照：记录录制前目录里已有的文件
        _preRecordFiles = GetAnimFiles();

        poseRecorder.ConfigureRuntimeMetadata(
            _recordingActionId, _recordingDisplayName, _currentTrajectoryName);
        poseRecorder.StartRecording();

        txtRecordLabel.text = "停止录制";
        txtFileName.text = $"当前录制文件：{_currentTrajectoryName}";
        btnCalibrate.interactable = false;
        btnPlayback.interactable = false;
        txtStatus.text = string.IsNullOrWhiteSpace(sensorNotice)
            ? "正在录制..."
            : "正在录制... " + sensorNotice;
    }

    private void StopRecording()
    {
        poseRecorder.StopRecording();
        _isRecording = false;
        float recordedDuration = Mathf.Max(0f, Time.unscaledTime - _recordStartTime);
        bool trackerSaved = TrySaveTrackerRecording(recordedDuration, out string trackerMessage);
        bool rawImuSaved = TrySaveRawImuRecording(recordedDuration, out string rawImuMessage);
        string poseRecordingPath = poseRecorder.LastRuntimeRecordingPath;
        bool poseSaved = !string.IsNullOrWhiteSpace(poseRecordingPath) && File.Exists(poseRecordingPath);

        // 对比快照，找出新生成的 .anim 文件
        var afterFiles = GetAnimFiles();
        var newFiles = new List<string>();
        foreach (var f in afterFiles)
            if (!_preRecordFiles.Contains(f))
                newFiles.Add(f);

        if (newFiles.Count > 0)
        {
            SaveTrajectoryMeta(newFiles, poseRecordingPath, _lastTrackerJsonPath, _lastRawImuJsonPath);
            if (txtStatus != null)
                txtStatus.text = BuildRecordingResultStatus(newFiles.Count, poseSaved, trackerSaved, trackerMessage,
                    rawImuSaved, rawImuMessage);
        }
        else
        {
            SaveTrajectoryMeta(newFiles, poseRecordingPath, _lastTrackerJsonPath, _lastRawImuJsonPath);
            if (txtStatus != null)
                txtStatus.text = BuildRecordingResultStatus(0, poseSaved, trackerSaved, trackerMessage,
                    rawImuSaved, rawImuMessage);
        }

        RefreshTrajectoryList();
        txtRecordLabel.text = "开始录制";
        btnCalibrate.interactable = true;
        btnPlayback.interactable = true;
    }

    // ===== 轨迹元数据 =====
    private void SaveTrajectoryMeta(List<string> animFilePaths, string poseRecordingPath,
        string trackerJsonPath, string rawImuJsonPath)
    {
        var metadata = new TrajectoryMetadata
        {
            name = _currentTrajectoryName,
            actionId = _recordingActionId,
            displayName = _recordingDisplayName,
            recordedAt = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            duration = Mathf.Max(0f, Time.unscaledTime - _recordStartTime),
            poseRecordingFile = poseRecordingPath ?? string.Empty,
            trackerJsonFile = trackerJsonPath ?? string.Empty,
            rawImuJsonFile = rawImuJsonPath ?? string.Empty,
            imported = false,
            importLabel = string.Empty,
            importSourcePath = string.Empty,
            files = animFilePaths == null ? System.Array.Empty<string>() : animFilePaths.ToArray()
        };
        var path = Path.Combine(_metaDataPath, _currentTrajectoryName + ".json");
        File.WriteAllText(path, JsonUtility.ToJson(metadata, true));
        _selectedExportMetaPath = Path.GetFullPath(path);
    }

    // ===== 轨迹列表 =====
    private void RefreshTrajectoryList()
    {
        if (trajectoryListContainer == null || trajectoryItemPrefab == null) return;

        ClearPendingDeleteConfirmation(false);

        // Destroy is deferred until the end of the frame. Hide and detach the
        // stale items first so they cannot overlap the freshly rebuilt list or
        // be included in its synchronous layout pass.
        var staleItems = new List<Transform>();
        foreach (Transform child in trajectoryListContainer)
            staleItems.Add(child);
        foreach (Transform staleItem in staleItems)
        {
            if (staleItem == null) continue;
            staleItem.gameObject.SetActive(false);
            staleItem.SetParent(null, false);
            Destroy(staleItem.gameObject);
        }
        _trajectoryItemMetadataPaths.Clear();

        var files = new List<string>(Directory.GetFiles(_metaDataPath, "*.json"));
        files.Sort((left, right) =>
            File.GetLastWriteTimeUtc(right).CompareTo(File.GetLastWriteTimeUtc(left)));

        var validEntries = new List<KeyValuePair<string, TrajectoryMetadata>>();
        foreach (string file in files)
        {
            if (!TryLoadTrajectoryMetadata(file, out TrajectoryMetadata metadata,
                    out string metadataError))
            {
                Debug.LogWarning($"[Trajectory] Skipping invalid metadata '{file}': {metadataError}", this);
                continue;
            }
            validEntries.Add(new KeyValuePair<string, TrajectoryMetadata>(
                Path.GetFullPath(file), metadata));
        }

        bool selectedStillExists = validEntries.Exists(entry =>
            PathsEqual(entry.Key, _selectedExportMetaPath));
        if (!selectedStillExists)
            _selectedExportMetaPath = validEntries.Count > 0 ? validEntries[0].Key : null;

        var newItems = new List<GameObject>();
        foreach (KeyValuePair<string, TrajectoryMetadata> entry in validEntries)
        {
            string f = entry.Key;
            TrajectoryMetadata metadata = entry.Value;

            var item = Instantiate(trajectoryItemPrefab, trajectoryListContainer);
            item.SetActive(false);
            newItems.Add(item);
            _trajectoryItemMetadataPaths[item.GetInstanceID()] = f;
            var texts = item.GetComponentsInChildren<Text>(true);
            if (texts.Length >= 2)
            {
                texts[0].supportRichText = true;
                string importBadge = metadata.imported
                    ? $"<color=#F05264>导入</color>  "
                    : string.Empty;
                string reconstructedName = metadata.reconstructed
                    ? $"<color=#FF4FD8>{EscapeRichText(metadata.reconstructionSourceName)} · 重构 · " +
                      $"{metadata.reconstructionTrackerCount} Tracker</color>"
                    : EscapeRichText(metadata.name);
                texts[0].text = importBadge + reconstructedName;
                texts[1].text = $"{metadata.recordedAt} · {metadata.duration:F1}秒";
            }

            var btns = item.GetComponentsInChildren<Button>(true);
            // 跳过根节点上的 Button，只取子按钮
            var childBtns = new List<Button>();
            foreach (var b in btns)
                if (b.gameObject != item) childBtns.Add(b);
            
            var capturedFile = f;
            Button rootButton = item.GetComponent<Button>();
            if (rootButton != null)
            {
                rootButton.onClick.RemoveAllListeners();
                rootButton.onClick.AddListener(() => SelectTrajectoryForExport(capturedFile));
            }
            if (childBtns.Count >= 1)
                childBtns[0].onClick.AddListener(() =>
                {
                    SelectTrajectoryForExport(capturedFile, false);
                    OnPlaybackMeta(capturedFile);
                });
            if (childBtns.Count >= 2)
            {
                Button deleteButton = childBtns[1];
                deleteButton.onClick.AddListener(() => RequestDeleteTrajectory(capturedFile, deleteButton));
            }
        }

        // The commercial UI receives this notification synchronously. New
        // items stay hidden until their dark skin, geometry, and ordering are
        // all applied, eliminating the one-frame white prefab flash.
        TrajectoryListRefreshed?.Invoke();
        foreach (GameObject item in newItems)
            if (item != null) item.SetActive(true);
    }

    public bool IsTrajectorySelectedForExport(string metadataPath)
    {
        return PathsEqual(metadataPath, _selectedExportMetaPath);
    }

    public bool IsTrajectoryItemSelectedForExport(GameObject item)
    {
        return item != null &&
               _trajectoryItemMetadataPaths.TryGetValue(item.GetInstanceID(),
                   out string metadataPath) &&
               IsTrajectorySelectedForExport(metadataPath);
    }

    public void SelectTrajectoryForExport(string metadataPath)
    {
        SelectTrajectoryForExport(metadataPath, true);
    }

    private void SelectTrajectoryForExport(string metadataPath, bool showStatus)
    {
        if (!TryGetRecordingForExport(metadataPath, out _, out _, out string displayName,
                out _, out _, out _, out string error))
        {
            if (showStatus) ShowOperationStatus("无法选择该历史记录导出：" + error);
            return;
        }

        _selectedExportMetaPath = Path.GetFullPath(metadataPath);
        TrajectoryExportSelectionChanged?.Invoke();
        if (showStatus) ShowOperationStatus("已选择导出：" + displayName);
    }

    private void OnPlaybackMeta(string metaPath)
    {
        ClearPendingDeleteConfirmation(true);
        if (!TryLoadTrajectoryMetadata(metaPath, out TrajectoryMetadata metadata, out string metadataError))
        {
            ShowOperationStatus("轨迹记录无效：" + metadataError);
            return;
        }

        OpenTrajectoryEditor(metadata);
    }

    private bool OpenTrajectoryEditor(TrajectoryMetadata metadata)
    {
        string posePath = string.IsNullOrWhiteSpace(metadata?.poseRecordingFile)
            ? null
            : ResolveStoredPath(metadata.poseRecordingFile);
        if (string.IsNullOrWhiteSpace(posePath) || !File.Exists(posePath))
        {
            ShowOperationStatus("该历史项没有人物姿态录制，无法打开关键帧编辑与动作回放界面");
            return false;
        }

        if (!RuntimePoseRecordingStorage.TryLoad(posePath,
                out RuntimePoseRecordingPackage recording, out string loadError))
        {
            ShowOperationStatus("无法读取该历史项的人物姿态：" + loadError);
            return false;
        }

        Animator targetAnimator = GetPlaybackAnimator();
        string actionId = MotionPackageStorage.IsValidIdentifier(recording.actionId)
            ? recording.actionId
            : "recorded_motion";
        string displayName = !string.IsNullOrWhiteSpace(recording.displayName)
            ? recording.displayName.Trim()
            : metadata.name;

        if (_isPlayingBack) EndPlayback();
        if (!RuntimeRecordedMotionEditor.TryOpen(this, targetAnimator, posePath,
                actionId, displayName, out string editorError))
        {
            ShowOperationStatus("无法打开关键帧编辑与动作回放界面：" + editorError);
            return false;
        }

        ShowOperationStatus("已打开历史动作：" + metadata.name);
        return true;
    }

    private bool StartPosePlayback(string path)
    {
        Animator targetAnimator = GetPlaybackAnimator();
        if (!RuntimePosePlaybackUtility.IsValidHumanoid(targetAnimator))
        {
            ShowOperationStatus("人物回放失败：没有绑定有效的 Humanoid Animator");
            return false;
        }

        if (!RuntimePoseRecordingStorage.TryLoad(path, out RuntimePoseRecordingPackage recording,
                out string error))
        {
            ShowOperationStatus(error);
            return false;
        }

        return StartPosePlayback(recording, Path.GetFileName(path));
    }

    private bool StartPosePlayback(RuntimePoseRecordingPackage recording, string displayLabel)
    {
        Animator targetAnimator = GetPlaybackAnimator();
        if (!RuntimePosePlaybackUtility.IsValidHumanoid(targetAnimator))
        {
            ShowOperationStatus("人物回放失败：没有绑定有效的 Humanoid Animator");
            return false;
        }

        if (recording?.frames == null || recording.frames.Length == 0)
        {
            ShowOperationStatus("人物回放失败：动作中没有可用姿态帧");
            return false;
        }

        if (_isPlayingBack) EndPlayback();
        _posePlaybackHandler?.Dispose();
        _posePlaybackHandler = new HumanPoseHandler(targetAnimator.avatar, targetAnimator.transform);
        _posePlaybackPose = new HumanPose { muscles = new float[HumanTrait.MuscleCount] };
        _loadedPoseRecording = recording;
        _loadedTrackerRecording = null;
        _currentPlaybackClip = null;
        _isRawImuPlayback = false;
        _posePlaybackTime = 0f;
        _posePlaybackFrameIndex = 0;
        _isPlayingBack = true;
        var receiver = FindObjectOfType<EVMC4U.ExternalReceiver>();
        if (receiver != null) receiver.Freeze = true;
        RuntimePosePlaybackUtility.ApplyFrame(recording, 0, _posePlaybackHandler, ref _posePlaybackPose);
        if (txtStatus != null)
            txtStatus.text = "人物动作回放中：" +
                             (string.IsNullOrWhiteSpace(displayLabel) ? "导入动作" : displayLabel);
        return true;
    }

    private void UpdatePosePlayback()
    {
        RuntimePoseFrame[] frames = _loadedPoseRecording?.frames;
        if (frames == null || frames.Length == 0 || _posePlaybackHandler == null)
        {
            EndPlayback();
            return;
        }

        float duration = Mathf.Max(_loadedPoseRecording.durationSeconds,
            frames[frames.Length - 1].timeSeconds);
        _posePlaybackTime += Time.unscaledDeltaTime;
        if (duration <= 0f || _posePlaybackTime >= duration)
        {
            RuntimePosePlaybackUtility.ApplyFrame(_loadedPoseRecording, frames.Length - 1,
                _posePlaybackHandler, ref _posePlaybackPose);
            EndPlayback();
            return;
        }

        RuntimePosePlaybackUtility.ApplyAtTime(_loadedPoseRecording, _posePlaybackTime,
            ref _posePlaybackFrameIndex, _posePlaybackHandler, ref _posePlaybackPose);
        if (txtStatus != null)
            txtStatus.text = $"人物动作回放中... {_posePlaybackTime:F1}s / {duration:F1}s";
    }

    private Animator GetPlaybackAnimator()
    {
        if (playbackAnimator != null) return playbackAnimator;
        if (poseRecorder != null && poseRecorder.AvatarAnimator != null)
            playbackAnimator = poseRecorder.AvatarAnimator;
        return playbackAnimator;
    }

    public void OnPlaybackLatest()
    {
        string latestPath = null;
        System.DateTime latestWriteTime = System.DateTime.MinValue;
        foreach (string file in Directory.GetFiles(_metaDataPath, "*.json"))
        {
            System.DateTime writeTime = File.GetLastWriteTimeUtc(file);
            if (writeTime <= latestWriteTime) continue;
            latestWriteTime = writeTime;
            latestPath = file;
        }

        if (!string.IsNullOrWhiteSpace(latestPath))
            OnPlaybackMeta(latestPath);
    }

    /// <summary>
    /// Opens IMUTrack's own keyframe/FBX/JSON post-processing window.
    /// Bind this method to a button in the IMUTrack recording UI.
    /// </summary>
    public void OnOpenPostProcessor()
    {
        if (_isRecording)
        {
            if (txtStatus != null) txtStatus.text = "Stop recording before processing it.";
            return;
        }

        if (!TryGetLatestRecordingForExport(out string posePath, out string actionId,
                out string displayName, out _, out string identityError))
        {
            ShowOperationStatus("无法打开关键帧编辑器：" + identityError);
            return;
        }

#if UNITY_EDITOR
        UnityEditor.EditorApplication.ExecuteMenuItem("IMUTrack/Process Latest Recording");
#else
        if (_isPlayingBack) EndPlayback();
        Animator targetAnimator = GetPlaybackAnimator();
        if (!RuntimeRecordedMotionEditor.TryOpen(this, targetAnimator, posePath,
                actionId, displayName, out string editorError))
        {
            ShowOperationStatus("无法打开关键帧编辑器：" + editorError);
            return;
        }

        ShowOperationStatus("关键帧编辑器已打开");
#endif
    }

    /// <summary>
    /// Exports the history item selected by the user as one complete FBX clip and one
    /// dead_bug_demo.motion.json package without opening another window.
    /// Bind this method to a button in the existing IMUTrack Game UI.
    /// </summary>
    public void OnExportLatestMotionPackage()
    {
        if (_isRecording)
        {
            ShowOperationStatus("请先停止录制");
            return;
        }

        if (!TryGetRecordingForExport(_selectedExportMetaPath,
                out string posePath, out string actionId, out string displayName, out _,
                out string trackerJsonPath, out string rawImuJsonPath, out string identityError))
        {
            ShowOperationStatus("导出失败：请先在录制历史中选择一条可导出的记录。" + identityError);
            return;
        }

        ShowOperationStatus("正在导出动作包...");
        if (!RuntimeMotionPackageExporter.TryExport(
                posePath, trackerJsonPath, rawImuJsonPath,
                actionId, displayName, out MotionPackageManifest manifest,
                out string manifestPath, out string exportError))
        {
            ShowOperationStatus(exportError);
            Debug.LogError("[Runtime Motion Export] " + exportError, this);
            return;
        }

        ShowOperationStatus($"导出成功：{actionId} v{manifest.version}，目录：" +
                            (Path.GetDirectoryName(manifestPath) ?? manifestPath));
        Debug.Log($"[Runtime Motion Export] Manifest: {manifestPath}", this);
    }

    public string GetLatestPoseRecordingPath()
    {
        if (poseRecorder != null && !string.IsNullOrWhiteSpace(poseRecorder.LastRuntimeRecordingPath) &&
            File.Exists(poseRecorder.LastRuntimeRecordingPath))
            return poseRecorder.LastRuntimeRecordingPath;
        return RuntimePoseRecordingStorage.TryFindLatest(out string latest) ? latest : null;
    }

    public bool TryGetLatestRecordingForExport(out string posePath, out string actionId,
        out string displayName, out string trajectoryName, out string error)
    {
        posePath = GetLatestPoseRecordingPath();
        actionId = null;
        displayName = null;
        trajectoryName = null;
        error = null;

        if (string.IsNullOrWhiteSpace(posePath) || !File.Exists(posePath))
        {
            error = "没有姿态录制，请先完成一次录制";
            return false;
        }

        if (!RuntimePoseRecordingStorage.TryLoad(
                posePath, out RuntimePoseRecordingPackage recording, out error))
            return false;

        actionId = recording.actionId?.Trim();
        if (!MotionPackageStorage.IsValidIdentifier(actionId))
        {
            error = "录制开始时指定的 actionId 无效";
            return false;
        }

        displayName = string.IsNullOrWhiteSpace(recording.displayName)
            ? actionId
            : recording.displayName.Trim();
        trajectoryName = string.IsNullOrWhiteSpace(recording.trajectoryName)
            ? MotionPackageStorage.GetPackageFolderName(actionId, displayName)
            : recording.trajectoryName.Trim();
        return true;
    }

    private bool TryGetRecordingForExport(string metadataPath, out string posePath,
        out string actionId, out string displayName, out string trajectoryName,
        out string trackerJsonPath, out string rawImuJsonPath, out string error)
    {
        posePath = null;
        actionId = null;
        displayName = null;
        trajectoryName = null;
        trackerJsonPath = null;
        rawImuJsonPath = null;
        error = null;

        if (string.IsNullOrWhiteSpace(metadataPath))
        {
            error = "当前未选择历史记录";
            return false;
        }

        if (!TryLoadTrajectoryMetadata(metadataPath,
                out TrajectoryMetadata metadata, out error)) return false;

        try
        {
            posePath = string.IsNullOrWhiteSpace(metadata.poseRecordingFile)
                ? null
                : ResolveStoredPath(metadata.poseRecordingFile);
            trackerJsonPath = string.IsNullOrWhiteSpace(metadata.trackerJsonFile)
                ? null
                : ResolveStoredPath(metadata.trackerJsonFile);
            rawImuJsonPath = string.IsNullOrWhiteSpace(metadata.rawImuJsonFile)
                ? null
                : ResolveStoredPath(metadata.rawImuJsonFile);
        }
        catch (Exception exception)
        {
            error = "历史记录文件路径无效：" + exception.Message;
            return false;
        }

        if (string.IsNullOrWhiteSpace(posePath) || !File.Exists(posePath))
        {
            error = "该历史记录没有可用的人物姿态文件";
            return false;
        }

        if (!RuntimePoseRecordingStorage.TryLoad(
                posePath, out RuntimePoseRecordingPackage recording, out error)) return false;

        actionId = recording.actionId?.Trim();
        if (!MotionPackageStorage.IsValidIdentifier(actionId))
        {
            error = "该历史记录的 actionId 无效";
            return false;
        }

        displayName = string.IsNullOrWhiteSpace(recording.displayName)
            ? actionId
            : recording.displayName.Trim();
        trajectoryName = string.IsNullOrWhiteSpace(recording.trajectoryName)
            ? MotionPackageStorage.GetPackageFolderName(actionId, displayName)
            : recording.trajectoryName.Trim();
        return true;
    }

    public string GetMotionPackageActionId()
    {
        string value = inputActionId == null ? null : inputActionId.text;
        return string.IsNullOrWhiteSpace(value) ? defaultActionId.Trim() : value.Trim();
    }

    public string GetMotionPackageDisplayName()
    {
        string value = inputActionName == null ? null : inputActionName.text;
        if (!string.IsNullOrWhiteSpace(value)) return value.Trim();
        if (!string.IsNullOrWhiteSpace(defaultActionName)) return defaultActionName.Trim();
        return GetMotionPackageActionId();
    }

    private string BuildRecordingTrajectoryName(string actionIdValue, string displayNameValue)
    {
        string actionId = MakeSafeFileName(actionIdValue);
        string displayName = MakeSafeFileName(displayNameValue);
        if (string.IsNullOrWhiteSpace(actionId)) actionId = "recorded_motion";
        if (string.IsNullOrWhiteSpace(displayName)) displayName = actionId;

        string baseName = string.Equals(actionId, displayName, System.StringComparison.OrdinalIgnoreCase)
            ? actionId
            : actionId + "_" + displayName;
        return GetUniqueTrajectoryName(baseName);
    }

    private string GetUniqueTrajectoryName(string baseName)
    {
        if (string.IsNullOrWhiteSpace(baseName)) baseName = "recorded_motion";
        if (string.IsNullOrWhiteSpace(_metaDataPath)) return baseName;

        string candidate = baseName;
        int suffix = 2;
        while (File.Exists(Path.Combine(_metaDataPath, candidate + ".json")))
        {
            candidate = baseName + "_" + suffix;
            suffix++;
        }

        return candidate;
    }

    private static string MakeSafeFileName(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return string.Empty;
        value = value.Trim();
        foreach (char invalid in Path.GetInvalidFileNameChars())
            value = value.Replace(invalid, '_');
        return value.TrimEnd('.', ' ');
    }

    /// <summary>
    /// Performs the two safety gates required before the reconstruction chooser is shown:
    /// no live physical tracker may be present and the SlimeVR SolarXR endpoint must respond.
    /// </summary>
    public bool TryPrepareHistoryReconstruction(out List<HistoryReconstructionCandidate> candidates,
        out string error)
    {
        candidates = new List<HistoryReconstructionCandidate>();
        error = null;

        if (IsRecording || _isHistoryReconstructing)
        {
            error = "当前正在录制或重构，请先等待操作结束";
            return false;
        }

        RawImuDataSource.TryStart(out _);
        if (RawImuDataSource.TryCopyLatestSamples(_rawImuSampleBuffer, out _))
        {
            int onlineCount = 0;
            foreach (RawImuSensorSample sample in _rawImuSampleBuffer)
                if (sample != null && sample.online) onlineCount++;
            if (onlineCount > 0)
            {
                error = $"检测到 {onlineCount} 个真实 Tracker 在线。为避免与虚拟回灌数据冲突，请先关闭 Tracker，再进行历史重构";
                return false;
            }
        }

        if (!IsSlimeVrServerListening())
        {
            error = "未检测到 SlimeVR 服务。请先启动 SlimeVR，并保持 SolarXR(21110) 与 Tracker UDP(6969) 可用";
            return false;
        }

        EnsureMetadataDirectory();
        foreach (string metaPath in Directory.GetFiles(_metaDataPath, "*.json"))
        {
            if (!TryLoadTrajectoryMetadata(metaPath, out TrajectoryMetadata metadata, out _) ||
                string.IsNullOrWhiteSpace(metadata.rawImuJsonFile))
                continue;

            string rawPath;
            try { rawPath = ResolveStoredPath(metadata.rawImuJsonFile); }
            catch { continue; }
            if (!RawImuRecordingStorage.TryLoad(rawPath, out RawImuRecordingPackage package, out _))
                continue;

            var sensors = BuildReconstructionSensors(package);
            if (sensors.Count == 0) continue;
            candidates.Add(new HistoryReconstructionCandidate
            {
                metadataPath = metaPath,
                name = metadata.name,
                recordedAt = metadata.recordedAt,
                duration = metadata.duration,
                rawFormatVersion = package.formatVersion,
                sensors = sensors.ToArray()
            });
        }

        candidates.Sort((left, right) => string.CompareOrdinal(right.recordedAt, left.recordedAt));
        if (candidates.Count == 0)
        {
            error = "录制历史中没有包含 Raw IMU 数据、可供重构的记录";
            return false;
        }

        if (_isPlayingBack) EndPlayback();
        return true;
    }

    public bool StartHistoryReconstruction(string metadataPath, IEnumerable<string> sensorIds,
        out string error)
    {
        error = null;
        if (_isHistoryReconstructing || IsRecording)
        {
            error = "当前正在录制或重构";
            return false;
        }

        var selected = new HashSet<string>(StringComparer.Ordinal);
        if (sensorIds != null)
            foreach (string sensorId in sensorIds)
                if (!string.IsNullOrWhiteSpace(sensorId)) selected.Add(sensorId);
        if (selected.Count == 0)
        {
            error = "请至少选择一个 Tracker 部位";
            return false;
        }

        // Repeat both gates at execution time: the chooser may have been open for a while.
        if (RawImuDataSource.TryCopyLatestSamples(_rawImuSampleBuffer, out _))
        {
            foreach (RawImuSensorSample sample in _rawImuSampleBuffer)
            {
                if (sample == null || !sample.online) continue;
                error = "检测到真实 Tracker 已上线，已阻止重构以避免数据冲突";
                return false;
            }
        }
        if (!IsSlimeVrServerListening())
        {
            error = "SlimeVR 服务已断开，请重新启动后再试";
            return false;
        }

        if (!TryLoadTrajectoryMetadata(metadataPath, out TrajectoryMetadata metadata,
                out string metadataError))
        {
            error = "无法读取原历史记录：" + metadataError;
            return false;
        }
        string rawError = "原记录未关联 Raw IMU 数据";
        if (string.IsNullOrWhiteSpace(metadata.rawImuJsonFile) ||
            !RawImuRecordingStorage.TryLoad(ResolveStoredPath(metadata.rawImuJsonFile),
                out RawImuRecordingPackage package, out rawError))
        {
            error = "无法读取原记录的 Raw IMU 数据：" + rawError;
            return false;
        }

        var validSelected = new HashSet<string>(StringComparer.Ordinal);
        foreach (string available in package.sensorIds)
            if (selected.Contains(available)) validSelected.Add(available);
        if (validSelected.Count == 0)
        {
            error = "所选 Tracker 不存在于该记录中";
            return false;
        }

        StartCoroutine(ReconstructHistoryCoroutine(metadataPath, metadata, package, validSelected));
        return true;
    }

    private IEnumerator ReconstructHistoryCoroutine(string sourceMetadataPath,
        TrajectoryMetadata sourceMetadata, RawImuRecordingPackage package,
        HashSet<string> selectedSensorIds)
    {
        _isHistoryReconstructing = true;
        HistoryReconstructionStateChanged?.Invoke(true);
        if (_isPlayingBack) EndPlayback();
        if (btnRecord != null) btnRecord.interactable = false;
        if (btnCalibrate != null) btnCalibrate.interactable = false;
        ShowOperationStatus($"正在建立 {selectedSensorIds.Count} 个虚拟 Tracker，请保持 SlimeVR 运行...");

        var definitions = FindSelectedSensorDefinitions(package, selectedSensorIds);
        SlimeVrVirtualTrackerReplay replay = null;
        string failure = null;
        try
        {
            replay = new SlimeVrVirtualTrackerReplay(definitions);
        }
        catch (Exception exception)
        {
            failure = "无法建立 SlimeVR 虚拟 Tracker：" + exception.Message;
        }

        if (failure == null)
        {
            // Registration pre-roll: hold the first recorded orientation so SlimeVR can
            // create, assign and solve every selected virtual tracker before capture.
            float preRollStart = Time.unscaledTime;
            EVMC4U.ExternalReceiver externalReceiver = FindObjectOfType<EVMC4U.ExternalReceiver>();
            bool receivedVmcPose = false;
            int maximumOnlineReplayTrackers = 0;
            var verificationSamples = new List<RawImuSensorSample>();
            while (Time.unscaledTime - preRollStart < 1.5f)
            {
                replay.SendFrame(package.frames[0], selectedSensorIds);
                if (Time.frameCount % 15 == 0) replay.SendSensorInfo();
                receivedVmcPose |= (externalReceiver != null &&
                                    externalReceiver.LastPacketframeCounterInFrame > 0) ||
                                   (vmcBridge != null && vmcBridge.HasRecentPoseMessages());
                if (SlimeVrRawImuDataSource.TryCopyLatestSamples(verificationSamples, true, out _))
                {
                    int onlineReplayTrackers = 0;
                    foreach (RawImuSensorSample sample in verificationSamples)
                    {
                        if (sample != null && sample.online &&
                            IsReconstructionLoopbackTracker(sample.sensorId))
                            onlineReplayTrackers++;
                    }
                    maximumOnlineReplayTrackers = Mathf.Max(maximumOnlineReplayTrackers,
                        onlineReplayTrackers);
                }
                yield return null;
            }

            if (maximumOnlineReplayTrackers < selectedSensorIds.Count)
            {
                failure = $"SlimeVR 仅接纳了 {maximumOnlineReplayTrackers}/{selectedSensorIds.Count} 个虚拟 Tracker，已停止重构，避免生成静止姿态。请等待 SlimeVR 完全启动后重试";
                replay.Dispose();
                _isHistoryReconstructing = false;
                if (btnRecord != null) btnRecord.interactable = true;
                if (btnCalibrate != null) btnCalibrate.interactable = true;
                HistoryReconstructionStateChanged?.Invoke(false);
                ShowOperationStatus(failure);
                yield break;
            }

            if (!receivedVmcPose)
            {
                failure = "SlimeVR 已运行，但 Unity 在 39539 端口未收到 VMC 人物姿态。请在 SlimeVR 中启用 VMC 输出并确认 Port Out=39539";
                replay.Dispose();
                _isHistoryReconstructing = false;
                if (btnRecord != null) btnRecord.interactable = true;
                if (btnCalibrate != null) btnCalibrate.interactable = true;
                HistoryReconstructionStateChanged?.Invoke(false);
                ShowOperationStatus(failure);
                yield break;
            }

            string derivedName = GetUniqueTrajectoryName(
                MakeSafeFileName(sourceMetadata.name) + $"_重构_{selectedSensorIds.Count}Tracker");
            string actionId = MotionPackageStorage.IsValidIdentifier(sourceMetadata.actionId)
                ? sourceMetadata.actionId
                : "recorded_motion";
            string displayName = sourceMetadata.name + $" · 重构 · {selectedSensorIds.Count} Tracker";
            HashSet<string> beforeFiles = GetAnimFiles();
            poseRecorder.ConfigureRuntimeMetadata(actionId, displayName, derivedName);
            poseRecorder.StartRecording();

            float replayStart = Time.unscaledTime;
            int frameIndex = 0;
            while (frameIndex < package.frames.Length)
            {
                float elapsed = Time.unscaledTime - replayStart;
                while (frameIndex < package.frames.Length &&
                       package.frames[frameIndex].timeSeconds <= elapsed + 0.0001f)
                {
                    replay.SendFrame(package.frames[frameIndex], selectedSensorIds);
                    frameIndex++;
                }

                ShowOperationStatus($"历史重构中：{Mathf.Min(elapsed, package.durationSeconds):F1}s / " +
                                    $"{package.durationSeconds:F1}s · {selectedSensorIds.Count} Tracker");
                yield return null;
            }

            float settleStart = Time.unscaledTime;
            while (Time.unscaledTime - settleStart < 0.25f)
            {
                replay.SendFrame(package.frames[package.frames.Length - 1], selectedSensorIds);
                yield return null;
            }

            poseRecorder.StopRecording();
            string posePath = poseRecorder.LastRuntimeRecordingPath;
            if (string.IsNullOrWhiteSpace(posePath) || !File.Exists(posePath))
            {
                failure = "重构完成，但未生成新的人物姿态文件";
            }
            else if (RawRecordingContainsMotion(package, selectedSensorIds) &&
                     !PoseRecordingContainsMotion(poseRecorder.LastRuntimeRecording,
                         out float maximumPoseDelta))
            {
                failure = $"虚拟 Tracker 已成功回灌，但 SlimeVR/VMC 输出的人物姿态没有变化（最大姿态差 {maximumPoseDelta:F4}）。已阻止写入无效历史记录；请确认 SlimeVR VMC 输出使用当前骨骼并重新执行重构";
                TryDeleteFailedReconstructionPose(posePath);
            }
            else
            {
                var newAnimFiles = new List<string>();
                foreach (string path in GetAnimFiles())
                    if (!beforeFiles.Contains(path)) newAnimFiles.Add(path);
                var selectedRoles = new List<string>();
                foreach (RawImuSensorSample definition in definitions)
                    selectedRoles.Add(definition.trackerRole ?? string.Empty);
                var metadata = new TrajectoryMetadata
                {
                    name = derivedName,
                    actionId = actionId,
                    displayName = displayName,
                    recordedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                    duration = poseRecorder.LastRuntimeRecording?.durationSeconds ?? package.durationSeconds,
                    poseRecordingFile = posePath,
                    trackerJsonFile = string.Empty,
                    rawImuJsonFile = string.Empty,
                    reconstructed = true,
                    reconstructionSourceName = sourceMetadata.name,
                    reconstructionSourceMetadata = sourceMetadataPath,
                    reconstructionSourceRawImuFile = ResolveStoredPath(sourceMetadata.rawImuJsonFile),
                    reconstructionTrackerCount = selectedSensorIds.Count,
                    reconstructionSensorIds = new List<string>(selectedSensorIds).ToArray(),
                    reconstructionTrackerRoles = selectedRoles.ToArray(),
                    files = newAnimFiles.ToArray()
                };
                try
                {
                    EnsureMetadataDirectory();
                    File.WriteAllText(Path.Combine(_metaDataPath, derivedName + ".json"),
                        JsonUtility.ToJson(metadata, true));
                }
                catch (Exception exception)
                {
                    failure = "重构姿态已生成，但新增历史记录失败：" + exception.Message;
                }
            }
        }

        if (poseRecorder != null && poseRecorder.IsRecording) poseRecorder.StopRecording();
        replay?.Dispose();
        _isHistoryReconstructing = false;
        if (btnRecord != null) btnRecord.interactable = true;
        if (btnCalibrate != null) btnCalibrate.interactable = true;
        RefreshTrajectoryList();
        HistoryReconstructionStateChanged?.Invoke(false);
        ShowOperationStatus(failure ??
            $"历史重构完成：原记录保持不变，已新增 {selectedSensorIds.Count} Tracker 重构记录");
    }

    private static bool IsReconstructionLoopbackTracker(string sensorId)
    {
        // SlimeVR derives the no-MAC device name from 127.0.0.42. Depending on
        // its display-name formatter this appears as either "0.0.42" or ".0.42".
        return !string.IsNullOrWhiteSpace(sensorId) &&
               sensorId.IndexOf(".0.42", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static bool RawRecordingContainsMotion(RawImuRecordingPackage package,
        ISet<string> selectedSensorIds)
    {
        if (package?.frames == null || package.frames.Length < 2) return false;
        var baseline = new Dictionary<string, RawImuSensorSample>(StringComparer.Ordinal);
        foreach (RawImuFrame frame in package.frames)
        {
            if (frame?.sensors == null) continue;
            foreach (RawImuSensorSample sample in frame.sensors)
            {
                if (sample == null || !selectedSensorIds.Contains(sample.sensorId)) continue;
                if (!baseline.TryGetValue(sample.sensorId, out RawImuSensorSample first))
                {
                    baseline[sample.sensorId] = sample;
                    continue;
                }

                if (sample.hasOrientation && first.hasOrientation &&
                    sample.orientation != null && first.orientation != null)
                {
                    double dot = Math.Abs(sample.orientation.x * first.orientation.x +
                                          sample.orientation.y * first.orientation.y +
                                          sample.orientation.z * first.orientation.z +
                                          sample.orientation.w * first.orientation.w);
                    if (dot < 0.99996d) return true; // approximately 1 degree
                }
                else if (sample.eulerAnglesDeg != null && first.eulerAnglesDeg != null &&
                         (Math.Abs(Mathf.DeltaAngle((float)first.eulerAnglesDeg.x,
                                       (float)sample.eulerAnglesDeg.x)) > 1f ||
                          Math.Abs(Mathf.DeltaAngle((float)first.eulerAnglesDeg.y,
                                       (float)sample.eulerAnglesDeg.y)) > 1f ||
                          Math.Abs(Mathf.DeltaAngle((float)first.eulerAnglesDeg.z,
                                       (float)sample.eulerAnglesDeg.z)) > 1f))
                    return true;
            }
        }
        return false;
    }

    private static bool PoseRecordingContainsMotion(RuntimePoseRecordingPackage package,
        out float maximumDelta)
    {
        maximumDelta = 0f;
        if (package?.frames == null || package.frames.Length < 2) return false;
        RuntimePoseFrame first = package.frames[0];
        if (first == null) return false;
        foreach (RuntimePoseFrame frame in package.frames)
        {
            if (frame == null) continue;
            maximumDelta = Mathf.Max(maximumDelta,
                Vector3.Distance(first.bodyPosition, frame.bodyPosition));
            maximumDelta = Mathf.Max(maximumDelta,
                Quaternion.Angle(first.bodyRotation, frame.bodyRotation) / 180f);
            if (first.muscles == null || frame.muscles == null) continue;
            int count = Mathf.Min(first.muscles.Length, frame.muscles.Length);
            for (int muscle = 0; muscle < count; muscle++)
                maximumDelta = Mathf.Max(maximumDelta,
                    Mathf.Abs(first.muscles[muscle] - frame.muscles[muscle]));
        }
        return maximumDelta > 0.002f;
    }

    private static void TryDeleteFailedReconstructionPose(string posePath)
    {
        try
        {
            string fullPath = Path.GetFullPath(posePath);
            string root = Path.GetFullPath(RuntimePoseRecordingStorage.RootDirectory)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                          Path.DirectorySeparatorChar;
            if (fullPath.StartsWith(root, StringComparison.OrdinalIgnoreCase) &&
                string.Equals(Path.GetExtension(fullPath), ".json",
                    StringComparison.OrdinalIgnoreCase))
                File.Delete(fullPath);
        }
        catch (Exception exception)
        {
            Debug.LogWarning("[History Reconstruction] Failed to remove invalid pose file: " +
                             exception.Message);
        }
    }

    private static List<HistoryReconstructionSensor> BuildReconstructionSensors(
        RawImuRecordingPackage package)
    {
        var result = new List<HistoryReconstructionSensor>();
        if (package?.sensorIds == null) return result;
        var definitions = FindSelectedSensorDefinitions(package,
            new HashSet<string>(package.sensorIds, StringComparer.Ordinal));
        foreach (RawImuSensorSample definition in definitions)
        {
            result.Add(new HistoryReconstructionSensor
            {
                sensorId = definition.sensorId,
                trackerRole = definition.trackerRole,
                hasQuaternion = definition.hasOrientation
            });
        }
        return result;
    }

    private static List<RawImuSensorSample> FindSelectedSensorDefinitions(
        RawImuRecordingPackage package, ISet<string> selected)
    {
        var byId = new Dictionary<string, RawImuSensorSample>(StringComparer.Ordinal);
        if (package?.frames != null)
        {
            foreach (RawImuFrame frame in package.frames)
            {
                if (frame?.sensors == null) continue;
                foreach (RawImuSensorSample sensor in frame.sensors)
                {
                    if (sensor == null || !selected.Contains(sensor.sensorId) ||
                        byId.ContainsKey(sensor.sensorId)) continue;
                    byId.Add(sensor.sensorId, sensor);
                }
            }
        }

        var result = new List<RawImuSensorSample>();
        if (package?.sensorIds != null)
            foreach (string sensorId in package.sensorIds)
                if (byId.TryGetValue(sensorId, out RawImuSensorSample sensor)) result.Add(sensor);
        return result;
    }

    private void EnsureMetadataDirectory()
    {
        if (string.IsNullOrWhiteSpace(_metaDataPath))
            _metaDataPath = Path.Combine(Application.persistentDataPath, "TrajectoriesMeta");
        Directory.CreateDirectory(_metaDataPath);
    }

    private static bool IsSlimeVrServerListening()
    {
        try
        {
            using (var client = new TcpClient())
            {
                IAsyncResult connect = client.BeginConnect("127.0.0.1", 21110, null, null);
                bool connected = connect.AsyncWaitHandle.WaitOne(350);
                if (!connected) return false;
                client.EndConnect(connect);
                return client.Connected;
            }
        }
        catch
        {
            return false;
        }
    }

    public void ShowOperationStatus(string message)
    {
        _operationStatusUntil = Time.unscaledTime + Mathf.Max(1f, operationStatusSeconds);
        if (txtStatus != null) txtStatus.text = message ?? string.Empty;
    }

    private void RequestDeleteTrajectory(string metaPath, Button sourceButton)
    {
        if (!IsDeleteConfirmationPending(metaPath))
        {
            ClearPendingDeleteConfirmation(true);
            _pendingDeleteMetaPath = Path.GetFullPath(metaPath);
            _pendingDeleteButton = sourceButton;
            _pendingDeleteOriginalLabel = GetButtonLabel(sourceButton);
            _pendingDeleteDeadline = Time.unscaledTime + Mathf.Max(2f, deleteConfirmationSeconds);
            SetButtonLabel(sourceButton, "!");
            ShowOperationStatus(
                $"再次点击该删除按钮以确认删除；{Mathf.CeilToInt(deleteConfirmationSeconds)} 秒后取消");
            return;
        }

        string confirmedPath = _pendingDeleteMetaPath;
        ClearPendingDeleteConfirmation(true);
        if (!TryDeleteTrajectory(confirmedPath, out int deletedFileCount, out string error))
        {
            ShowOperationStatus("删除失败：" + error);
            return;
        }

        RefreshTrajectoryList();
        ShowOperationStatus($"轨迹已删除，共移除 {deletedFileCount} 个文件");
    }

    private bool IsDeleteConfirmationPending(string metaPath)
    {
        if (string.IsNullOrWhiteSpace(_pendingDeleteMetaPath) ||
            Time.unscaledTime > _pendingDeleteDeadline ||
            string.IsNullOrWhiteSpace(metaPath))
        {
            return false;
        }

        return string.Equals(
            _pendingDeleteMetaPath,
            Path.GetFullPath(metaPath),
            System.StringComparison.OrdinalIgnoreCase);
    }

    private void ClearPendingDeleteConfirmation(bool restoreLabel)
    {
        if (restoreLabel && _pendingDeleteButton != null)
            SetButtonLabel(_pendingDeleteButton, _pendingDeleteOriginalLabel);

        _pendingDeleteMetaPath = null;
        _pendingDeleteButton = null;
        _pendingDeleteOriginalLabel = null;
        _pendingDeleteDeadline = 0f;
    }

    private static string GetButtonLabel(Button button)
    {
        Text label = button == null ? null : button.GetComponentInChildren<Text>();
        return label == null ? "X" : label.text;
    }

    private static void SetButtonLabel(Button button, string value)
    {
        Text label = button == null ? null : button.GetComponentInChildren<Text>();
        if (label != null) label.text = value ?? string.Empty;
    }

    private bool TryDeleteTrajectory(string metaPath, out int deletedFileCount, out string error)
    {
        deletedFileCount = 0;
        error = null;
        string recordingsRoot = Path.GetFullPath(Path.Combine(Application.dataPath, OutputFolder));
        var targets = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);
        var metadataTarget = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);

        if (!TryAddOwnedTarget(metaPath, _metaDataPath, ".json", metadataTarget, out error))
            return false;
        if (!TryLoadTrajectoryMetadata(metaPath, out TrajectoryMetadata metadata, out error))
            return false;

        if (metadata.files != null)
        {
            foreach (string animationPath in metadata.files)
            {
                if (string.IsNullOrWhiteSpace(animationPath)) continue;
                if (!TryAddOwnedTarget(animationPath, recordingsRoot, ".anim", targets, out error))
                    return false;

                string resolvedAnimationPath = ResolveStoredPath(animationPath);
                string metaCompanion = resolvedAnimationPath + ".meta";
                if (File.Exists(metaCompanion)) targets.Add(metaCompanion);
            }
        }

        if (!string.IsNullOrWhiteSpace(metadata.poseRecordingFile))
        {
            if (!TryAddOwnedTarget(metadata.poseRecordingFile, RuntimePoseRecordingStorage.RootDirectory,
                    RuntimePoseRecordingStorage.FileExtension, targets, out error))
                return false;

            string keyframesPath = RuntimePoseRecordingStorage.GetKeyframesPath(
                ResolveStoredPath(metadata.poseRecordingFile));
            if (File.Exists(keyframesPath) &&
                !TryAddOwnedTarget(keyframesPath, RuntimePoseRecordingStorage.RootDirectory,
                    ".keyframes.json", targets, out error))
                return false;
        }

        if (!string.IsNullOrWhiteSpace(metadata.trackerJsonFile) &&
            !TryAddOwnedTarget(metadata.trackerJsonFile, TrackerRecordingStorage.RootDirectory,
                ".tracker.json", targets, out error))
            return false;

        if (!string.IsNullOrWhiteSpace(metadata.rawImuJsonFile) &&
            !TryAddOwnedTarget(metadata.rawImuJsonFile, RawImuRecordingStorage.RootDirectory,
                RawImuRecordingStorage.FileExtension, targets, out error))
            return false;

        try
        {
            foreach (string target in targets)
            {
                if (!File.Exists(target)) continue;
                File.Delete(target);
                deletedFileCount++;
            }

            foreach (string target in metadataTarget)
            {
                if (!File.Exists(target)) continue;
                File.Delete(target);
                deletedFileCount++;
            }

#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif
            return true;
        }
        catch (System.Exception exception)
        {
            error = exception.Message;
            return false;
        }
    }

    private static bool TryAddOwnedTarget(string storedPath, string allowedRoot, string requiredSuffix,
        HashSet<string> targets, out string error)
    {
        error = null;
        string fullPath;
        string fullRoot;
        try
        {
            fullPath = ResolveStoredPath(storedPath);
            fullRoot = Path.GetFullPath(allowedRoot);
        }
        catch (System.Exception exception)
        {
            error = "文件路径无效：" + exception.Message;
            return false;
        }

        if (!IsPathInsideDirectory(fullPath, fullRoot))
        {
            error = "拒绝删除预期目录以外的文件：" + fullPath;
            return false;
        }

        if (!fullPath.EndsWith(requiredSuffix, System.StringComparison.OrdinalIgnoreCase))
        {
            error = $"文件类型不符合预期（应为 {requiredSuffix}）：{fullPath}";
            return false;
        }

        targets.Add(fullPath);
        return true;
    }

    private static bool IsPathInsideDirectory(string filePath, string directoryPath)
    {
        string root = Path.GetFullPath(directoryPath)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        string candidate = Path.GetFullPath(filePath);
        return candidate.StartsWith(root, System.StringComparison.OrdinalIgnoreCase);
    }

    private static string ResolveStoredPath(string storedPath)
    {
        if (Path.IsPathRooted(storedPath)) return Path.GetFullPath(storedPath);
        string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        return Path.GetFullPath(Path.Combine(projectRoot, storedPath));
    }

    private static bool TryLoadTrajectoryMetadata(string metaPath, out TrajectoryMetadata metadata,
        out string error)
    {
        metadata = null;
        error = null;
        try
        {
            if (string.IsNullOrWhiteSpace(metaPath) || !File.Exists(metaPath))
            {
                error = "轨迹记录不存在";
                return false;
            }

            metadata = JsonUtility.FromJson<TrajectoryMetadata>(File.ReadAllText(metaPath));
            if (metadata == null || string.IsNullOrWhiteSpace(metadata.name))
            {
                error = "轨迹记录格式无效";
                metadata = null;
                return false;
            }

            if (metadata.files == null) metadata.files = System.Array.Empty<string>();
            return true;
        }
        catch (System.Exception exception)
        {
            error = exception.Message;
            metadata = null;
            return false;
        }
    }

    // ===== 工具函数 =====
    private HashSet<string> GetAnimFiles()
    {
        var result = new HashSet<string>();
        var dir = Path.GetFullPath(Path.Combine(Application.dataPath, OutputFolder));
        if (!Directory.Exists(dir)) return result;
        foreach (var f in Directory.GetFiles(dir, "*.anim", SearchOption.AllDirectories))
            result.Add(f);
        return result;
    }

    private bool TryStartRawImuUdp(bool showError, out string error)
    {
        error = null;
        if (!enableRawImuCapture)
        {
            error = "Raw IMU capture is disabled.";
            return false;
        }

        // Do not use the component-level flag as the source of truth. With
        // Enter Play Mode options or a script/domain reload, the static SolarXR
        // worker can be reset while this MonoBehaviour field remains true.
        // TryStart is idempotent: it reuses a live worker and recreates a worker
        // that has stopped or faulted.
        _rawImuUdpStarted = RawImuDataSource.TryStart(out error);
        if (!_rawImuUdpStarted)
        {
            if (showError) ShowOperationStatus(error);
            else Debug.LogWarning("[Raw IMU] " + error, this);
            return false;
        }

        return true;
    }

    private string GetRawImuSensorCountNotice()
    {
        if (!TryStartRawImuUdp(false, out _)) return null;
        if (!RawImuDataSource.TryCopyLatestSamples(_rawImuSampleBuffer, out _)) return null;

        int onlineCount = 0;
        foreach (RawImuSensorSample sample in _rawImuSampleBuffer)
            if (sample != null && sample.online) onlineCount++;

        int requiredCount = Mathf.Max(1, minimumRawImuSensors);
        return onlineCount >= requiredCount
            ? $"{onlineCount} raw IMU tracker(s) online"
            : $"demo target {requiredCount} tracker(s), currently {onlineCount}";
    }

    private bool TryCalibrateRawImu(out string message)
    {
        if (!enableRawImuCapture)
        {
            message = "Raw IMU capture is disabled";
            return false;
        }

        if (!TryStartRawImuUdp(false, out string startError))
        {
            message = startError;
            return false;
        }

        if (!RawImuDataSource.TryCopyLatestSamples(_rawImuSampleBuffer, out string sampleError))
        {
            message = sampleError;
            return false;
        }

        int calibratedCount = 0;
        string capturedAtUtc = System.DateTime.UtcNow.ToString("o");
        foreach (RawImuSensorSample sample in _rawImuSampleBuffer)
        {
            if (sample == null || !sample.online || string.IsNullOrWhiteSpace(sample.sensorId)) continue;

            UpsertRawImuCalibration(new RawImuCalibration
            {
                sensorId = sample.sensorId,
                capturedAtUtc = capturedAtUtc,
                eulerBaselineDeg = sample.eulerAnglesDeg?.Clone() ?? new RawImuVector3(),
                hasOrientationBaseline = sample.hasOrientation,
                orientationBaseline = sample.orientation?.Clone() ?? new RawImuQuaternion()
            });
            calibratedCount++;
        }

        if (calibratedCount == 0)
        {
            message = "No online raw IMU sensor was found";
            return false;
        }

        message = $"Raw IMU calibrated: {calibratedCount} sensor(s)";
        Debug.Log("[Raw IMU] " + message, this);
        return true;
    }

    private void UpsertRawImuCalibration(RawImuCalibration calibration)
    {
        for (int index = 0; index < _rawImuCalibrations.Count; index++)
        {
            if (!string.Equals(_rawImuCalibrations[index].sensorId, calibration.sensorId,
                    System.StringComparison.Ordinal)) continue;
            _rawImuCalibrations[index] = calibration;
            return;
        }

        _rawImuCalibrations.Add(calibration);
        _rawImuCalibrations.Sort((left, right) => string.CompareOrdinal(left.sensorId, right.sensorId));
    }

    private void CaptureRawImuFrame()
    {
        if (!enableRawImuCapture) return;

        float elapsed = Mathf.Max(0f, Time.unscaledTime - _recordStartTime);
        if (elapsed + 0.0001f < _nextRawImuSampleTime) return;
        _nextRawImuSampleTime = elapsed + 1f / Mathf.Max(1f, rawImuSampleRate);

        if (!RawImuDataSource.TryCopyLatestSamples(_rawImuSampleBuffer, out _) ||
            _rawImuSampleBuffer.Count == 0)
            return;

        var samples = new RawImuSensorSample[_rawImuSampleBuffer.Count];
        for (int index = 0; index < samples.Length; index++)
            samples[index] = _rawImuSampleBuffer[index].Clone();

        _rawImuFrames.Add(new RawImuFrame
        {
            timeSeconds = elapsed,
            capturedAtUtc = System.DateTime.UtcNow.ToString("o"),
            sensors = samples
        });
    }

    private bool TrySaveRawImuRecording(float durationSeconds, out string message)
    {
        message = null;
        if (!enableRawImuCapture)
        {
            message = "Raw IMU capture disabled";
            return false;
        }

        if (_rawImuFrames.Count == 0)
        {
            message = "no raw IMU frames received";
            return false;
        }

        if (!TryReadCoachProfile(out CoachProfileData coach, out message)) return false;

        var sensorIds = new HashSet<string>(System.StringComparer.Ordinal);
        foreach (RawImuFrame frame in _rawImuFrames)
        {
            if (frame?.sensors == null) continue;
            foreach (RawImuSensorSample sensor in frame.sensors)
                if (sensor != null && !string.IsNullOrWhiteSpace(sensor.sensorId)) sensorIds.Add(sensor.sensorId);
        }

        var sortedSensorIds = new List<string>(sensorIds);
        sortedSensorIds.Sort(System.StringComparer.Ordinal);
        float sampleRate = durationSeconds > 0f && _rawImuFrames.Count > 1
            ? (_rawImuFrames.Count - 1) / durationSeconds
            : 0f;

        var package = new RawImuRecordingPackage
        {
            actionId = _recordingActionId,
            displayName = _recordingDisplayName,
            trajectoryName = _currentTrajectoryName,
            recordedAtUtc = System.DateTime.UtcNow.ToString("o"),
            source = GetRawImuPackageSource(),
            sampleRate = sampleRate,
            durationSeconds = durationSeconds,
            frameCount = _rawImuFrames.Count,
            sensorIds = sortedSensorIds.ToArray(),
            coach = new RawImuCoachProfile
            {
                heightCm = coach.heightCm,
                weightKg = coach.weightKg,
                sex = coach.sex
            },
            calibrations = _rawImuCalibrations.ToArray(),
            frames = _rawImuFrames.ToArray()
        };

        if (!RawImuRecordingStorage.TrySave(package, out _lastRawImuJsonPath, out message)) return false;

        message = $"Raw IMU JSON saved ({package.frameCount} frames, {package.sensorIds.Length} sensors)";
        Debug.Log($"[Raw IMU] {message}\n{_lastRawImuJsonPath}", this);
        if (inputRawImuJsonPath != null) inputRawImuJsonPath.text = _lastRawImuJsonPath;
        return true;
    }

    private string GetRawImuPackageSource()
    {
        foreach (RawImuFrame frame in _rawImuFrames)
        {
            if (frame?.sensors == null) continue;
            foreach (RawImuSensorSample sensor in frame.sensors)
                if (sensor != null && !string.IsNullOrWhiteSpace(sensor.source)) return sensor.source;
        }

        return "Direct physical IMU tracker data";
    }

    private string BuildRecordingResultStatus(int animationFileCount, bool poseSaved, bool trackerSaved,
        string trackerMessage, bool rawImuSaved, string rawImuMessage)
    {
        var parts = new List<string>
        {
            poseSaved ? "avatar pose saved" : "avatar pose save failed",
            animationFileCount > 0
                ? $"{animationFileCount} animation file(s) saved"
                : "no new .anim file"
        };
        parts.Add(trackerSaved ? "Tracker JSON saved" : trackerMessage);
        if (enableRawImuCapture)
            parts.Add(rawImuSaved ? "Raw IMU JSON saved" : rawImuMessage);
        return "Recording complete: " + string.Join("; ", parts);
    }

    private void CaptureTrackerFrame()
    {
        float elapsed = Mathf.Max(0f, Time.unscaledTime - _recordStartTime);
        if (elapsed + 0.0001f < _nextTrackerSampleTime) return;
        _nextTrackerSampleTime = elapsed + 1f / Mathf.Max(1f, trackerSampleRate);

        _trackerPoseBuffer.Clear();
        if (trackerPoseSource != null && trackerPoseSource.CopyLatestTrackerPoses(_trackerPoseBuffer) > 0)
            _trackerDataSource = trackerPoseSource.CurrentSourceDescription;
        if (_trackerPoseBuffer.Count == 0)
        {
            CopyDeviceReceiverTrackerPoses(_trackerPoseBuffer);
            if (_trackerPoseBuffer.Count > 0) _trackerDataSource = "EVMC4U DeviceReceiver tracker transforms";
        }
        if (_trackerPoseBuffer.Count == 0 && vmcBridge != null)
        {
            vmcBridge.CopyLatestTrackerPoses(_trackerPoseBuffer);
            if (_trackerPoseBuffer.Count > 0) _trackerDataSource = "SlimeVR VMC tracker poses (debug bridge)";
        }
        if (_trackerPoseBuffer.Count == 0) return;

        var poses = new TrackerPoseData[_trackerPoseBuffer.Count];
        for (int index = 0; index < poses.Length; index++)
            poses[index] = _trackerPoseBuffer[index].Clone();

        _trackerFrames.Add(new TrackerRecordingFrame
        {
            timeSeconds = elapsed,
            trackers = poses
        });
    }

    private void CopyDeviceReceiverTrackerPoses(List<TrackerPoseData> destination)
    {
        if (trackerDeviceReceiver?.Serials == null || trackerDeviceReceiver.Transforms == null) return;

        int count = Mathf.Min(trackerDeviceReceiver.Serials.Length, trackerDeviceReceiver.Transforms.Length);
        for (int index = 0; index < count; index++)
        {
            string trackerName = trackerDeviceReceiver.Serials[index];
            Transform trackerTransform = trackerDeviceReceiver.Transforms[index];
            if (string.IsNullOrWhiteSpace(trackerName) || trackerTransform == null) continue;

            destination.Add(new TrackerPoseData
            {
                name = trackerName,
                position = trackerTransform.localPosition,
                rotation = trackerTransform.localRotation
            });
        }

        destination.Sort((left, right) => string.CompareOrdinal(left.name, right.name));
    }

    private bool TrySaveTrackerRecording(float durationSeconds, out string message)
    {
        message = null;
        if (_trackerFrames.Count == 0)
        {
            message = "没有收到 Tracker 姿态，请检查 SlimeVR VMC 输出和端口";
            return false;
        }

        if (!TryReadCoachProfile(out CoachProfileData coach, out message)) return false;

        var trackerNames = new HashSet<string>(System.StringComparer.Ordinal);
        foreach (TrackerRecordingFrame frame in _trackerFrames)
        {
            if (frame.trackers == null) continue;
            foreach (TrackerPoseData tracker in frame.trackers)
                if (tracker != null && !string.IsNullOrWhiteSpace(tracker.name)) trackerNames.Add(tracker.name);
        }

        var sortedNames = new List<string>(trackerNames);
        sortedNames.Sort(System.StringComparer.Ordinal);
        float sampleRate = durationSeconds > 0f && _trackerFrames.Count > 1
            ? (_trackerFrames.Count - 1) / durationSeconds
            : 0f;

        var package = new TrackerRecordingPackage
        {
            actionId = _recordingActionId,
            displayName = _recordingDisplayName,
            trajectoryName = _currentTrajectoryName,
            recordedAtUtc = System.DateTime.UtcNow.ToString("o"),
            source = string.IsNullOrWhiteSpace(_trackerDataSource)
                ? "SlimeVR VMC pose stream"
                : _trackerDataSource,
            sampleRate = sampleRate,
            durationSeconds = durationSeconds,
            frameCount = _trackerFrames.Count,
            trackerNames = sortedNames.ToArray(),
            coach = coach,
            frames = _trackerFrames.ToArray()
        };

        if (!TrackerRecordingStorage.TrySave(package, out _lastTrackerJsonPath, out message)) return false;

        message = $"Tracker JSON 已保存（{package.frameCount} 帧，{package.trackerNames.Length} 个 Tracker）";
        Debug.Log($"[Tracker Recording] {message}\n{_lastTrackerJsonPath}", this);
        if (inputTrackerJsonPath != null) inputTrackerJsonPath.text = _lastTrackerJsonPath;
        return true;
    }

    private bool TryReadCoachProfile(out CoachProfileData profile, out string error)
    {
        profile = new CoachProfileData();
        error = null;
        bool hasCoachUi = inputCoachHeightCm != null || inputCoachWeightKg != null || dropdownCoachSex != null;

        if (hasCoachUi && (inputCoachHeightCm == null || inputCoachWeightKg == null || dropdownCoachSex == null))
        {
            error = "教练信息 UI 未完整挂载：需要身高、体重和性别三个控件";
            return false;
        }

        if (hasCoachUi && (string.IsNullOrWhiteSpace(inputCoachHeightCm.text) ||
                           string.IsNullOrWhiteSpace(inputCoachWeightKg.text)))
        {
            error = "请先填写教练身高和体重";
            return false;
        }

        if (inputCoachHeightCm != null && !string.IsNullOrWhiteSpace(inputCoachHeightCm.text))
        {
            if (!TryParseFloat(inputCoachHeightCm.text, out profile.heightCm) ||
                profile.heightCm < 50f || profile.heightCm > 250f)
            {
                error = "教练身高应为 50-250 cm";
                return false;
            }
        }

        if (inputCoachWeightKg != null && !string.IsNullOrWhiteSpace(inputCoachWeightKg.text))
        {
            if (!TryParseFloat(inputCoachWeightKg.text, out profile.weightKg) ||
                profile.weightKg < 20f || profile.weightKg > 300f)
            {
                error = "教练体重应为 20-300 kg";
                return false;
            }
        }

        if (dropdownCoachSex != null && dropdownCoachSex.options != null && dropdownCoachSex.options.Count > 0)
        {
            int index = Mathf.Clamp(dropdownCoachSex.value, 0, dropdownCoachSex.options.Count - 1);
            string label = dropdownCoachSex.options[index].text?.Trim();
            profile.sex = NormalizeSex(label);
            if (profile.sex == "unspecified")
            {
                error = "请选择教练性别";
                return false;
            }
        }

        return true;
    }

    public string GetLatestTrackerJsonPath()
    {
        if (!string.IsNullOrWhiteSpace(_lastTrackerJsonPath) && File.Exists(_lastTrackerJsonPath))
            return _lastTrackerJsonPath;
        return TrackerRecordingStorage.TryFindLatest(out string latest) ? latest : null;
    }

    public string GetLatestRawImuJsonPath()
    {
        if (!string.IsNullOrWhiteSpace(_lastRawImuJsonPath) && File.Exists(_lastRawImuJsonPath))
            return _lastRawImuJsonPath;
        return RawImuRecordingStorage.TryFindLatest(out string latest) ? latest : null;
    }

    public string GetLatestRawImuJsonPathForAction(string actionId)
    {
        string path = GetLatestRawImuJsonPath();
        if (string.IsNullOrWhiteSpace(path)) return null;
        return RawImuRecordingStorage.TryLoad(path, out RawImuRecordingPackage package, out _) &&
               string.Equals(package.actionId, actionId, System.StringComparison.Ordinal)
            ? path
            : null;
    }

    public void OnImportRawImuJson()
    {
        if (_isRecording)
        {
            ShowOperationStatus("Stop recording before importing Raw IMU JSON.");
            return;
        }

        string typedPath = inputRawImuJsonPath == null ? null : inputRawImuJsonPath.text?.Trim();
        // The import button doubles as the local-file picker. If the picker is
        // cancelled (or the current platform has no native picker), retain the
        // manually pasted path as a fallback.
        if (!TrySelectLocalJsonFile(out string path))
            path = typedPath;

        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            ShowOperationStatus(string.IsNullOrWhiteSpace(path)
                ? "请选择本地 JSON 动作包，或在输入框中粘贴完整路径。"
                : "找不到指定的 JSON 文件，请重新选择本地动作包。");
            return;
        }

        // Keep the existing UnityEvent binding, but route exported motion
        // packages to the Humanoid pose importer instead of the Raw IMU path.
        if (LooksLikeMotionPackagePath(path))
        {
            ImportMotionPackageJson(path, true);
            return;
        }

        ImportRawImuJson(path, true);
    }

    /// <summary>
    /// Optional explicit entry point for a dedicated motion-package button.
    /// The existing Recorder scene can keep calling OnImportRawImuJson because
    /// that method now detects manifest.json and .motion.json automatically.
    /// </summary>
    public void OnImportMotionPackageJson()
    {
        if (_isRecording)
        {
            ShowOperationStatus("请先停止录制，再导入动作包。");
            return;
        }

        string typedPath = inputRawImuJsonPath == null ? null : inputRawImuJsonPath.text?.Trim();
        if (!TrySelectLocalJsonFile(out string path))
            path = typedPath;

        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            ShowOperationStatus(string.IsNullOrWhiteSpace(path)
                ? "请选择 manifest.json 或 .motion.json。"
                : "找不到指定的动作包 JSON 文件。");
            return;
        }

        ImportMotionPackageJson(path, true);
    }

    public void OnPlayRawImuJson()
    {
        if (_isRawImuPlayback && _isRawImuPlaybackPaused)
        {
            _isRawImuPlaybackPaused = false;
            ShowOperationStatus("Raw IMU playback resumed.");
            return;
        }

        string typedPath = inputRawImuJsonPath == null ? null : inputRawImuJsonPath.text?.Trim();
        if (LooksLikeMotionPackagePath(typedPath) ||
            (string.IsNullOrWhiteSpace(typedPath) && _pendingImportedPoseRecording != null))
        {
            if (_pendingImportedPoseRecording == null ||
                (!string.IsNullOrWhiteSpace(typedPath) &&
                 !PathsEqual(typedPath, _lastImportedMotionSourcePath)))
            {
                if (!ImportMotionPackageJson(typedPath, false)) return;
            }

            StartPosePlayback(_pendingImportedPoseRecording,
                Path.GetFileName(_lastImportedMotionSourcePath));
            return;
        }

        if (_loadedRawImuRecording == null)
        {
            OnImportRawImuJson();
            if (_pendingImportedPoseRecording != null &&
                LooksLikeMotionPackagePath(inputRawImuJsonPath == null
                    ? null
                    : inputRawImuJsonPath.text?.Trim()))
            {
                StartPosePlayback(_pendingImportedPoseRecording,
                    Path.GetFileName(_lastImportedMotionSourcePath));
                return;
            }
            if (_loadedRawImuRecording == null) return;
        }

        StartRawImuPlayback();
    }

    public void OnToggleRawImuPreviewPause()
    {
        if (!_isRawImuPlayback)
        {
            ShowOperationStatus("Raw IMU playback is not running.");
            return;
        }

        _isRawImuPlaybackPaused = !_isRawImuPlaybackPaused;
        ShowOperationStatus(_isRawImuPlaybackPaused
            ? "Raw IMU playback paused."
            : "Raw IMU playback resumed.");
    }

    public void OnStopRawImuPreview()
    {
        if (_isRawImuPlayback)
        {
            EndRawImuPlayback("Raw IMU playback stopped; live preview restored.");
            return;
        }

        if (rawImuTrackerPreview == null)
            rawImuTrackerPreview = FindObjectOfType<RawImuTrackerPreview>();
        if (rawImuTrackerPreview != null) rawImuTrackerPreview.ReturnToLive();
        ShowOperationStatus("Raw IMU live preview active.");
    }

    private bool ImportAndPreviewRawImuJson(string path)
    {
        // History playback already points at the copied, owned Raw IMU file.
        // Do not create another history item when the user clicks its play button.
        if (!ImportRawImuJson(path, false)) return false;
        return StartRawImuPlayback();
    }

    private bool ImportRawImuJson(string path, bool registerHistory)
    {
        if (_isPlayingBack) EndPlayback();
        if (!TryLoadRawImuImport(path, out _loadedRawImuRecording, out string sourceRawImuPath,
                out string sourcePackagePath, out string error))
        {
            ShowOperationStatus(error);
            return false;
        }

        _lastRawImuJsonPath = Path.GetFullPath(sourceRawImuPath);
        _pendingImportedPoseRecording = null;
        _lastImportedMotionSourcePath = null;
        if (registerHistory && !RegisterImportedRawImu(_loadedRawImuRecording, sourcePackagePath,
                out string historyError))
        {
            ShowOperationStatus(historyError);
            return false;
        }

        if (inputRawImuJsonPath != null) inputRawImuJsonPath.text = _lastRawImuJsonPath;
        ShowOperationStatus(
            $"已导入 Raw IMU 数据：{GetImportedDisplayName(_loadedRawImuRecording)} · " +
            $"{_loadedRawImuRecording.frameCount} 帧，" +
            $"{_loadedRawImuRecording.sensorIds.Length} sensors, " +
            $"{_loadedRawImuRecording.durationSeconds:F1} 秒。人物动作请从录制历史进入编辑与回放。");
        Debug.Log($"[Raw IMU] Imported {_lastRawImuJsonPath}", this);
        return true;
    }

    private bool ImportMotionPackageJson(string path, bool registerHistory)
    {
        _lastImportedPoseRecordingPath = null;
        if (_isPlayingBack) EndPlayback();

        if (!TryLoadMotionPackageImport(path, out RecordedCoachMotionPackage motionPackage,
                out string sourceMotionPath, out string sourcePackagePath, out string error))
        {
            ShowOperationStatus(error);
            return false;
        }

        if (!TryConvertMotionPackageToPoseRecording(motionPackage, out RuntimePoseRecordingPackage pose,
                out error))
        {
            ShowOperationStatus(error);
            return false;
        }

        if (registerHistory && !RegisterImportedMotionPackage(pose, sourcePackagePath ?? sourceMotionPath,
                out error))
        {
            ShowOperationStatus(error);
            return false;
        }

        // A motion import must become the active JSON playback source. Otherwise
        // a Raw IMU package imported earlier can incorrectly win when the shared
        // "play data" button is pressed.
        _loadedRawImuRecording = null;
        _pendingImportedPoseRecording = pose;
        _lastImportedMotionSourcePath = Path.GetFullPath(sourcePackagePath ?? sourceMotionPath);
        if (inputRawImuJsonPath != null) inputRawImuJsonPath.text = sourcePackagePath ?? sourceMotionPath;
        ShowOperationStatus(
            $"已导入动作：{GetImportedDisplayName(motionPackage)} · " +
            $"{pose.frameCount} 帧，{pose.durationSeconds:F1} 秒；点击录制历史中的播放按钮进入编辑与回放。" );
        Debug.Log($"[Motion Package] Imported {sourceMotionPath} and converted to runtime pose recording.", this);
        return true;
    }

    private static bool LooksLikeMotionPackagePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) return false;
        string fileName = Path.GetFileName(path.Trim());
        return string.Equals(fileName, MotionPackageStorage.ManifestFileName,
                   StringComparison.OrdinalIgnoreCase) ||
               fileName.EndsWith(".motion.json", StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryLoadMotionPackageImport(string selectedPath,
        out RecordedCoachMotionPackage package, out string sourceMotionPath,
        out string sourcePackagePath, out string error)
    {
        package = null;
        sourceMotionPath = null;
        sourcePackagePath = null;
        error = null;

        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            error = "动作包 JSON 路径为空。";
            return false;
        }

        string fullSelectedPath;
        try
        {
            fullSelectedPath = Path.GetFullPath(selectedPath.Trim());
        }
        catch (Exception exception)
        {
            error = "动作包 JSON 路径无效：" + exception.Message;
            return false;
        }

        if (!File.Exists(fullSelectedPath))
        {
            error = "找不到动作包 JSON：" + fullSelectedPath;
            return false;
        }

        string selectedFileName = Path.GetFileName(fullSelectedPath);
        bool selectedManifest = string.Equals(selectedFileName, MotionPackageStorage.ManifestFileName,
            StringComparison.OrdinalIgnoreCase);
        string manifestPath = selectedManifest ? fullSelectedPath : null;
        string motionPath = fullSelectedPath;
        MotionPackageManifest selectedPackageManifest = null;

        try
        {
            // Selecting manifest.json is the preferred workflow because it
            // identifies the motion file and keeps the package self-contained.
            // Selecting *.motion.json is also supported, including a standalone
            // file copied without its manifest.
            if (selectedManifest)
            {
                selectedPackageManifest = JsonUtility.FromJson<MotionPackageManifest>(
                    File.ReadAllText(manifestPath, Encoding.UTF8));
                if (selectedPackageManifest == null ||
                    selectedPackageManifest.formatVersion != MotionPackageStorage.CurrentFormatVersion)
                {
                    error = "不支持的 manifest.json 格式。";
                    return false;
                }

                if (string.IsNullOrWhiteSpace(selectedPackageManifest.motionJsonFile))
                {
                    error = "manifest.json 中没有 motionJsonFile。";
                    return false;
                }

                string packageDirectory = Path.GetFullPath(
                    Path.GetDirectoryName(manifestPath) ?? string.Empty);
                motionPath = Path.GetFullPath(Path.Combine(packageDirectory,
                    selectedPackageManifest.motionJsonFile.Replace('/', Path.DirectorySeparatorChar)));
                if (!IsPathInsideDirectory(motionPath, packageDirectory))
                {
                    error = "manifest.json 中的 motion JSON 路径无效。";
                    return false;
                }

                if (!File.Exists(motionPath))
                {
                    error = "动作包中的 motion JSON 不存在：" + motionPath;
                    return false;
                }

                if (!string.IsNullOrWhiteSpace(selectedPackageManifest.motionJsonSha256) &&
                    !string.Equals(MotionPackageStorage.ComputeSha256(motionPath),
                        selectedPackageManifest.motionJsonSha256, StringComparison.OrdinalIgnoreCase))
                {
                    error = "动作包 motion JSON 校验失败，文件可能已被修改。";
                    return false;
                }

                sourcePackagePath = manifestPath;
            }
            else if (!selectedFileName.EndsWith(".motion.json", StringComparison.OrdinalIgnoreCase))
            {
                error = "请选择 manifest.json 或导出的 *.motion.json。";
                return false;
            }

            package = JsonUtility.FromJson<RecordedCoachMotionPackage>(
                File.ReadAllText(motionPath, Encoding.UTF8));
            if (!TryValidateImportedMotionPackage(package, out error))
            {
                package = null;
                return false;
            }

            if (selectedManifest)
            {
                if (!string.Equals(package.actionId, selectedPackageManifest.actionId,
                        StringComparison.Ordinal))
                {
                    package = null;
                    error = "motion JSON 的 actionId 与 manifest.json 不一致。";
                    return false;
                }
            }

            sourceMotionPath = motionPath;
            sourcePackagePath = sourcePackagePath ?? motionPath;
            return true;
        }
        catch (Exception exception)
        {
            package = null;
            error = "动作包 JSON 读取失败：" + exception.Message;
            return false;
        }
    }

    private static bool TryValidateImportedMotionPackage(RecordedCoachMotionPackage package,
        out string error)
    {
        error = null;
        if (package == null || package.formatVersion != 1)
        {
            error = "不支持的 motion JSON 格式。";
            return false;
        }

        if (package.segments == null || package.segments.Length == 0)
        {
            error = "motion JSON 中没有动作片段。";
            return false;
        }

        for (int segmentIndex = 0; segmentIndex < package.segments.Length; segmentIndex++)
        {
            RecordedCoachMotionSegment segment = package.segments[segmentIndex];
            if (segment == null || segment.frames == null || segment.frames.Length == 0)
            {
                error = "motion JSON 的第 " + (segmentIndex + 1) + " 个动作片段没有帧数据。";
                return false;
            }

            for (int frameIndex = 0; frameIndex < segment.frames.Length; frameIndex++)
            {
                RecordedCoachMotionFrame frame = segment.frames[frameIndex];
                if (frame == null || frame.muscles == null)
                {
                    error = "motion JSON 包含空姿态帧。";
                    return false;
                }
            }
        }

        return true;
    }

    private static bool PathsEqual(string left, string right)
    {
        if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right)) return false;
        try
        {
            return string.Equals(Path.GetFullPath(left.Trim()), Path.GetFullPath(right.Trim()),
                StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private static bool TryConvertMotionPackageToPoseRecording(
        RecordedCoachMotionPackage source, out RuntimePoseRecordingPackage result,
        out string error)
    {
        result = null;
        error = null;
        if (!TryValidateImportedMotionPackage(source, out error)) return false;

        var frames = new List<RuntimePoseFrame>();
        float timelineOffset = 0f;
        bool hasFrame = false;

        foreach (RecordedCoachMotionSegment segment in source.segments)
        {
            int repeatCount = Mathf.Max(1, segment.repeatCount);
            for (int repeat = 0; repeat < repeatCount; repeat++)
            {
                RecordedCoachMotionFrame[] segmentFrames = segment.frames;
                float segmentStart = segmentFrames[0].time;
                for (int index = 0; index < segmentFrames.Length; index++)
                {
                    // Adjacent exported segments share their boundary frame.
                    // Keep it once so the runtime timeline remains monotonic.
                    if (hasFrame && index == 0) continue;

                    RecordedCoachMotionFrame sourceFrame = segmentFrames[index];
                    float localTime = Mathf.Max(0f, sourceFrame.time - segmentStart);
                    var muscles = new float[HumanTrait.MuscleCount];
                    int muscleCount = Mathf.Min(muscles.Length, sourceFrame.muscles.Length);
                    Array.Copy(sourceFrame.muscles, muscles, muscleCount);

                    frames.Add(new RuntimePoseFrame
                    {
                        timeSeconds = timelineOffset + localTime,
                        bodyPosition = sourceFrame.bodyPosition,
                        bodyRotation = sourceFrame.bodyRotation,
                        muscles = muscles,
                        transforms = Array.Empty<RuntimeTransformPose>()
                    });
                    hasFrame = true;
                }

                float segmentEnd = Mathf.Max(0f,
                    segmentFrames[segmentFrames.Length - 1].time - segmentStart);
                timelineOffset += segmentEnd;
            }
        }

        if (frames.Count == 0)
        {
            error = "motion JSON 没有可回放的姿态帧。";
            return false;
        }

        result = new RuntimePoseRecordingPackage
        {
            actionId = source.actionId,
            displayName = source.displayName,
            trajectoryName = string.IsNullOrWhiteSpace(source.actionId)
                ? "imported_motion"
                : source.actionId,
            recordedAtUtc = DateTime.UtcNow.ToString("o"),
            avatarName = source.sourceClipName,
            sampleRate = Mathf.Max(1f, source.fps),
            durationSeconds = frames[frames.Count - 1].timeSeconds,
            frameCount = frames.Count,
            transformPaths = Array.Empty<string>(),
            frames = frames.ToArray()
        };
        return true;
    }

    private bool RegisterImportedMotionPackage(RuntimePoseRecordingPackage package,
        string sourcePackagePath, out string error)
    {
        error = null;
        if (package == null || package.frames == null || package.frames.Length == 0)
        {
            error = "无法登记空的动作包。";
            return false;
        }

        if (string.IsNullOrWhiteSpace(_metaDataPath))
            _metaDataPath = Path.Combine(Application.persistentDataPath, "TrajectoriesMeta");
        Directory.CreateDirectory(_metaDataPath);

        string importedName = GetUniqueTrajectoryName(BuildImportedTrajectoryName(package));
        package.trajectoryName = importedName;
        if (!RuntimePoseRecordingStorage.TrySave(package, out string storedPosePath, out error))
            return false;

        _lastImportedPoseRecordingPath = storedPosePath;
        string metadataPath = Path.Combine(_metaDataPath, importedName + ".json");
        var metadata = new TrajectoryMetadata
        {
            name = importedName,
            recordedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            duration = package.durationSeconds,
            poseRecordingFile = storedPosePath,
            trackerJsonFile = string.Empty,
            rawImuJsonFile = string.Empty,
            imported = true,
            importLabel = "导入",
            importSourcePath = sourcePackagePath ?? string.Empty,
            files = Array.Empty<string>()
        };

        try
        {
            File.WriteAllText(metadataPath, JsonUtility.ToJson(metadata, true),
                new UTF8Encoding(false));
        }
        catch (Exception exception)
        {
            error = "动作姿态已保存，但历史记录写入失败：" + exception.Message;
            return false;
        }

        RefreshTrajectoryList();
        return true;
    }

    private static string BuildImportedTrajectoryName(RecordedCoachMotionPackage package)
    {
        string actionId = MakeSafeFileName(package == null ? null : package.actionId);
        string displayName = MakeSafeFileName(package == null ? null : package.displayName);
        if (string.IsNullOrWhiteSpace(actionId)) actionId = "imported_motion";
        if (string.IsNullOrWhiteSpace(displayName)) displayName = actionId;
        return string.Equals(actionId, displayName, StringComparison.OrdinalIgnoreCase)
            ? actionId
            : actionId + "_" + displayName;
    }

    private static string BuildImportedTrajectoryName(RuntimePoseRecordingPackage package)
    {
        string actionId = MakeSafeFileName(package == null ? null : package.actionId);
        string displayName = MakeSafeFileName(package == null ? null : package.displayName);
        if (string.IsNullOrWhiteSpace(actionId)) actionId = "imported_motion";
        if (string.IsNullOrWhiteSpace(displayName)) displayName = actionId;
        return string.Equals(actionId, displayName, StringComparison.OrdinalIgnoreCase)
            ? actionId
            : actionId + "_" + displayName;
    }

    private static string GetImportedDisplayName(RecordedCoachMotionPackage package)
    {
        if (package == null) return "未命名动作";
        if (!string.IsNullOrWhiteSpace(package.displayName)) return package.displayName.Trim();
        return string.IsNullOrWhiteSpace(package.actionId) ? "未命名动作" : package.actionId.Trim();
    }

    /// <summary>
    /// Loads either a recorded .raw-imu.json file or a local exported motion
    /// package manifest whose rawImuJsonFile points at that file. The package
    /// is intentionally accepted outside MotionPackageStorage.RootDirectory so
    /// users can import a package copied from another computer or folder.
    /// </summary>
    private bool TryLoadRawImuImport(string selectedPath, out RawImuRecordingPackage package,
        out string sourceRawImuPath, out string sourcePackagePath, out string error)
    {
        package = null;
        sourceRawImuPath = null;
        sourcePackagePath = null;
        error = null;

        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            error = "JSON 动作包路径为空。";
            return false;
        }

        string fullSelectedPath;
        try
        {
            fullSelectedPath = Path.GetFullPath(selectedPath.Trim());
        }
        catch (System.Exception exception)
        {
            error = "JSON 动作包路径无效：" + exception.Message;
            return false;
        }

        if (RawImuRecordingStorage.TryLoad(fullSelectedPath, out package, out string rawError))
        {
            sourceRawImuPath = fullSelectedPath;
            sourcePackagePath = fullSelectedPath;
            return true;
        }

        // The exported action package uses manifest.json. Allow selecting the
        // motion JSON itself as a convenience by looking beside it as well.
        string selectedDirectory = Path.GetDirectoryName(fullSelectedPath);
        string manifestPath = string.Equals(Path.GetFileName(fullSelectedPath),
                MotionPackageStorage.ManifestFileName, System.StringComparison.OrdinalIgnoreCase)
            ? fullSelectedPath
            : Path.Combine(selectedDirectory ?? string.Empty, MotionPackageStorage.ManifestFileName);

        if (!File.Exists(manifestPath))
        {
            error = rawError ?? "所选文件不是有效的 Raw IMU JSON 动作包。";
            return false;
        }

        try
        {
            MotionPackageManifest manifest = JsonUtility.FromJson<MotionPackageManifest>(
                File.ReadAllText(manifestPath));
            if (manifest == null || string.IsNullOrWhiteSpace(manifest.rawImuJsonFile))
            {
                error = "动作包中没有可预览的 Raw IMU JSON。";
                return false;
            }

            string packageDirectory = Path.GetFullPath(Path.GetDirectoryName(manifestPath) ?? string.Empty);
            string rawPath = Path.GetFullPath(Path.Combine(packageDirectory,
                manifest.rawImuJsonFile.Replace('/', Path.DirectorySeparatorChar)));
            if (!IsPathInsideDirectory(rawPath, packageDirectory))
            {
                error = "动作包中的 Raw IMU 文件路径无效。";
                return false;
            }

            if (!RawImuRecordingStorage.TryLoad(rawPath, out package, out string packageRawError))
            {
                error = "动作包中的 Raw IMU JSON 无效：" + packageRawError;
                return false;
            }

            sourceRawImuPath = rawPath;
            sourcePackagePath = manifestPath;
            return true;
        }
        catch (System.Exception exception)
        {
            error = "动作包 JSON 读取失败：" + exception.Message;
            return false;
        }
    }

    private bool RegisterImportedRawImu(RawImuRecordingPackage package, string sourcePackagePath,
        out string error)
    {
        error = null;
        if (package == null)
        {
            error = "无法登记空的 Raw IMU 动作包。";
            return false;
        }

        if (string.IsNullOrWhiteSpace(_metaDataPath))
            _metaDataPath = Path.Combine(Application.persistentDataPath, "TrajectoriesMeta");
        Directory.CreateDirectory(_metaDataPath);

        string importedName = GetUniqueTrajectoryName(BuildImportedTrajectoryName(package));
        string originalTrajectoryName = package.trajectoryName;
        package.trajectoryName = importedName;
        if (!RawImuRecordingStorage.TrySave(package, out string storedRawImuPath, out error))
        {
            package.trajectoryName = originalTrajectoryName;
            return false;
        }

        _lastRawImuJsonPath = storedRawImuPath;
        string metadataPath = Path.Combine(_metaDataPath, importedName + ".json");
        var metadata = new TrajectoryMetadata
        {
            name = importedName,
            recordedAt = FormatImportedRecordedAt(package, sourcePackagePath),
            duration = Mathf.Max(package.durationSeconds,
                package.frames != null && package.frames.Length > 0
                    ? package.frames[package.frames.Length - 1].timeSeconds
                    : 0f),
            poseRecordingFile = string.Empty,
            trackerJsonFile = string.Empty,
            rawImuJsonFile = storedRawImuPath,
            imported = true,
            importLabel = "导入",
            importSourcePath = sourcePackagePath ?? string.Empty,
            files = System.Array.Empty<string>()
        };

        try
        {
            File.WriteAllText(metadataPath, JsonUtility.ToJson(metadata, true));
        }
        catch (System.Exception exception)
        {
            error = "导入动作已复制，但历史记录写入失败：" + exception.Message;
            return false;
        }

        RefreshTrajectoryList();
        return true;
    }

    private string BuildImportedTrajectoryName(RawImuRecordingPackage package)
    {
        string actionId = MakeSafeFileName(package.actionId);
        string displayName = MakeSafeFileName(package.displayName);
        string trajectoryName = MakeSafeFileName(package.trajectoryName);
        if (string.IsNullOrWhiteSpace(actionId)) actionId = "imported_motion";
        if (string.IsNullOrWhiteSpace(displayName)) displayName = trajectoryName;
        if (string.IsNullOrWhiteSpace(displayName)) displayName = actionId;

        return string.Equals(actionId, displayName, System.StringComparison.OrdinalIgnoreCase)
            ? actionId
            : actionId + "_" + displayName;
    }

    private static string GetImportedDisplayName(RawImuRecordingPackage package)
    {
        if (package == null) return "未命名动作";
        if (!string.IsNullOrWhiteSpace(package.displayName)) return package.displayName.Trim();
        if (!string.IsNullOrWhiteSpace(package.trajectoryName)) return package.trajectoryName.Trim();
        return string.IsNullOrWhiteSpace(package.actionId) ? "未命名动作" : package.actionId.Trim();
    }

    private static string FormatImportedRecordedAt(RawImuRecordingPackage package, string sourcePath)
    {
        if (package != null && !string.IsNullOrWhiteSpace(package.recordedAtUtc) &&
            DateTime.TryParse(package.recordedAtUtc, CultureInfo.InvariantCulture,
                DateTimeStyles.RoundtripKind, out DateTime recordedAt))
            return recordedAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss");

        try
        {
            return File.GetLastWriteTime(sourcePath ?? string.Empty).ToString("yyyy-MM-dd HH:mm:ss");
        }
        catch
        {
            return DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        }
    }

    private static string EscapeRichText(string value)
    {
        return (value ?? string.Empty).Replace("&", "&amp;").Replace("<", "&lt;")
            .Replace(">", "&gt;");
    }

    private static bool TrySelectLocalJsonFile(out string path)
    {
        path = null;
#if UNITY_EDITOR
        path = UnityEditor.EditorUtility.OpenFilePanel("选择 JSON 动作包", string.Empty, "json");
#elif UNITY_STANDALONE_WIN
        var dialog = new OpenDialogFile
        {
            structSize = Marshal.SizeOf(typeof(OpenDialogFile)),
            filter = "JSON files (*.json)\0*.json\0All files (*.*)\0*.*\0",
            filterIndex = 1,
            file = new string('\0', 4096),
            maxFile = 4096,
            fileTitle = new string('\0', 256),
            maxFileTitle = 256,
            initialDir = Application.persistentDataPath,
            title = "选择 JSON 动作包",
            defExt = "json"
        };
        if (DllOpenFileDialog.GetOpenFileName(dialog))
            path = (dialog.file ?? string.Empty).TrimEnd('\0').Trim();
#endif
        return !string.IsNullOrWhiteSpace(path);
    }

    private bool StartRawImuPlayback()
    {
        if (_isRecording)
        {
            ShowOperationStatus("Stop recording before starting Raw IMU playback.");
            return false;
        }

        if (_loadedRawImuRecording?.frames == null || _loadedRawImuRecording.frames.Length == 0)
        {
            ShowOperationStatus("Imported Raw IMU JSON has no frames.");
            return false;
        }

        if (rawImuTrackerPreview == null)
            rawImuTrackerPreview = FindObjectOfType<RawImuTrackerPreview>();
        if (rawImuTrackerPreview == null)
        {
            ShowOperationStatus("RawImuTrackerPreview is not assigned in the scene.");
            return false;
        }

        if (_isPlayingBack) EndPlayback();
        _currentPlaybackClip = null;
        _isPlayingBack = true;
        _isRawImuPlayback = true;
        _isRawImuPlaybackPaused = false;
        _rawImuPlaybackTime = 0f;
        _rawImuPlaybackFrameIndex = 0;
        rawImuTrackerPreview.ShowPlaybackFrame(_loadedRawImuRecording.frames[0]);
        ShowOperationStatus(
            $"Raw IMU playback started: {_loadedRawImuRecording.sensorIds.Length} sensors.");
        return true;
    }

    private void UpdateRawImuPlayback()
    {
        RawImuFrame[] frames = _loadedRawImuRecording?.frames;
        if (frames == null || frames.Length == 0)
        {
            EndRawImuPlayback("Raw IMU playback stopped: no frames.");
            return;
        }

        float duration = Mathf.Max(_loadedRawImuRecording.durationSeconds,
            frames[frames.Length - 1].timeSeconds);
        if (_isRawImuPlaybackPaused)
        {
            if (txtStatus != null)
                txtStatus.text = $"Raw IMU playback paused: {_rawImuPlaybackTime:F1}s / {duration:F1}s";
            return;
        }

        if (duration <= 0f)
        {
            rawImuTrackerPreview.ShowPlaybackFrame(frames[0]);
            _isRawImuPlaybackPaused = true;
            ShowOperationStatus("Raw IMU single-frame preview.");
            return;
        }

        _rawImuPlaybackTime += Time.unscaledDeltaTime;
        while (_rawImuPlaybackFrameIndex + 1 < frames.Length &&
               frames[_rawImuPlaybackFrameIndex + 1].timeSeconds <= _rawImuPlaybackTime)
            _rawImuPlaybackFrameIndex++;

        rawImuTrackerPreview.ShowPlaybackFrame(frames[_rawImuPlaybackFrameIndex]);
        if (_rawImuPlaybackTime >= duration)
        {
            rawImuTrackerPreview.ShowPlaybackFrame(frames[frames.Length - 1]);
            EndRawImuPlayback("Raw IMU playback complete; live preview restored.");
            return;
        }

        if (txtStatus != null)
            txtStatus.text =
                $"Raw IMU playback: {_rawImuPlaybackTime:F1}s / {duration:F1}s " +
                $"({_rawImuPlaybackFrameIndex + 1}/{frames.Length})";
    }

    private void EndRawImuPlayback(string message)
    {
        _isRawImuPlayback = false;
        _isRawImuPlaybackPaused = false;
        _isPlayingBack = false;
        _rawImuPlaybackTime = 0f;
        _rawImuPlaybackFrameIndex = 0;
        if (rawImuTrackerPreview != null) rawImuTrackerPreview.ReturnToLive();
        ShowOperationStatus(message);
    }

    public void OnImportTrackerJson()
    {
        string path = inputTrackerJsonPath == null ? null : inputTrackerJsonPath.text?.Trim();
        if (string.IsNullOrWhiteSpace(path) && !TrackerRecordingStorage.TryFindLatest(out path))
        {
            ShowOperationStatus("没有可导入的 Tracker JSON");
            return;
        }

        ImportTrackerJson(path);
    }

    public void OnPreviewImportedTrackerJson()
    {
        if (_loadedTrackerRecording == null)
        {
            OnImportTrackerJson();
            if (_loadedTrackerRecording == null) return;
        }

        StartTrackerPlayback();
    }

    public void OnImportAndPreviewTrackerJson()
    {
        string path = inputTrackerJsonPath == null ? null : inputTrackerJsonPath.text?.Trim();
        if (string.IsNullOrWhiteSpace(path) && !TrackerRecordingStorage.TryFindLatest(out path))
        {
            ShowOperationStatus("没有可导入的 Tracker JSON");
            return;
        }

        ImportAndPreviewTrackerJson(path);
    }

    private bool ImportAndPreviewTrackerJson(string path)
    {
        if (!ImportTrackerJson(path)) return false;
        return StartTrackerPlayback();
    }

    private bool ImportTrackerJson(string path)
    {
        if (!TrackerRecordingStorage.TryLoad(path, out _loadedTrackerRecording, out string error))
        {
            ShowOperationStatus(error);
            return false;
        }

        _lastTrackerJsonPath = Path.GetFullPath(path);
        if (inputTrackerJsonPath != null) inputTrackerJsonPath.text = _lastTrackerJsonPath;
        ShowOperationStatus(
            $"已导入：{_loadedTrackerRecording.frameCount} 帧 / {_loadedTrackerRecording.trackerNames.Length} 个 Tracker");
        Debug.Log($"[Tracker Recording] Imported {_lastTrackerJsonPath}", this);
        return true;
    }

    private bool StartTrackerPlayback()
    {
        if (_loadedTrackerRecording?.frames == null || _loadedTrackerRecording.frames.Length == 0) return false;

        int boundCount = CountPreviewBindings(_loadedTrackerRecording.trackerNames);
        if (boundCount == 0)
        {
            ShowOperationStatus("JSON 已导入，但没有绑定 Tracker 预览物体");
            return false;
        }

        if (_isPlayingBack) EndPlayback();
        _isRawImuPlayback = false;
        var receiver = FindObjectOfType<EVMC4U.ExternalReceiver>();
        if (receiver != null) receiver.Freeze = true;
        _currentPlaybackClip = null;
        _trackerPlaybackTime = 0f;
        _trackerPlaybackFrameIndex = 0;
        _isPlayingBack = true;
        ApplyTrackerFrame(_loadedTrackerRecording.frames[0]);
        ShowOperationStatus($"Tracker JSON 预览开始（已绑定 {boundCount} 个 Tracker）");
        return true;
    }

    private void UpdateTrackerPlayback()
    {
        TrackerRecordingFrame[] frames = _loadedTrackerRecording.frames;
        _trackerPlaybackTime += Time.deltaTime;
        float duration = Mathf.Max(_loadedTrackerRecording.durationSeconds, frames[frames.Length - 1].timeSeconds);
        if (_trackerPlaybackTime >= duration)
        {
            ApplyTrackerFrame(frames[frames.Length - 1]);
            EndPlayback();
            return;
        }

        while (_trackerPlaybackFrameIndex + 1 < frames.Length &&
               frames[_trackerPlaybackFrameIndex + 1].timeSeconds <= _trackerPlaybackTime)
            _trackerPlaybackFrameIndex++;

        ApplyTrackerFrame(frames[_trackerPlaybackFrameIndex]);
        txtStatus.text = $"Tracker 预览中... {_trackerPlaybackTime:F1}s / {duration:F1}s";
    }

    private void ApplyTrackerFrame(TrackerRecordingFrame frame)
    {
        if (frame?.trackers == null) return;

        foreach (TrackerPoseData pose in frame.trackers)
        {
            Transform target = FindPreviewTransform(pose.name);
            if (target == null) continue;
            target.localPosition = pose.position;
            target.localRotation = pose.rotation;
        }
    }

    private Transform FindPreviewTransform(string trackerName)
    {
        if (trackerPreviewBindings != null)
        {
            foreach (TrackerPreviewBinding binding in trackerPreviewBindings)
            {
                if (binding != null && binding.target != null &&
                    string.Equals(binding.trackerName, trackerName, System.StringComparison.Ordinal))
                    return binding.target;
            }
        }

        var deviceReceiver = FindObjectOfType<EVMC4U.DeviceReceiver>();
        if (deviceReceiver?.Serials == null || deviceReceiver.Transforms == null) return null;
        int count = Mathf.Min(deviceReceiver.Serials.Length, deviceReceiver.Transforms.Length);
        for (int index = 0; index < count; index++)
        {
            if (deviceReceiver.Transforms[index] != null &&
                string.Equals(deviceReceiver.Serials[index], trackerName, System.StringComparison.Ordinal))
                return deviceReceiver.Transforms[index];
        }

        return null;
    }

    private int CountPreviewBindings(string[] trackerNames)
    {
        if (trackerNames == null) return 0;

        int count = 0;
        foreach (string trackerName in trackerNames)
            if (FindPreviewTransform(trackerName) != null) count++;
        return count;
    }

    private static bool TryParseFloat(string value, out float result)
    {
        return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out result) ||
               float.TryParse(value, NumberStyles.Float, CultureInfo.CurrentCulture, out result);
    }

    private static string NormalizeSex(string label)
    {
        if (string.IsNullOrWhiteSpace(label)) return "unspecified";
        string normalized = label.Trim().ToLowerInvariant();
        if (normalized == "男" || normalized == "male" || normalized == "m") return "male";
        if (normalized == "女" || normalized == "female" || normalized == "f") return "female";
        if (normalized == "其他" || normalized == "other") return "other";
        return "unspecified";
    }

#if UNITY_EDITOR
    private static string ToAssetPath(string absolutePath)
    {
        string projectPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..")).Replace("\\", "/");
        string normalized = Path.GetFullPath(absolutePath).Replace("\\", "/");
        return normalized.StartsWith(projectPath + "/", System.StringComparison.OrdinalIgnoreCase)
            ? normalized.Substring(projectPath.Length + 1)
            : string.Empty;
    }
#endif

    private void EndPlayback()
    {
        bool wasRawImuPlayback = _isRawImuPlayback;
        _isPlayingBack = false;
        _isRawImuPlayback = false;
        _isRawImuPlaybackPaused = false;
        _currentPlaybackClip = null;
        _loadedPoseRecording = null;
        _posePlaybackHandler?.Dispose();
        _posePlaybackHandler = null;
        _posePlaybackPose = new HumanPose();
        _posePlaybackTime = 0f;
        _posePlaybackFrameIndex = 0;
        _trackerPlaybackTime = 0f;
        _trackerPlaybackFrameIndex = 0;
        _rawImuPlaybackTime = 0f;
        _rawImuPlaybackFrameIndex = 0;
        if (wasRawImuPlayback && rawImuTrackerPreview != null)
            rawImuTrackerPreview.ReturnToLive();
        var receiver = FindObjectOfType<EVMC4U.ExternalReceiver>();
        if (receiver != null) receiver.Freeze = false;
        if (txtStatus != null) txtStatus.text = "回放完成";
    }
}

[System.Serializable]
public sealed class TrackerPreviewBinding
{
    public string trackerName;
    public Transform target;
}
