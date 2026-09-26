using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Assets.Library.WitUnitySdk.Utils;
using UnityEngine;

/// <summary>
/// Runtime model library for the Recorder scene. Every model is stored as
/// modelLibraryPath/&lt;model-name&gt;/&lt;model-name&gt;.vrm and loaded through the
/// existing EVMC4U/UniVRM runtime importer.
/// </summary>
[DefaultExecutionOrder(-1050)]
[DisallowMultipleComponent]
public sealed class AvatarModelLibrary : MonoBehaviour
{
    [Serializable]
    public sealed class ModelEntry
    {
        public string id;
        public string displayName;
        public string folderPath;
        public string vrmPath;
    }

    [SerializeField] private string modelLibraryPath = @"D:\研一\luzhi\model";
    [SerializeField, Min(5f)] private float loadTimeoutSeconds = 30f;

    private readonly List<ModelEntry> models = new List<ModelEntry>();
    private readonly Dictionary<string, CachedAvatar> cachedAvatars =
        new Dictionary<string, CachedAvatar>(StringComparer.OrdinalIgnoreCase);

    private sealed class CachedAvatar
    {
        public GameObject model;
        public GameObject container;

        public GameObject ActivationRoot => container != null ? container : model;
    }
    private RecordingController controller;
    private PoseFbxRecorder poseRecorder;
    private AutoGroundRootFromAvatar groundController;
    private EVMC4U.ExternalReceiver receiver;
    private Transform groundRoot;
    private ModelEntry pendingModel;
    private Coroutine loadWatchdog;
    private string currentModelId;
    private string statusMessage = "正在扫描模型库...";
    private bool isLoading;
    private bool initialized;

    public IReadOnlyList<ModelEntry> Models => models;
    public string CurrentModelId => currentModelId;
    public string StatusMessage => statusMessage;
    public string ModelLibraryPath => modelLibraryPath;
    public bool IsLoading => isLoading;

    public event Action LibraryChanged;
    public event Action StateChanged;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void InstallInRecorderScene()
    {
        RecordingController found = FindObjectOfType<RecordingController>();
        if (found != null) EnsureInstalled(found);
    }

    public static AvatarModelLibrary EnsureInstalled(RecordingController recordingController)
    {
        if (recordingController == null) return null;
        PoseFbxRecorder recorder = recordingController.poseRecorder != null
            ? recordingController.poseRecorder
            : FindObjectOfType<PoseFbxRecorder>();
        GameObject host = recorder != null ? recorder.gameObject : recordingController.gameObject;
        AvatarModelLibrary library = host.GetComponent<AvatarModelLibrary>();
        if (library == null) library = host.AddComponent<AvatarModelLibrary>();
        library.Initialize(recordingController, recorder);
        return library;
    }

    private void Start()
    {
        if (!initialized)
            Initialize(FindObjectOfType<RecordingController>(), FindObjectOfType<PoseFbxRecorder>());
    }

    private void OnDestroy()
    {
        if (receiver != null) receiver.AfterAutoLoadAction -= HandleModelLoaded;
    }

    private void Initialize(RecordingController recordingController, PoseFbxRecorder recorder)
    {
        if (recordingController != null) controller = recordingController;
        if (recorder != null) poseRecorder = recorder;
        if (poseRecorder == null) poseRecorder = FindObjectOfType<PoseFbxRecorder>();
        if (controller == null) controller = FindObjectOfType<RecordingController>();
        if (groundRoot == null) groundRoot = poseRecorder != null ? poseRecorder.transform : transform;
        if (groundController == null && groundRoot != null)
            groundController = groundRoot.GetComponent<AutoGroundRootFromAvatar>();

        EVMC4U.ExternalReceiver foundReceiver = FindObjectOfType<EVMC4U.ExternalReceiver>();
        if (receiver != foundReceiver)
        {
            if (receiver != null) receiver.AfterAutoLoadAction -= HandleModelLoaded;
            receiver = foundReceiver;
            if (receiver != null)
            {
                receiver.AfterAutoLoadAction -= HandleModelLoaded;
                receiver.AfterAutoLoadAction += HandleModelLoaded;
                receiver.enableAutoLoadVRM = false;
            }
        }

        if (!initialized)
        {
            initialized = true;
            RefreshLibrary();
            MatchInitialSceneModel();
        }
    }

    public void RefreshLibrary()
    {
        models.Clear();
        try
        {
            Directory.CreateDirectory(modelLibraryPath);
            string[] directories = Directory.GetDirectories(modelLibraryPath);
            Array.Sort(directories, StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < directories.Length; i++)
            {
                string[] vrmFiles = Directory.GetFiles(directories[i], "*.vrm",
                    SearchOption.TopDirectoryOnly);
                if (vrmFiles.Length == 0) continue;
                Array.Sort(vrmFiles, StringComparer.OrdinalIgnoreCase);
                string folderName = Path.GetFileName(directories[i]);
                models.Add(new ModelEntry
                {
                    id = folderName,
                    displayName = FormatDisplayName(folderName),
                    folderPath = directories[i],
                    vrmPath = vrmFiles[0]
                });
            }

            statusMessage = models.Count == 0
                ? "模型库为空，请导入本地 VRM 模型"
                : $"已发现 {models.Count} 个人物模型";
        }
        catch (Exception exception)
        {
            statusMessage = "模型库读取失败：" + exception.Message;
        }

        LibraryChanged?.Invoke();
        StateChanged?.Invoke();
    }

    public void SelectModel(string id)
    {
        if (isLoading)
        {
            SetStatus("人物模型正在加载，请稍候");
            return;
        }

        ModelEntry entry = models.Find(item => string.Equals(item.id, id,
            StringComparison.OrdinalIgnoreCase));
        if (entry == null || !File.Exists(entry.vrmPath))
        {
            SetStatus("找不到所选 VRM 文件，请刷新模型库");
            return;
        }

        if (string.Equals(currentModelId, entry.id, StringComparison.OrdinalIgnoreCase))
        {
            SetStatus("当前已使用模型：" + entry.displayName);
            return;
        }

        if (controller != null && !controller.TryPrepareForAvatarSwitch(out string prepareError))
        {
            SetStatus(prepareError);
            return;
        }

        if (receiver == null)
        {
            SetStatus("未找到 ExternalReceiver，无法驱动人物模型");
            return;
        }

        pendingModel = entry;
        isLoading = true;
        SetStatus("正在加载人物模型：" + entry.displayName);
        try
        {
            CacheAndDeactivateCurrentAvatar();

            if (TryActivateCachedAvatar(entry))
            {
                return;
            }

            receiver.LoadVRM(entry.vrmPath);
            if (loadWatchdog != null) StopCoroutine(loadWatchdog);
            loadWatchdog = StartCoroutine(WatchLoadTimeout(entry));
        }
        catch (Exception exception)
        {
            isLoading = false;
            pendingModel = null;
            SetStatus("模型加载失败：" + exception.Message);
        }
    }

    public void ImportLocalModel()
    {
        if (isLoading)
        {
            SetStatus("人物模型正在加载，请稍候");
            return;
        }

        string sourcePath = SelectVrmFile();
        if (string.IsNullOrWhiteSpace(sourcePath)) return;
        if (!File.Exists(sourcePath) || !string.Equals(Path.GetExtension(sourcePath), ".vrm",
                StringComparison.OrdinalIgnoreCase))
        {
            SetStatus("请选择有效的 .vrm 人物模型文件");
            return;
        }

        try
        {
            string baseName = MakeSafeFolderName(Path.GetFileNameWithoutExtension(sourcePath));
            string folder = GetAvailableImportFolder(baseName);
            Directory.CreateDirectory(folder);
            string destination = Path.Combine(folder, Path.GetFileName(sourcePath));
            File.Copy(sourcePath, destination, false);

            string sourceProfile = Path.Combine(Path.GetDirectoryName(sourcePath) ?? string.Empty,
                Path.GetFileNameWithoutExtension(sourcePath) + ".profile.json");
            if (File.Exists(sourceProfile))
                File.Copy(sourceProfile, Path.Combine(folder, Path.GetFileName(sourceProfile)), false);

            RefreshLibrary();
            string importedId = Path.GetFileName(folder);
            SetStatus("模型已导入：" + FormatDisplayName(importedId));
            SelectModel(importedId);
        }
        catch (Exception exception)
        {
            SetStatus("模型导入失败：" + exception.Message);
        }
    }

    private void HandleModelLoaded(GameObject model)
    {
        if (!isLoading || model == null) return;
        if (loadWatchdog != null)
        {
            StopCoroutine(loadWatchdog);
            loadWatchdog = null;
        }

        CompleteModelActivation(model, receiver.LoadedModelParent);
    }

    private void CompleteModelActivation(GameObject model, GameObject container)
    {
        Animator animator = model.GetComponent<Animator>() ?? model.GetComponentInChildren<Animator>();
        if (animator == null || animator.avatar == null || !animator.isHuman)
        {
            isLoading = false;
            pendingModel = null;
            SetStatus("模型加载失败：VRM 不包含有效的 Humanoid Avatar");
            return;
        }

        if (container != null && groundRoot != null)
        {
            Transform containerTransform = container.transform;
            containerTransform.SetParent(groundRoot, false);
            containerTransform.localPosition = Vector3.zero;
            containerTransform.localRotation = Quaternion.identity;
            containerTransform.localScale = Vector3.one;
        }

        GameObject activationRoot = container != null ? container : model;
        if (!activationRoot.activeSelf) activationRoot.SetActive(true);
        model.transform.localPosition = Vector3.zero;
        model.transform.localRotation = Quaternion.identity;
        model.transform.localScale = Vector3.one;
        receiver.AttachExistingModel(model, container);

        // Rebind clears Unity Animator stream state accumulated before the model
        // was deactivated. Without this, a cached runtime Humanoid can continue
        // resolving bones against the previous avatar's internal pose stream.
        animator.Rebind();
        animator.Update(0f);

        if (poseRecorder != null && !poseRecorder.TrySetAvatarAnimator(animator, out string recorderError))
        {
            isLoading = false;
            pendingModel = null;
            SetStatus(recorderError);
            return;
        }

        groundController?.RebindAvatar(model.transform, animator);
        controller?.RebindPlaybackAnimator(animator);
        HumanoidPoseStabilizer stabilizer = groundRoot == null
            ? null
            : groundRoot.GetComponent<HumanoidPoseStabilizer>();
        stabilizer?.Configure(animator);
        stabilizer?.ResetStabilizer();

        RecorderCommercialUi ui = FindObjectOfType<RecorderCommercialUi>();
        string loadedModelId = pendingModel == null ? model.name : pendingModel.id;
        ui?.RebindPreviewAnimator(animator, loadedModelId);

        cachedAvatars[loadedModelId] = new CachedAvatar
        {
            model = model,
            container = container
        };

        currentModelId = loadedModelId;
        string displayName = pendingModel == null ? model.name : pendingModel.displayName;
        pendingModel = null;
        isLoading = false;
        SetStatus("已切换人物模型：" + displayName);
    }

    private void CacheAndDeactivateCurrentAvatar()
    {
        if (receiver == null || receiver.Model == null) return;

        string id = currentModelId;
        if (string.IsNullOrWhiteSpace(id)) id = receiver.Model.name;
        var cached = new CachedAvatar
        {
            model = receiver.Model,
            container = receiver.LoadedModelParent
        };
        cachedAvatars[id] = cached;

        receiver.DetachModelWithoutDestroy();
        GameObject activationRoot = cached.ActivationRoot;
        if (activationRoot != null) activationRoot.SetActive(false);
    }

    private bool TryActivateCachedAvatar(ModelEntry entry)
    {
        if (!cachedAvatars.TryGetValue(entry.id, out CachedAvatar cached) ||
            cached == null || cached.model == null)
        {
            cachedAvatars.Remove(entry.id);
            return false;
        }

        if (loadWatchdog != null)
        {
            StopCoroutine(loadWatchdog);
            loadWatchdog = null;
        }

        CompleteModelActivation(cached.model, cached.container);
        return true;
    }

    private IEnumerator WatchLoadTimeout(ModelEntry requested)
    {
        yield return new WaitForSecondsRealtime(loadTimeoutSeconds);
        if (!isLoading || pendingModel != requested) yield break;
        isLoading = false;
        pendingModel = null;
        loadWatchdog = null;
        SetStatus("模型加载超时，请检查 VRM 文件是否完整");
    }

    private void MatchInitialSceneModel()
    {
        if (receiver == null || receiver.Model == null) return;
        string sceneName = receiver.Model.name;
        ModelEntry match = models.Find(item =>
            string.Equals(item.id, sceneName, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(Path.GetFileNameWithoutExtension(item.vrmPath), sceneName,
                StringComparison.OrdinalIgnoreCase));
        if (match != null)
        {
            currentModelId = match.id;
            cachedAvatars[match.id] = new CachedAvatar
            {
                model = receiver.Model,
                container = receiver.LoadedModelParent
            };
            statusMessage = "当前人物模型：" + match.displayName;
            StateChanged?.Invoke();
        }
    }

    private void SetStatus(string message)
    {
        statusMessage = message;
        controller?.ShowOperationStatus(message);
        StateChanged?.Invoke();
    }

    private string GetAvailableImportFolder(string baseName)
    {
        string candidate = Path.Combine(modelLibraryPath, baseName);
        if (!Directory.Exists(candidate)) return candidate;
        int suffix = 2;
        while (Directory.Exists(Path.Combine(modelLibraryPath, baseName + "_" + suffix))) suffix++;
        return Path.Combine(modelLibraryPath, baseName + "_" + suffix);
    }

    private static string MakeSafeFolderName(string value)
    {
        string result = string.IsNullOrWhiteSpace(value) ? "ImportedModel" : value.Trim();
        foreach (char invalid in Path.GetInvalidFileNameChars()) result = result.Replace(invalid, '_');
        return result;
    }

    private static string FormatDisplayName(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? "未命名模型" : value.Replace('_', ' ');
    }

    private string SelectVrmFile()
    {
#if UNITY_EDITOR
        return UnityEditor.EditorUtility.OpenFilePanel("选择 VRM 人物模型", modelLibraryPath, "vrm");
#elif UNITY_STANDALONE_WIN
        var dialog = new OpenDialogFile
        {
            structSize = Marshal.SizeOf(typeof(OpenDialogFile)),
            filter = "VRM 人物模型 (*.vrm)\0*.vrm\0所有文件 (*.*)\0*.*\0\0",
            file = new string(new char[2048]),
            maxFile = 2048,
            fileTitle = new string(new char[256]),
            maxFileTitle = 256,
            initialDir = Directory.Exists(modelLibraryPath) ? modelLibraryPath : Application.persistentDataPath,
            title = "选择 VRM 人物模型",
            defExt = "vrm"
        };
        return DllOpenFileDialog.GetOpenFileName(dialog)
            ? (dialog.file ?? string.Empty).TrimEnd('\0').Trim()
            : null;
#else
        SetStatus("当前平台暂不支持系统文件选择器");
        return null;
#endif
    }
}
