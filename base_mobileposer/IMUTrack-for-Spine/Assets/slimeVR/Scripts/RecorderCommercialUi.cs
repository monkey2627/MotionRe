using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Presentation-only skin for the Recorder scene. It keeps every existing
/// control and UnityEvent intact, and arranges them around an unobstructed
/// central avatar preview.
/// </summary>
[DefaultExecutionOrder(-1000)]
[DisallowMultipleComponent]
public sealed class RecorderCommercialUi : MonoBehaviour
{
    private static readonly Color CanvasBackdrop = Hex("08111FFF");
    private static readonly Color Panel = Hex("101B2BEE");
    private static readonly Color PanelRaised = Hex("152337F4");
    private static readonly Color Field = Hex("17263AF8");
    private static readonly Color Stroke = Hex("29415FFF");
    private static readonly Color TextPrimary = Hex("F2F7FFFF");
    private static readonly Color TextSecondary = Hex("91A4BDFF");
    private static readonly Color Cyan = Hex("36D6E7FF");
    private static readonly Color Blue = Hex("397AF6FF");
    private static readonly Color Green = Hex("35D49AFF");
    private static readonly Color Red = Hex("F05264FF");
    private static readonly Color Amber = Hex("F5B84BFF");

    private RecordingController controller;
    private RectTransform panelRoot;
    private RectTransform canvasRoot;
    private RectTransform chromeRoot;
    private Image statusDot;
    private Image liveDot;
    private Text previewStateText;
    private Dropdown trackerDropdown;
    private Text trackerDetailsText;
    private RawImuTrackerPreview rawImuPreview;
    private RectTransform previewInputArea;
    private RecorderPreviewCameraController previewCameraController;
    private AvatarModelLibrary modelLibrary;
    private RectTransform modelButtonContainer;
    private Text modelLibraryStatusText;
    private Sprite roundedSprite;
    private Texture2D roundedTexture;
    private Font uiFont;
    private string lastStatus;
    private float nextHistoryStyleTime;
    private int lastStyledDropdownPopupId;
    private Button historyReconstructionButton;
    private RectTransform reconstructionOverlay;
    private RectTransform reconstructionHistoryContent;
    private RectTransform reconstructionSensorContent;
    private Text reconstructionSelectionTitle;
    private Text reconstructionWarningText;
    private Button reconstructionStartButton;
    private List<HistoryReconstructionCandidate> reconstructionCandidates;
    private HistoryReconstructionCandidate selectedReconstructionCandidate;
    private readonly HashSet<string> selectedReconstructionSensors = new HashSet<string>();
    private readonly Dictionary<string, Button> reconstructionSensorButtons =
        new Dictionary<string, Button>();

    private const float HistoryItemHeight = 86f;
    private const float HistoryItemSpacing = 12f;
    private const float HistoryTopInset = 112f;
    private const float PreviewCameraPitch = 18f;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void InstallInRecorderScene()
    {
        if (FindObjectOfType<RecorderCommercialUi>() != null) return;

        RecordingController recordingController = FindObjectOfType<RecordingController>();
        if (recordingController == null || recordingController.GetComponentInParent<Canvas>() == null)
            return;

        recordingController.gameObject.AddComponent<RecorderCommercialUi>();
    }

    private void Awake()
    {
        controller = GetComponent<RecordingController>();
        if (controller == null) controller = FindObjectOfType<RecordingController>();
        if (controller == null || GetComponentInParent<Canvas>() == null)
        {
            enabled = false;
            return;
        }

        panelRoot = controller.transform as RectTransform;
        canvasRoot = GetComponentInParent<Canvas>().transform as RectTransform;
        rawImuPreview = controller.rawImuTrackerPreview != null
            ? controller.rawImuTrackerPreview
            : FindObjectOfType<RawImuTrackerPreview>();
        modelLibrary = AvatarModelLibrary.EnsureInstalled(controller);
        if (modelLibrary != null)
        {
            modelLibrary.LibraryChanged -= RefreshModelButtons;
            modelLibrary.LibraryChanged += RefreshModelButtons;
            modelLibrary.StateChanged -= RefreshModelLibraryState;
            modelLibrary.StateChanged += RefreshModelLibraryState;
        }
        uiFont = controller.txtStatus != null ? controller.txtStatus.font : Resources.GetBuiltinResource<Font>("Arial.ttf");
        roundedSprite = CreateRuntimeRoundedSprite();
        controller.TrajectoryListRefreshed -= StyleTrajectoryItems;
        controller.TrajectoryListRefreshed += StyleTrajectoryItems;
        controller.TrajectoryExportSelectionChanged -= StyleTrajectoryItems;
        controller.TrajectoryExportSelectionChanged += StyleTrajectoryItems;
        controller.HistoryReconstructionStateChanged -= OnHistoryReconstructionStateChanged;
        controller.HistoryReconstructionStateChanged += OnHistoryReconstructionStateChanged;

        ConfigureCamera();
        ConfigureCanvasAndRoots();
        CreateChrome();
        ConfigurePreviewCameraControls();
        ArrangeExistingControls();
        ConfigureTrajectoryHistory();
        ConfigureRawImuTools();
        DisableCoachProfileControls();
        RefreshModelButtons();
        UpdateStatusVisual(true);
        UpdatePreviewActivity();
    }

    private void OnDestroy()
    {
        if (controller != null)
        {
            controller.TrajectoryListRefreshed -= StyleTrajectoryItems;
            controller.TrajectoryExportSelectionChanged -= StyleTrajectoryItems;
            controller.HistoryReconstructionStateChanged -= OnHistoryReconstructionStateChanged;
        }
        if (modelLibrary != null)
        {
            modelLibrary.LibraryChanged -= RefreshModelButtons;
            modelLibrary.StateChanged -= RefreshModelLibraryState;
        }

        if (roundedSprite != null)
            Destroy(roundedSprite);
        if (roundedTexture != null)
            Destroy(roundedTexture);
    }

    private Sprite CreateRuntimeRoundedSprite()
    {
        const int size = 16;
        const float radius = 5f;
        roundedTexture = new Texture2D(size, size, TextureFormat.RGBA32, false)
        {
            name = "RecorderRuntimeRoundedTexture",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
            hideFlags = HideFlags.DontSave
        };

        var pixels = new Color32[size * size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float cornerX = Mathf.Max(radius - x - 0.5f,
                    x + 0.5f - (size - radius));
                float cornerY = Mathf.Max(radius - y - 0.5f,
                    y + 0.5f - (size - radius));
                float outside = Mathf.Sqrt(Mathf.Max(0f, cornerX) * Mathf.Max(0f, cornerX) +
                                           Mathf.Max(0f, cornerY) * Mathf.Max(0f, cornerY));
                byte alpha = (byte)Mathf.RoundToInt(
                    Mathf.Clamp01(radius + 0.5f - outside) * 255f);
                pixels[y * size + x] = new Color32(255, 255, 255, alpha);
            }
        }

        roundedTexture.SetPixels32(pixels);
        roundedTexture.Apply(false, true);
        Sprite sprite = Sprite.Create(roundedTexture, new Rect(0f, 0f, size, size),
            new Vector2(0.5f, 0.5f), 100f, 0, SpriteMeshType.FullRect,
            new Vector4(radius, radius, radius, radius));
        sprite.name = "RecorderRuntimeRoundedSprite";
        sprite.hideFlags = HideFlags.DontSave;
        return sprite;
    }

    private void Start()
    {
        // Repeat after scene Awake callbacks so no later initialization can
        // overwrite the Recorder preview's presentation-only framing.
        ConfigureCamera();
        ConfigurePreviewCameraControls();
        StyleTrajectoryItems();
    }

    private void Update()
    {
        UpdateStatusVisual(false);
        UpdatePreviewActivity();
        StyleOpenDropdownPopups();

        if (Time.unscaledTime >= nextHistoryStyleTime)
        {
            nextHistoryStyleTime = Time.unscaledTime + 0.5f;
            StyleTrajectoryItems();
        }
    }

    private void ConfigureCamera()
    {
        Camera targetCamera = Camera.main != null ? Camera.main : FindObjectOfType<Camera>();
        if (targetCamera == null) return;
        targetCamera.clearFlags = CameraClearFlags.SolidColor;
        targetCamera.backgroundColor = CanvasBackdrop;

        // Keep the camera at the scene-authored position and use a modest,
        // deterministic downward pitch. Renderer bounds are not used here:
        // the VRM's mesh bounds are below its humanoid viewing center, which
        // previously caused LookAt(bounds.center) to push the avatar out of
        // the top of the preview. This changes only presentation framing.
        Vector3 euler = targetCamera.transform.eulerAngles;
        targetCamera.transform.rotation = Quaternion.Euler(PreviewCameraPitch, euler.y, 0f);
    }

    private void ConfigureCanvasAndRoots()
    {
        CanvasScaler scaler = canvasRoot.GetComponent<CanvasScaler>();
        if (scaler != null)
        {
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920f, 1080f);
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 0.5f;
        }

        Stretch(panelRoot, 0f, 0f, 0f, 0f);
        Image oldPanelImage = panelRoot.GetComponent<Image>();
        if (oldPanelImage != null)
        {
            oldPanelImage.color = Color.clear;
            oldPanelImage.raycastTarget = false;
        }

        chromeRoot = CreateRect("CommercialChrome", panelRoot);
        Stretch(chromeRoot, 0f, 0f, 0f, 0f);
        chromeRoot.SetAsFirstSibling();
    }

    private void CreateChrome()
    {
        CreatePanel("TopNavigation", chromeRoot,
            new Vector2(0f, 1f), new Vector2(1f, 1f), Vector2.zero,
            new Vector2(0f, 78f), new Vector2(0.5f, 1f), Hex("0B1524F7"), true);

        RectTransform leftCard = CreatePanel("SessionSetupCard", chromeRoot,
            new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(202f, -26f),
            new Vector2(364f, -132f), new Vector2(0.5f, 0.5f), Panel, true);
        AddShadow(leftCard.gameObject, new Color(0f, 0f, 0f, 0.42f), new Vector2(0f, -8f));
        CreatePanel("Accent", leftCard, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(3f, -30f), new Vector2(6f, 42f), new Vector2(0.5f, 0.5f), Cyan, false);

        RectTransform transport = CreatePanel("TransportDock", chromeRoot,
            new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(0f, 28f),
            new Vector2(1120f, 102f), new Vector2(0.5f, 0f), Hex("0E1929F4"), true);
        AddShadow(transport.gameObject, new Color(0f, 0f, 0f, 0.5f), new Vector2(0f, -7f));

        RectTransform statusPill = CreatePanel("StatusPill", chromeRoot,
            new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -56f),
            new Vector2(620f, 46f), new Vector2(0.5f, 0.5f), Hex("101D30E8"), true);
        Outline statusOutline = statusPill.gameObject.AddComponent<Outline>();
        statusOutline.effectColor = Hex("294563B0");
        statusOutline.effectDistance = new Vector2(1f, -1f);
        statusDot = CreatePanel("StatusIndicator", statusPill,
            new Vector2(0f, 0.5f), new Vector2(0f, 0.5f), new Vector2(25f, 0f),
            new Vector2(12f, 12f), new Vector2(0.5f, 0.5f), Green, true).GetComponent<Image>();

        CreateViewportFrame();

        CreateText("SessionHeading", chromeRoot, "录制设置", 22, FontStyle.Bold,
            TextPrimary, TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(40f, -118f), new Vector2(320f, 32f), new Vector2(0f, 0.5f));
        CreateText("SessionCaption", chromeRoot, "配置动作信息与教练档案", 13, FontStyle.Normal,
            TextSecondary, TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(40f, -146f), new Vector2(320f, 24f), new Vector2(0f, 0.5f));

        AddFieldLabel("动作 ID", -179f);
        AddFieldLabel("动作名称", -269f);
        CreateModelLibraryControls();
        CreateText("RawImuHeading", chromeRoot, "RAW IMU 数据工具", 16, FontStyle.Bold,
            TextPrimary, TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(40f, -674f), new Vector2(320f, 26f), new Vector2(0f, 0.5f));

        CreateText("PreviewLabel", chromeRoot, "实时人体预览", 18, FontStyle.Bold,
            TextPrimary, TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(420f, -116f), new Vector2(280f, 30f), new Vector2(0f, 0.5f));
        previewStateText = CreateText("PreviewHint", chromeRoot, "LIVE  ·  录制与回放动作将在此处实时呈现", 13,
            FontStyle.Normal, TextSecondary, TextAnchor.MiddleLeft,
            new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(420f, -143f),
            new Vector2(440f, 24f), new Vector2(0f, 0.5f));
        liveDot = CreatePanel("LiveDot", chromeRoot, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(403f, -116f), new Vector2(9f, 9f), new Vector2(0.5f, 0.5f), Green, true)
            .GetComponent<Image>();

        Button resetViewButton = CreateButton("BtnResetPreviewView", chromeRoot,
            "视角复原", PanelRaised);
        PositionAndStyleButton(resetViewButton, new Vector2(1f, 1f),
            new Vector2(-435f, -126f), new Vector2(112f, 36f), PanelRaised,
            "视角复原", false, 13);
        resetViewButton.onClick.AddListener(ResetPreviewView);

        CreateText("TransportHint", chromeRoot, "CAPTURE CONTROLS", 11, FontStyle.Bold,
            Hex("647A96FF"), TextAnchor.MiddleCenter, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
            new Vector2(0f, 114f), new Vector2(220f, 20f), new Vector2(0.5f, 0.5f));
    }

    private void CreateViewportFrame()
    {
        const float left = 392f;
        const float right = 370f;
        const float top = 91f;
        const float bottom = 151f;
        Color line = Hex("2D4963A8");

        previewInputArea = CreateRect("PreviewCameraInputArea", chromeRoot);
        Stretch(previewInputArea, left, top, right, bottom);
        previewInputArea.SetAsFirstSibling();

        CreatePanel("ViewportTop", chromeRoot, new Vector2(0f, 1f), new Vector2(1f, 1f),
            new Vector2((left - right) * 0.5f, -top), new Vector2(-(left + right), 1f),
            new Vector2(0.5f, 0.5f), line, false);
        CreatePanel("ViewportBottom", chromeRoot, new Vector2(0f, 0f), new Vector2(1f, 0f),
            new Vector2((left - right) * 0.5f, bottom), new Vector2(-(left + right), 1f),
            new Vector2(0.5f, 0.5f), line, false);
        CreatePanel("ViewportLeft", chromeRoot, new Vector2(0f, 0f), new Vector2(0f, 1f),
            new Vector2(left, (bottom - top) * 0.5f), new Vector2(1f, -(top + bottom)),
            new Vector2(0.5f, 0.5f), line, false);
        CreatePanel("ViewportRight", chromeRoot, new Vector2(1f, 0f), new Vector2(1f, 1f),
            new Vector2(-right, (bottom - top) * 0.5f), new Vector2(1f, -(top + bottom)),
            new Vector2(0.5f, 0.5f), line, false);
    }

    private void ConfigurePreviewCameraControls(string modelId = null)
    {
        Camera targetCamera = Camera.main != null ? Camera.main : FindObjectOfType<Camera>();
        if (targetCamera == null || previewInputArea == null) return;

        if (previewCameraController == null)
            previewCameraController = targetCamera.GetComponent<RecorderPreviewCameraController>() ??
                                      targetCamera.gameObject.AddComponent<RecorderPreviewCameraController>();

        Animator previewAnimator = controller.playbackAnimator;
        if (previewAnimator == null && controller.poseRecorder != null)
            previewAnimator = controller.poseRecorder.AvatarAnimator;
        previewCameraController.Initialize(targetCamera, previewInputArea,
            canvasRoot.GetComponent<Canvas>(), previewAnimator == null ? null : previewAnimator.transform,
            string.IsNullOrWhiteSpace(modelId) ? modelLibrary?.CurrentModelId : modelId);
    }

    public void RebindPreviewAnimator(Animator animator, string modelId)
    {
        if (controller != null) controller.playbackAnimator = animator;
        ConfigurePreviewCameraControls(modelId);
    }

    private void ResetPreviewView()
    {
        previewCameraController?.ResetViewSmoothly();
    }

    private void AddFieldLabel(string label, float y)
    {
        CreateText("Label_" + label, chromeRoot, label, 13, FontStyle.Normal,
            TextSecondary, TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(46f, y), new Vector2(310f, 22f), new Vector2(0f, 0.5f));
    }

    private void ArrangeExistingControls()
    {
        StyleTitle(controller.txtStatus == null ? null : FindDeep(panelRoot, "TxtTitle")?.GetComponent<Text>());
        Position(controller.txtStatus?.rectTransform, new Vector2(0.5f, 1f), new Vector2(0f, -56f),
            new Vector2(540f, 42f), new Vector2(0.5f, 0.5f));
        if (controller.txtStatus != null)
        {
            controller.txtStatus.fontSize = 18;
            controller.txtStatus.fontStyle = FontStyle.Normal;
            controller.txtStatus.color = TextPrimary;
            controller.txtStatus.alignment = TextAnchor.MiddleCenter;
            controller.txtStatus.raycastTarget = false;
        }

        Position(controller.txtFileName?.rectTransform, new Vector2(0.5f, 0f), new Vector2(0f, 149f),
            new Vector2(850f, 30f), new Vector2(0.5f, 0.5f));
        if (controller.txtFileName != null)
        {
            controller.txtFileName.fontSize = 15;
            controller.txtFileName.color = TextSecondary;
            controller.txtFileName.alignment = TextAnchor.MiddleCenter;
            controller.txtFileName.raycastTarget = false;
        }

        EnsureActionIdField();
        PositionAndStyleField(controller.inputActionId, -216f);
        PositionAndStyleField(controller.inputActionName, -306f);

        PositionAndStyleButton(controller.btnCalibrate, new Vector2(0.5f, 0f), new Vector2(-210f, 54f),
            new Vector2(156f, 54f), PanelRaised, "校准设备", false);
        historyReconstructionButton = CreateButton("BtnHistoryReconstruction", panelRoot,
            "录制历史重制", Hex("8B35F2FF"));
        PositionAndStyleButton(historyReconstructionButton, new Vector2(0.5f, 0f),
            new Vector2(-405f, 54f), new Vector2(178f, 54f), Hex("8B35F2FF"),
            "录制历史重制", true, 15);
        historyReconstructionButton.onClick.AddListener(OpenHistoryReconstruction);
        PositionAndStyleButton(controller.btnRecord, new Vector2(0.5f, 0f), new Vector2(0f, 54f),
            new Vector2(190f, 58f), Red, null, true);
        if (controller.btnPlayback != null) controller.btnPlayback.gameObject.SetActive(false);

        Button edit = FindDeep(panelRoot, "BtnEditKeyframes")?.GetComponent<Button>();
        if (edit != null) edit.gameObject.SetActive(false);
        Button export = FindDeep(panelRoot, "BtnGetJson")?.GetComponent<Button>();
        PositionAndStyleButton(export, new Vector2(0.5f, 0f), new Vector2(220f, 54f),
            new Vector2(178f, 54f), Blue, "导出所选动作包", true);

        Button returnButton = FindDeep(panelRoot, "BtnReturn")?.GetComponent<Button>();
        PositionAndStyleButton(returnButton, new Vector2(1f, 1f), new Vector2(-104f, -39f),
            new Vector2(168f, 42f), Hex("1B2A3FFF"), "完成并返回", false);
    }

    private void StyleTitle(Text title)
    {
        if (title == null) return;
        title.text = "SPINEFLOW  ·  动作录制工作台";
        title.fontSize = 27;
        title.fontStyle = FontStyle.Bold;
        title.color = TextPrimary;
        title.alignment = TextAnchor.MiddleLeft;
        title.raycastTarget = false;
        Position(title.rectTransform, new Vector2(0f, 1f), new Vector2(34f, -39f),
            new Vector2(520f, 48f), new Vector2(0f, 0.5f));
    }

    private void PositionAndStyleField(InputField input, float y)
    {
        if (input == null) return;
        input.transform.SetParent(panelRoot, false);
        Position(input.transform as RectTransform, new Vector2(0f, 1f), new Vector2(202f, y),
            new Vector2(312f, 46f), new Vector2(0.5f, 0.5f));
        StyleFieldGraphic(input.GetComponent<Image>());
        input.caretColor = Cyan;
        input.selectionColor = Hex("397AF666");
        if (input.textComponent != null)
        {
            input.textComponent.color = TextPrimary;
            input.textComponent.fontSize = 17;
        }
        Text placeholder = input.placeholder as Text;
        if (placeholder == null)
        {
            placeholder = CreateText("Placeholder", input.transform, ResolvePlaceholder(input), 16,
                FontStyle.Italic, Hex("617690FF"), TextAnchor.MiddleLeft,
                Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero,
                new Vector2(0.5f, 0.5f));
            Stretch(placeholder.rectTransform, 10f, 0f, 10f, 0f);
            input.placeholder = placeholder;
        }

        if (placeholder != null)
        {
            placeholder.color = Hex("617690FF");
            placeholder.fontSize = 16;
        }
    }

    private void CreateModelLibraryControls()
    {
        CreateText("ModelLibraryHeading", chromeRoot, "人物模型", 16, FontStyle.Bold,
            TextPrimary, TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(40f, -356f), new Vector2(320f, 26f), new Vector2(0f, 0.5f));
        CreateText("ModelLibraryCaption", chromeRoot, "选择模型，或从本地导入 VRM", 12,
            FontStyle.Normal, TextSecondary, TextAnchor.MiddleLeft,
            new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(40f, -381f),
            new Vector2(320f, 22f), new Vector2(0f, 0.5f));

        RectTransform viewport = CreatePanel("ModelLibraryViewport", chromeRoot,
            new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(202f, -466f),
            new Vector2(312f, 142f), new Vector2(0.5f, 0.5f), Field, true);
        viewport.gameObject.AddComponent<RectMask2D>();
        Outline outline = viewport.gameObject.AddComponent<Outline>();
        outline.effectColor = Stroke;
        outline.effectDistance = new Vector2(1f, -1f);

        modelButtonContainer = CreateRect("ModelButtons", viewport);
        modelButtonContainer.anchorMin = new Vector2(0f, 1f);
        modelButtonContainer.anchorMax = new Vector2(1f, 1f);
        modelButtonContainer.pivot = new Vector2(0.5f, 1f);
        modelButtonContainer.anchoredPosition = new Vector2(0f, -7f);
        modelButtonContainer.sizeDelta = new Vector2(-14f, 0f);
        VerticalLayoutGroup layout = modelButtonContainer.gameObject.AddComponent<VerticalLayoutGroup>();
        layout.padding = new RectOffset(2, 2, 2, 2);
        layout.spacing = 7f;
        layout.childControlHeight = true;
        layout.childControlWidth = true;
        layout.childForceExpandHeight = false;
        layout.childForceExpandWidth = true;
        ContentSizeFitter fitter = modelButtonContainer.gameObject.AddComponent<ContentSizeFitter>();
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        ScrollRect scroll = viewport.gameObject.AddComponent<ScrollRect>();
        scroll.viewport = viewport;
        scroll.content = modelButtonContainer;
        scroll.horizontal = false;
        scroll.vertical = true;
        scroll.movementType = ScrollRect.MovementType.Clamped;
        scroll.scrollSensitivity = 24f;

        Button importButton = CreateButton("BtnImportAvatarModel", chromeRoot,
            "＋  导入本地 VRM 模型", Blue);
        PositionAndStyleButton(importButton, new Vector2(0f, 1f), new Vector2(202f, -563f),
            new Vector2(312f, 42f), Blue, "＋  导入本地 VRM 模型", true, 14);
        importButton.onClick.AddListener(() => modelLibrary?.ImportLocalModel());

        modelLibraryStatusText = CreateText("ModelLibraryStatus", chromeRoot,
            "正在扫描模型库...", 11, FontStyle.Normal, TextSecondary, TextAnchor.MiddleLeft,
            new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(46f, -598f),
            new Vector2(306f, 38f), new Vector2(0f, 0.5f));
        modelLibraryStatusText.horizontalOverflow = HorizontalWrapMode.Wrap;
        modelLibraryStatusText.verticalOverflow = VerticalWrapMode.Truncate;
    }

    private void RefreshModelButtons()
    {
        if (modelButtonContainer == null) return;
        foreach (Transform child in modelButtonContainer)
        {
            child.gameObject.SetActive(false);
            Destroy(child.gameObject);
        }

        if (modelLibrary == null || modelLibrary.Models.Count == 0)
        {
            Text empty = CreateText("EmptyModelLibrary", modelButtonContainer,
                "暂无模型，请点击下方按钮导入", 12, FontStyle.Normal, TextSecondary,
                TextAnchor.MiddleCenter, Vector2.zero, Vector2.one, Vector2.zero,
                new Vector2(0f, 42f), new Vector2(0.5f, 0.5f));
            empty.gameObject.AddComponent<LayoutElement>().preferredHeight = 42f;
            RefreshModelLibraryState();
            return;
        }

        for (int i = 0; i < modelLibrary.Models.Count; i++)
        {
            AvatarModelLibrary.ModelEntry entry = modelLibrary.Models[i];
            string entryId = entry.id;
            bool selected = string.Equals(entryId, modelLibrary.CurrentModelId,
                System.StringComparison.OrdinalIgnoreCase);
            Button button = CreateButton("Model_" + entryId, modelButtonContainer,
                (selected ? "✓  " : string.Empty) + entry.displayName,
                selected ? Hex("176A73FF") : PanelRaised);
            LayoutElement element = button.gameObject.AddComponent<LayoutElement>();
            element.preferredHeight = 38f;
            element.minHeight = 38f;
            button.onClick.AddListener(() => modelLibrary.SelectModel(entryId));
            button.interactable = !modelLibrary.IsLoading && !selected;
        }
        RefreshModelLibraryState();
    }

    private void RefreshModelLibraryState()
    {
        if (modelLibraryStatusText != null && modelLibrary != null)
            modelLibraryStatusText.text = modelLibrary.StatusMessage;
        if (modelButtonContainer != null) RefreshSelectedModelButtons();
    }

    private void RefreshSelectedModelButtons()
    {
        if (modelLibrary == null || modelButtonContainer == null) return;
        foreach (Transform child in modelButtonContainer)
        {
            Button button = child.GetComponent<Button>();
            if (button == null) continue;
            bool selected = child.name == "Model_" + modelLibrary.CurrentModelId;
            button.interactable = !modelLibrary.IsLoading && !selected;
            Image image = button.GetComponent<Image>();
            if (image != null) image.color = selected ? Hex("176A73FF") : PanelRaised;
            Text label = button.GetComponentInChildren<Text>(true);
            AvatarModelLibrary.ModelEntry entry = null;
            for (int i = 0; i < modelLibrary.Models.Count; i++)
                if (child.name == "Model_" + modelLibrary.Models[i].id) entry = modelLibrary.Models[i];
            if (label != null && entry != null)
                label.text = (selected ? "✓  " : string.Empty) + entry.displayName;
        }
    }

    private void DisableCoachProfileControls()
    {
        if (controller == null) return;
        if (controller.inputCoachHeightCm != null)
            controller.inputCoachHeightCm.gameObject.SetActive(false);
        if (controller.inputCoachWeightKg != null)
            controller.inputCoachWeightKg.gameObject.SetActive(false);
        if (controller.dropdownCoachSex != null)
            controller.dropdownCoachSex.gameObject.SetActive(false);
        controller.inputCoachHeightCm = null;
        controller.inputCoachWeightKg = null;
        controller.dropdownCoachSex = null;
    }

    private void EnsureActionIdField()
    {
        if (controller == null || controller.inputActionId != null) return;

        InputField template = controller.inputActionName ?? controller.inputCoachHeightCm ??
                              controller.inputCoachWeightKg;
        if (template == null) return;

        InputField actionId = Instantiate(template, panelRoot, false);
        actionId.name = "InputActionId";
        actionId.SetTextWithoutNotify(string.Empty);
        actionId.onValueChanged.RemoveAllListeners();
        actionId.onEndEdit.RemoveAllListeners();
        if (actionId.placeholder is Text placeholder)
            placeholder.text = "输入动作ID";
        controller.inputActionId = actionId;
    }

    private string ResolvePlaceholder(InputField input)
    {
        if (input == controller.inputActionId) return "输入动作ID";
        if (input == controller.inputActionName) return "输入动作名称";
        if (input == controller.inputCoachHeightCm) return "输入身高";
        if (input == controller.inputCoachWeightKg) return "输入体重";
        if (input == controller.inputRawImuJsonPath) return "选择或粘贴本地 JSON 动作包路径";
        return string.Empty;
    }

    private void PositionAndStyleDropdown(Dropdown dropdown, float y)
    {
        if (dropdown == null) return;
        ConfigureSexOptions(dropdown);
        StyleDropdown(dropdown, y);
    }

    private void StyleDropdown(Dropdown dropdown, float y)
    {
        if (dropdown == null) return;
        dropdown.transform.SetParent(panelRoot, false);
        Position(dropdown.transform as RectTransform, new Vector2(0f, 1f), new Vector2(202f, y),
            new Vector2(312f, 46f), new Vector2(0.5f, 0.5f));
        StyleFieldGraphic(dropdown.GetComponent<Image>());
        if (dropdown.captionText != null)
        {
            dropdown.captionText.color = TextPrimary;
            dropdown.captionText.fontSize = 17;
        }
        if (dropdown.itemText != null)
        {
            dropdown.itemText.color = TextPrimary;
            dropdown.itemText.fontSize = 16;
        }

        // Legacy Dropdown clones this inactive template at runtime. Its
        // default white item background used to be combined with the white
        // item text above, making 男/女/其他 appear blank. Style the template
        // itself so the cloned popup keeps the same dark theme.
        if (dropdown.template == null) return;

        Image templateImage = dropdown.template.GetComponent<Image>();
        if (templateImage != null)
        {
            templateImage.color = PanelRaised;
            templateImage.sprite = roundedSprite;
            templateImage.type = Image.Type.Sliced;
        }

        foreach (Image image in dropdown.template.GetComponentsInChildren<Image>(true))
        {
            if (image == null) continue;
            if (image.name == "Item Background")
            {
                image.color = Field;
                image.sprite = roundedSprite;
                image.type = Image.Type.Sliced;
            }
            else if (image.name == "Item Checkmark")
            {
                image.color = Cyan;
            }
        }

        foreach (Toggle toggle in dropdown.template.GetComponentsInChildren<Toggle>(true))
        {
            ColorBlock colors = toggle.colors;
            colors.normalColor = Field;
            colors.highlightedColor = Hex("23415BFF");
            colors.pressedColor = Hex("2D6F7AFF");
            colors.selectedColor = Hex("214B63FF");
            colors.disabledColor = Hex("33445A80");
            colors.colorMultiplier = 1f;
            colors.fadeDuration = 0.1f;
            toggle.colors = colors;
        }

        foreach (Text optionText in dropdown.template.GetComponentsInChildren<Text>(true))
        {
            if (optionText == null) continue;
            optionText.color = TextPrimary;
            optionText.fontSize = 16;
            optionText.alignment = TextAnchor.MiddleLeft;
            optionText.raycastTarget = false;
            RectTransform labelRect = optionText.rectTransform;
            labelRect.anchorMin = Vector2.zero;
            labelRect.anchorMax = Vector2.one;
            labelRect.offsetMin = new Vector2(34f, 0f);
            labelRect.offsetMax = new Vector2(-12f, 0f);
            labelRect.localScale = Vector3.one;
        }
    }

    private static void ConfigureSexOptions(Dropdown dropdown)
    {
        if (dropdown == null) return;

        string selected = string.Empty;
        if (dropdown.options != null && dropdown.options.Count > 0)
        {
            int oldIndex = Mathf.Clamp(dropdown.value, 0, dropdown.options.Count - 1);
            selected = dropdown.options[oldIndex].text == null
                ? string.Empty
                : dropdown.options[oldIndex].text.Trim();
        }

        // Rebuild the list instead of appending/removing entries. Legacy
        // Dropdown clones the template after this list is populated, and
        // leaving old OptionData instances in the list can make both cloned
        // labels point at the same visible value.
        dropdown.options = new List<Dropdown.OptionData>
        {
            new Dropdown.OptionData("男"),
            new Dropdown.OptionData("女")
        };
        dropdown.SetValueWithoutNotify(selected == "女" ? 1 : 0);
        dropdown.RefreshShownValue();
    }

    private void StyleOpenDropdownPopups()
    {
        if (canvasRoot == null) return;

        Transform popup = null;
        foreach (Transform candidate in canvasRoot.GetComponentsInChildren<Transform>(true))
        {
            if (candidate != null && candidate.gameObject.activeInHierarchy &&
                candidate.name.StartsWith("Dropdown List"))
            {
                popup = candidate;
                break;
            }
        }

        if (popup == null)
        {
            lastStyledDropdownPopupId = 0;
            return;
        }

        int popupId = popup.gameObject.GetInstanceID();
        if (popupId == lastStyledDropdownPopupId) return;
        lastStyledDropdownPopupId = popupId;

        Image popupImage = popup.GetComponent<Image>();
        if (popupImage != null)
        {
            popupImage.color = PanelRaised;
            popupImage.sprite = roundedSprite;
            popupImage.type = Image.Type.Sliced;
        }

        foreach (Image image in popup.GetComponentsInChildren<Image>(true))
        {
            if (image == null) continue;
            if (image.name == "Item Background")
            {
                image.color = Field;
                image.sprite = roundedSprite;
                image.type = Image.Type.Sliced;
            }
            else if (image.name == "Item Checkmark")
            {
                image.color = Cyan;
            }
        }

        foreach (Toggle toggle in popup.GetComponentsInChildren<Toggle>(true))
        {
            ColorBlock colors = toggle.colors;
            colors.normalColor = Field;
            colors.highlightedColor = Hex("23415BFF");
            colors.pressedColor = Hex("2D6F7AFF");
            colors.selectedColor = Hex("214B63FF");
            colors.disabledColor = Hex("33445A80");
            colors.colorMultiplier = 1f;
            colors.fadeDuration = 0.1f;
            toggle.colors = colors;
        }

        foreach (Text optionText in popup.GetComponentsInChildren<Text>(true))
        {
            if (optionText == null || optionText.name != "Item Label") continue;
            optionText.enabled = true;
            if (uiFont != null) optionText.font = uiFont;
            optionText.color = TextPrimary;
            optionText.fontSize = 16;
            optionText.alignment = TextAnchor.MiddleLeft;
            optionText.horizontalOverflow = HorizontalWrapMode.Overflow;
            optionText.verticalOverflow = VerticalWrapMode.Truncate;
            optionText.raycastTarget = false;

            RectTransform labelRect = optionText.rectTransform;
            labelRect.anchorMin = Vector2.zero;
            labelRect.anchorMax = Vector2.one;
            labelRect.offsetMin = new Vector2(34f, 0f);
            labelRect.offsetMax = new Vector2(-12f, 0f);
            labelRect.localScale = Vector3.one;
            labelRect.SetAsLastSibling();
        }

    }

    private void StyleFieldGraphic(Image image)
    {
        if (image == null) return;
        image.color = Field;
        image.sprite = roundedSprite;
        image.type = Image.Type.Sliced;
        Outline outline = image.GetComponent<Outline>() ?? image.gameObject.AddComponent<Outline>();
        outline.effectColor = Stroke;
        outline.effectDistance = new Vector2(1f, -1f);
    }

    private void ConfigureTrajectoryHistory()
    {
        if (controller.trajectoryListContainer == null) return;
        RectTransform content = controller.trajectoryListContainer as RectTransform;
        RectTransform historyPanel = content == null ? null : content.parent as RectTransform;
        if (content == null || historyPanel == null) return;

        historyPanel.anchorMin = new Vector2(1f, 0f);
        historyPanel.anchorMax = new Vector2(1f, 1f);
        historyPanel.pivot = new Vector2(0.5f, 0.5f);
        historyPanel.anchoredPosition = new Vector2(-190f, -26f);
        historyPanel.sizeDelta = new Vector2(340f, -132f);
        Image background = historyPanel.GetComponent<Image>();
        if (background != null)
        {
            background.color = Panel;
            background.sprite = roundedSprite;
            background.type = Image.Type.Sliced;
        }
        AddShadow(historyPanel.gameObject, new Color(0f, 0f, 0f, 0.42f), new Vector2(0f, -8f));

        foreach (HorizontalOrVerticalLayoutGroup group in historyPanel.GetComponents<HorizontalOrVerticalLayoutGroup>())
            group.enabled = false;

        ScrollRect scrollRect = historyPanel.GetComponent<ScrollRect>();
        if (scrollRect != null)
        {
            scrollRect.content = content;
            scrollRect.horizontal = false;
            scrollRect.vertical = true;
            scrollRect.scrollSensitivity = 28f;
        }

        content.anchorMin = new Vector2(0f, 1f);
        content.anchorMax = new Vector2(1f, 1f);
        content.pivot = new Vector2(0.5f, 1f);
        // ScrollRect may clamp a short content rect back to its top edge. Keep
        // the content at that edge and put the header clearance on the items
        // themselves so the first record cannot overlap the title/caption.
        content.anchoredPosition = Vector2.zero;
        content.sizeDelta = new Vector2(-28f, 0f);
        foreach (HorizontalOrVerticalLayoutGroup group in content.GetComponents<HorizontalOrVerticalLayoutGroup>())
            group.enabled = false;
        ContentSizeFitter fitter = content.GetComponent<ContentSizeFitter>();
        if (fitter != null) fitter.enabled = false;

        Text title = CreateText("HistoryTitle", historyPanel, "录制历史", 22, FontStyle.Bold,
            TextPrimary, TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(22f, -32f), new Vector2(280f, 32f), new Vector2(0f, 0.5f));
        title.gameObject.AddComponent<LayoutElement>().ignoreLayout = true;
        Text caption = CreateText("HistoryCaption", historyPanel,
            "点击记录选择导出；▶ 进入编辑与回放", 13,
            FontStyle.Normal, TextSecondary, TextAnchor.MiddleLeft,
            new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(22f, -62f),
            new Vector2(280f, 24f), new Vector2(0f, 0.5f));
        caption.gameObject.AddComponent<LayoutElement>().ignoreLayout = true;
        RectTransform divider = CreatePanel("HistoryDivider", historyPanel,
            new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -88f),
            new Vector2(-36f, 1f), new Vector2(0.5f, 0.5f), Stroke, false);
        divider.gameObject.AddComponent<LayoutElement>().ignoreLayout = true;
    }

    private void ConfigureRawImuTools()
    {
        InputField path = controller.inputRawImuJsonPath;
        if (path == null)
        {
            Transform found = FindDeep(canvasRoot, "InputRawImuJsonPath");
            if (found != null) path = found.GetComponent<InputField>();
        }
        PositionAndStyleField(path, -716f);
        if (path != null && path.placeholder is Text pathPlaceholder)
            pathPlaceholder.text = "选择或粘贴本地 JSON 动作包路径";

        Button importButton = FindDeep(canvasRoot, "BtnImportRawImuJson")?.GetComponent<Button>();
        Button playButton = FindDeep(canvasRoot, "BtnPlayRawImuJson")?.GetComponent<Button>();
        Button stopButton = FindDeep(canvasRoot, "BtnStopRawImuPreview")?.GetComponent<Button>();
        Button pauseButton = FindDeep(canvasRoot, "BtnPauseRawImuPreview")?.GetComponent<Button>();
        if (importButton != null) importButton.transform.SetParent(panelRoot, false);
        PositionAndStyleButton(importButton, new Vector2(0f, 1f), new Vector2(202f, -770f),
            new Vector2(312f, 42f), Field, "选择 / 导入", false);
        if (playButton != null) playButton.gameObject.SetActive(false);
        if (stopButton != null) stopButton.gameObject.SetActive(false);
        if (pauseButton != null) pauseButton.gameObject.SetActive(false);

        if (controller.dropdownCoachSex != null)
        {
            trackerDropdown = Instantiate(controller.dropdownCoachSex, panelRoot, false);
            trackerDropdown.name = "TrackerSelectorDropdown";
            trackerDropdown.onValueChanged.RemoveAllListeners();
            trackerDropdown.ClearOptions();
            trackerDropdown.AddOptions(new List<Dropdown.OptionData>
            {
                new Dropdown.OptionData("等待 Tracker 连接...")
            });
            trackerDropdown.SetValueWithoutNotify(0);
            trackerDropdown.interactable = false;
            trackerDropdown.RefreshShownValue();
            StyleDropdown(trackerDropdown, -824f);
        }

        RectTransform detailsCard = CreatePanel("CompactSensorDetails", panelRoot,
            new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(202f, -936f),
            new Vector2(312f, 178f), new Vector2(0.5f, 0.5f), Field, true);
        Outline detailsOutline = detailsCard.gameObject.AddComponent<Outline>();
        detailsOutline.effectColor = Stroke;
        detailsOutline.effectDistance = new Vector2(1f, -1f);
        RectTransform detailsAccent = CreatePanel("Accent", detailsCard,
            new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(3f, 0f),
            new Vector2(4f, -18f), new Vector2(0.5f, 0.5f), Cyan, false);
        detailsAccent.GetComponent<Image>().raycastTarget = false;
        trackerDetailsText = CreateText("SensorReadout", detailsCard,
            "<b>等待 Tracker 连接</b>\n\n连接后可在上方下拉框选择设备，\n这里将实时显示对应物理 IMU 数据。",
            11, FontStyle.Normal, Hex("C9D7E8FF"), TextAnchor.UpperLeft,
            Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero, new Vector2(0.5f, 0.5f));
        Stretch(trackerDetailsText.rectTransform, 15f, 12f, 11f, 10f);
        trackerDetailsText.lineSpacing = 1.12f;
        trackerDetailsText.horizontalOverflow = HorizontalWrapMode.Wrap;
        trackerDetailsText.verticalOverflow = VerticalWrapMode.Truncate;

        Transform oldContainer = FindDeep(canvasRoot, "RawImuPlaybackControls");
        if (oldContainer != null) oldContainer.gameObject.SetActive(false);
        Transform originalGrid = FindDeep(canvasRoot, "RawImuTrackerGrid");
        if (originalGrid != null) originalGrid.gameObject.SetActive(false);

        if (rawImuPreview != null)
            rawImuPreview.BindCompactView(trackerDropdown, trackerDetailsText);
    }

    private void StyleTrajectoryItems()
    {
        if (controller == null || controller.trajectoryListContainer == null) return;
        RectTransform content = controller.trajectoryListContainer as RectTransform;
        if (content == null) return;

        int index = 0;
        foreach (Transform child in controller.trajectoryListContainer)
        {
            if (child == null) continue;
            bool selectedForExport =
                controller.IsTrajectoryItemSelectedForExport(child.gameObject);
            RectTransform rect = child as RectTransform;
            if (rect != null)
            {
                // The list is deliberately laid out here instead of relying
                // on the prefab's centered anchors or a runtime layout group.
                // This keeps the first newly recorded item below the history
                // header and makes every later item deterministic.
                rect.anchorMin = new Vector2(0f, 1f);
                rect.anchorMax = new Vector2(1f, 1f);
                rect.pivot = new Vector2(0.5f, 1f);
                rect.anchoredPosition = new Vector2(0f,
                    -HistoryTopInset - index * (HistoryItemHeight + HistoryItemSpacing));
                rect.sizeDelta = new Vector2(0f, HistoryItemHeight);
                rect.localScale = Vector3.one;
            }

            Image card = child.GetComponent<Image>();
            if (card != null)
            {
                card.color = selectedForExport ? Hex("174153F8") : PanelRaised;
                card.sprite = roundedSprite;
                card.type = Image.Type.Sliced;
                Outline outline = child.GetComponent<Outline>() ?? child.gameObject.AddComponent<Outline>();
                outline.effectColor = selectedForExport ? Cyan : Hex("29415FA0");
                outline.effectDistance = selectedForExport
                    ? new Vector2(2f, -2f)
                    : new Vector2(1f, -1f);
            }

            Text[] texts = child.GetComponentsInChildren<Text>(true);
            if (texts.Length > 0)
            {
                texts[0].fontSize = 16;
                texts[0].fontStyle = FontStyle.Bold;
                texts[0].color = selectedForExport ? Cyan : TextPrimary;
                texts[0].alignment = TextAnchor.MiddleLeft;
                Position(texts[0].rectTransform, new Vector2(0f, 1f), new Vector2(14f, -24f),
                    new Vector2(205f, 28f), new Vector2(0f, 0.5f));
            }
            if (texts.Length > 1)
            {
                texts[1].fontSize = 12;
                texts[1].fontStyle = FontStyle.Normal;
                texts[1].color = TextSecondary;
                texts[1].alignment = TextAnchor.MiddleLeft;
                Position(texts[1].rectTransform, new Vector2(0f, 1f), new Vector2(14f, -56f),
                    new Vector2(205f, 24f), new Vector2(0f, 0.5f));
            }

            var childButtons = new List<Button>();
            foreach (Button button in child.GetComponentsInChildren<Button>(true))
                if (button.gameObject != child.gameObject) childButtons.Add(button);
            if (childButtons.Count > 0)
                PositionAndStyleButton(childButtons[0], new Vector2(1f, 0.5f), new Vector2(-67f, 0f),
                    new Vector2(44f, 44f), Hex("1C806FFF"), "▶", false, 18);
            if (childButtons.Count > 1)
                PositionAndStyleButton(childButtons[1], new Vector2(1f, 0.5f), new Vector2(-24f, 0f),
                    new Vector2(34f, 34f), Hex("512A36FF"), "×", false, 20);

            index++;
        }

        float contentHeight = index <= 0
            ? HistoryTopInset
            : HistoryTopInset + index * HistoryItemHeight + (index - 1) * HistoryItemSpacing;
        content.sizeDelta = new Vector2(-28f, contentHeight);
    }

    private void OpenHistoryReconstruction()
    {
        if (!controller.TryPrepareHistoryReconstruction(
                out List<HistoryReconstructionCandidate> candidates, out string error))
        {
            controller.ShowOperationStatus(error);
            return;
        }

        CloseHistoryReconstructionOverlay();
        reconstructionCandidates = candidates;
        BuildHistoryReconstructionOverlay();
        SelectReconstructionCandidate(candidates[0]);
    }

    private void BuildHistoryReconstructionOverlay()
    {
        reconstructionOverlay = CreatePanel("HistoryReconstructionOverlay", canvasRoot,
            Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero,
            new Vector2(0.5f, 0.5f), Hex("020713E8"), false);
        Stretch(reconstructionOverlay, 0f, 0f, 0f, 0f);
        reconstructionOverlay.GetComponent<Image>().raycastTarget = true;
        reconstructionOverlay.SetAsLastSibling();

        RectTransform dialog = CreatePanel("Dialog", reconstructionOverlay,
            new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), Vector2.zero,
            new Vector2(1040f, 720f), new Vector2(0.5f, 0.5f), Hex("101B2BFF"), true);
        AddShadow(dialog.gameObject, new Color(0f, 0f, 0f, 0.7f), new Vector2(0f, -10f));
        Outline outline = dialog.gameObject.AddComponent<Outline>();
        outline.effectColor = Hex("8B35F2CC");
        outline.effectDistance = new Vector2(2f, -2f);

        CreateText("Title", dialog, "录制历史重制", 25, FontStyle.Bold, TextPrimary,
            TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(34f, -40f), new Vector2(500f, 40f), new Vector2(0f, 0.5f));
        CreateText("Description", dialog,
            "选择一条原记录及参与重构的 Tracker。原记录保持只读，结果将作为新历史项保存。",
            13, FontStyle.Normal, TextSecondary, TextAnchor.MiddleLeft,
            new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(34f, -76f),
            new Vector2(850f, 28f), new Vector2(0f, 0.5f));
        Button close = CreateButton("Close", dialog, "×", Hex("512A36FF"));
        PositionAndStyleButton(close, new Vector2(1f, 1f), new Vector2(-30f, -34f),
            new Vector2(42f, 42f), Hex("512A36FF"), "×", false, 22);
        close.onClick.AddListener(CloseHistoryReconstructionOverlay);

        CreateText("HistoryHeader", dialog, "① 选择原录制历史", 16, FontStyle.Bold,
            TextPrimary, TextAnchor.MiddleLeft, new Vector2(0f, 1f), new Vector2(0f, 1f),
            new Vector2(34f, -122f), new Vector2(430f, 30f), new Vector2(0f, 0.5f));
        CreateText("SensorsHeader", dialog, "② 选择用于重构的 Tracker 部位", 16,
            FontStyle.Bold, TextPrimary, TextAnchor.MiddleLeft,
            new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(506f, -122f),
            new Vector2(470f, 30f), new Vector2(0f, 0.5f));

        reconstructionHistoryContent = CreateScrollableContent("HistoryScroll", dialog,
            new Vector2(34f, -395f), new Vector2(430f, 500f));
        reconstructionSensorContent = CreateScrollableContent("SensorScroll", dialog,
            new Vector2(506f, -370f), new Vector2(500f, 450f));

        for (int index = 0; index < reconstructionCandidates.Count; index++)
        {
            HistoryReconstructionCandidate candidate = reconstructionCandidates[index];
            Button item = CreateButton("History_" + index, reconstructionHistoryContent,
                candidate.name + "\n" + candidate.recordedAt, Field);
            PositionAndStyleButton(item, new Vector2(0f, 1f), new Vector2(207f, -34f - index * 72f),
                new Vector2(414f, 62f), Field, candidate.name + "\n" + candidate.recordedAt,
                false, 13);
            HistoryReconstructionCandidate captured = candidate;
            item.onClick.AddListener(() => SelectReconstructionCandidate(captured));
        }
        reconstructionHistoryContent.sizeDelta = new Vector2(0f,
            Mathf.Max(500f, reconstructionCandidates.Count * 72f + 10f));

        reconstructionSelectionTitle = CreateText("Selection", dialog, string.Empty, 14,
            FontStyle.Bold, Hex("FF4FD8FF"), TextAnchor.MiddleLeft,
            new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(506f, 98f),
            new Vector2(500f, 26f), new Vector2(0f, 0.5f));
        reconstructionWarningText = CreateText("Warning", dialog, string.Empty, 12,
            FontStyle.Normal, Amber, TextAnchor.UpperLeft,
            new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(506f, 70f),
            new Vector2(500f, 40f), new Vector2(0f, 1f));

        reconstructionStartButton = CreateButton("StartReconstruction", dialog,
            "开始重构", Hex("FF4FD8FF"));
        PositionAndStyleButton(reconstructionStartButton, new Vector2(1f, 0f),
            new Vector2(-126f, 34f), new Vector2(210f, 52f), Hex("FF4FD8FF"),
            "开始重构", true, 16);
        reconstructionStartButton.onClick.AddListener(StartSelectedHistoryReconstruction);
    }

    private RectTransform CreateScrollableContent(string name, Transform parent,
        Vector2 position, Vector2 size)
    {
        RectTransform viewport = CreatePanel(name, parent, new Vector2(0f, 1f),
            new Vector2(0f, 1f), position, size, new Vector2(0f, 0.5f), Hex("0B1524FF"), true);
        viewport.gameObject.AddComponent<RectMask2D>();
        var scroll = viewport.gameObject.AddComponent<ScrollRect>();
        scroll.horizontal = false;
        scroll.vertical = true;
        scroll.movementType = ScrollRect.MovementType.Clamped;
        scroll.viewport = viewport;
        RectTransform content = CreateRect("Content", viewport);
        content.anchorMin = new Vector2(0f, 1f);
        content.anchorMax = new Vector2(1f, 1f);
        content.pivot = new Vector2(0.5f, 1f);
        content.anchoredPosition = Vector2.zero;
        content.sizeDelta = new Vector2(0f, size.y);
        scroll.content = content;
        return content;
    }

    private void SelectReconstructionCandidate(HistoryReconstructionCandidate candidate)
    {
        selectedReconstructionCandidate = candidate;
        selectedReconstructionSensors.Clear();
        reconstructionSensorButtons.Clear();
        foreach (Transform child in reconstructionSensorContent)
            Destroy(child.gameObject);

        if (candidate?.sensors == null) return;
        for (int index = 0; index < candidate.sensors.Length; index++)
        {
            HistoryReconstructionSensor sensor = candidate.sensors[index];
            selectedReconstructionSensors.Add(sensor.sensorId);
            int column = index % 2;
            int row = index / 2;
            string role = FriendlyTrackerRole(sensor.trackerRole);
            Button item = CreateButton("Sensor_" + index, reconstructionSensorContent,
                "✓ " + role, Hex("6E35D8FF"));
            PositionAndStyleButton(item, new Vector2(0f, 1f),
                new Vector2(121f + column * 248f, -28f - row * 58f),
                new Vector2(232f, 48f), Hex("6E35D8FF"), "✓ " + role, false, 13);
            string capturedId = sensor.sensorId;
            item.onClick.AddListener(() => ToggleReconstructionSensor(capturedId));
            reconstructionSensorButtons[capturedId] = item;
        }
        reconstructionSensorContent.sizeDelta = new Vector2(0f,
            Mathf.Max(450f, Mathf.CeilToInt(candidate.sensors.Length / 2f) * 58f + 10f));
        reconstructionWarningText.text = candidate.rawFormatVersion < 2
            ? "提示：此旧记录仅含欧拉角，将自动转换为四元数，重构精度可能低于新录制。"
            : "使用原始融合四元数重构。可取消不希望参与求解的部位。";
        RefreshReconstructionSelection();
    }

    private void ToggleReconstructionSensor(string sensorId)
    {
        if (selectedReconstructionSensors.Contains(sensorId))
            selectedReconstructionSensors.Remove(sensorId);
        else
            selectedReconstructionSensors.Add(sensorId);
        RefreshReconstructionSelection();
    }

    private void RefreshReconstructionSelection()
    {
        if (selectedReconstructionCandidate == null) return;
        reconstructionSelectionTitle.text =
            $"将生成：{selectedReconstructionCandidate.name} · 重构 · " +
            $"{selectedReconstructionSensors.Count} Tracker";
        reconstructionStartButton.interactable = selectedReconstructionSensors.Count > 0;
        foreach (HistoryReconstructionSensor sensor in selectedReconstructionCandidate.sensors)
        {
            if (!reconstructionSensorButtons.TryGetValue(sensor.sensorId, out Button button)) continue;
            bool selected = selectedReconstructionSensors.Contains(sensor.sensorId);
            PositionAndStyleButton(button, new Vector2(0f, 1f),
                (button.transform as RectTransform).anchoredPosition,
                (button.transform as RectTransform).sizeDelta,
                selected ? Hex("6E35D8FF") : Hex("25344AFF"),
                (selected ? "✓ " : "○ ") + FriendlyTrackerRole(sensor.trackerRole), false, 13);
        }
    }

    private void StartSelectedHistoryReconstruction()
    {
        if (selectedReconstructionCandidate == null) return;
        if (!controller.StartHistoryReconstruction(selectedReconstructionCandidate.metadataPath,
                selectedReconstructionSensors, out string error))
        {
            controller.ShowOperationStatus(error);
            return;
        }
        CloseHistoryReconstructionOverlay();
    }

    private void OnHistoryReconstructionStateChanged(bool active)
    {
        if (historyReconstructionButton != null)
            historyReconstructionButton.interactable = !active;
        if (active) CloseHistoryReconstructionOverlay();
    }

    private void CloseHistoryReconstructionOverlay()
    {
        if (reconstructionOverlay != null) Destroy(reconstructionOverlay.gameObject);
        reconstructionOverlay = null;
        reconstructionHistoryContent = null;
        reconstructionSensorContent = null;
        reconstructionCandidates = null;
        selectedReconstructionCandidate = null;
        selectedReconstructionSensors.Clear();
        reconstructionSensorButtons.Clear();
    }

    private static string FriendlyTrackerRole(string role)
    {
        if (string.IsNullOrWhiteSpace(role) || role == "NONE") return "未分配部位";
        return role.Replace("LEFT", "左").Replace("RIGHT", "右")
            .Replace("UPPER", "上").Replace("LOWER", "下")
            .Replace("CHEST", "胸").Replace("ARM", "臂").Replace("HAND", "手")
            .Replace("LEG", "腿").Replace("FOOT", "脚").Replace("HEAD", "头")
            .Replace("WAIST", "腰").Replace("HIP", "髋").Replace("NECK", "颈")
            .Replace("_", "");
    }

    private void UpdateStatusVisual(bool force)
    {
        if (controller == null || controller.txtStatus == null || statusDot == null) return;
        string value = controller.txtStatus.text ?? string.Empty;
        if (!force && value == lastStatus) return;
        lastStatus = value;

        string normalized = value.ToLowerInvariant();
        if (value.Contains("录制") && (value.Contains("正在") || value.Contains("中")))
            statusDot.color = Red;
        else if (value.Contains("回放") || normalized.Contains("playback"))
            statusDot.color = Cyan;
        else if (value.Contains("失败") || value.Contains("错误") || normalized.Contains("error"))
            statusDot.color = Amber;
        else
            statusDot.color = Green;
    }

    private void UpdatePreviewActivity()
    {
        if (controller == null || controller.txtStatus == null || liveDot == null) return;

        string value = controller.txtStatus.text ?? string.Empty;
        string normalized = value.ToLowerInvariant();
        bool recording = value.Contains("正在录制");
        bool playback = value.Contains("回放中") || normalized.Contains("playback");
        Color activityColor = recording ? Red : playback ? Cyan : Green;
        float pulse = recording || playback
            ? Mathf.Lerp(0.45f, 1f, 0.5f + 0.5f * Mathf.Sin(Time.unscaledTime * 5f))
            : 1f;
        activityColor.a = pulse;
        liveDot.color = activityColor;

        if (previewStateText == null) return;
        previewStateText.text = recording
            ? "REC  ·  正在采集  ·  左键旋转  ·  右键平移"
            : playback
                ? "PLAYBACK  ·  正在回放  ·  左键旋转  ·  右键平移"
                : "LIVE  ·  左键旋转视角  ·  右键平移视角";
    }

    private void PositionAndStyleButton(Button button, Vector2 anchor, Vector2 position,
        Vector2 size, Color baseColor, string label, bool emphasized, int fontSize = 17)
    {
        if (button == null) return;
        Position(button.transform as RectTransform, anchor, position, size, new Vector2(0.5f, 0.5f));
        Image image = button.GetComponent<Image>();
        if (image != null)
        {
            image.color = baseColor;
            image.sprite = roundedSprite;
            image.type = Image.Type.Sliced;
        }

        ColorBlock colors = button.colors;
        colors.normalColor = Color.white;
        colors.highlightedColor = Hex("FFFFFFFF");
        colors.pressedColor = Hex("B9C6D7FF");
        colors.selectedColor = Color.white;
        colors.disabledColor = Hex("63708380");
        colors.colorMultiplier = 1f;
        colors.fadeDuration = 0.12f;
        button.colors = colors;

        Text text = button.GetComponentInChildren<Text>(true);
        if (text == null)
        {
            text = CreateText("Label", button.transform, label ?? string.Empty, fontSize,
                emphasized ? FontStyle.Bold : FontStyle.Normal, TextPrimary,
                TextAnchor.MiddleCenter, Vector2.zero, Vector2.one, Vector2.zero,
                Vector2.zero, new Vector2(0.5f, 0.5f));
            Stretch(text.rectTransform, 4f, 2f, 4f, 2f);
        }
        if (text != null)
        {
            if (label != null) text.text = label;
            text.fontSize = fontSize;
            text.fontStyle = emphasized ? FontStyle.Bold : FontStyle.Normal;
            text.color = TextPrimary;
            text.alignment = TextAnchor.MiddleCenter;
            text.raycastTarget = false;
        }
        if (emphasized) AddShadow(button.gameObject, new Color(baseColor.r, baseColor.g, baseColor.b, 0.35f),
            new Vector2(0f, -4f));
    }

    private Button CreateButton(string name, Transform parent, string label, Color color)
    {
        RectTransform rect = CreatePanel(name, parent, new Vector2(0.5f, 0.5f),
            new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(120f, 44f),
            new Vector2(0.5f, 0.5f), color, true);
        Button button = rect.gameObject.AddComponent<Button>();
        button.targetGraphic = rect.GetComponent<Image>();
        button.targetGraphic.raycastTarget = true;
        Text text = CreateText("Label", rect, label, 16, FontStyle.Normal, TextPrimary,
            TextAnchor.MiddleCenter, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero,
            new Vector2(0.5f, 0.5f));
        Stretch(text.rectTransform, 4f, 2f, 4f, 2f);
        PositionAndStyleButton(button, new Vector2(0.5f, 0.5f), Vector2.zero,
            new Vector2(120f, 44f), color, label, false);
        return button;
    }

    private RectTransform CreatePanel(string name, Transform parent, Vector2 anchorMin,
        Vector2 anchorMax, Vector2 position, Vector2 size, Vector2 pivot, Color color, bool rounded)
    {
        RectTransform rect = CreateRect(name, parent);
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        rect.pivot = pivot;
        Image image = rect.gameObject.AddComponent<Image>();
        image.color = color;
        image.raycastTarget = false;
        if (rounded)
        {
            image.sprite = roundedSprite;
            image.type = Image.Type.Sliced;
        }
        return rect;
    }

    private Text CreateText(string name, Transform parent, string value, int size,
        FontStyle style, Color color, TextAnchor alignment, Vector2 anchorMin,
        Vector2 anchorMax, Vector2 position, Vector2 rectSize, Vector2 pivot)
    {
        RectTransform rect = CreateRect(name, parent);
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = position;
        rect.sizeDelta = rectSize;
        rect.pivot = pivot;
        Text text = rect.gameObject.AddComponent<Text>();
        text.font = uiFont;
        text.fontSize = size;
        text.fontStyle = style;
        text.color = color;
        text.alignment = alignment;
        text.raycastTarget = false;
        text.supportRichText = true;
        text.text = value;
        return text;
    }

    private static RectTransform CreateRect(string name, Transform parent)
    {
        var gameObject = new GameObject(name, typeof(RectTransform));
        gameObject.layer = 5;
        RectTransform rect = gameObject.GetComponent<RectTransform>();
        rect.SetParent(parent, false);
        rect.localScale = Vector3.one;
        return rect;
    }

    private static void Position(RectTransform rect, Vector2 anchor, Vector2 position,
        Vector2 size, Vector2 pivot)
    {
        if (rect == null) return;
        rect.anchorMin = anchor;
        rect.anchorMax = anchor;
        rect.pivot = pivot;
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        rect.localScale = Vector3.one;
    }

    private static void Stretch(RectTransform rect, float left, float top, float right, float bottom)
    {
        if (rect == null) return;
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.offsetMin = new Vector2(left, bottom);
        rect.offsetMax = new Vector2(-right, -top);
        rect.localScale = Vector3.one;
    }

    private static void AddShadow(GameObject target, Color color, Vector2 distance)
    {
        Shadow shadow = target.GetComponent<Shadow>() ?? target.AddComponent<Shadow>();
        shadow.effectColor = color;
        shadow.effectDistance = distance;
        shadow.useGraphicAlpha = true;
    }

    private static Transform FindDeep(Transform root, string objectName)
    {
        if (root == null) return null;
        foreach (Transform child in root.GetComponentsInChildren<Transform>(true))
            if (child.name == objectName) return child;
        return null;
    }

    private static Color Hex(string value)
    {
        return ColorUtility.TryParseHtmlString("#" + value, out Color color) ? color : Color.white;
    }
}
