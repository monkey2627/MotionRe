#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using SpineFlow.MotionEditing;
using SpineFlow.MotionPackages;
using UnityEditor;
using UnityEngine;

/// <summary>
/// IMUTrack-local post processor:
/// recorded .anim -> one FBX -> keyframe ranges -> CoachMotionPackage JSON.
/// </summary>
public sealed class RecordedMotionPostProcessorWindow : EditorWindow
{
    private const int EditDefinitionFormatVersion = 1;
    private const double AutoSaveDelaySeconds = 0.75d;

    private static readonly string[] BodyPartValues =
    {
        "whole_body", "torso", "hips", "head_neck", "left_arm", "right_arm", "left_leg", "right_leg"
    };

    private static readonly string[] BodyPartLabels =
    {
        "Whole body", "Torso", "Hips", "Head / neck", "Left arm", "Right arm", "Left leg", "Right leg"
    };

    private readonly List<RecordedMotionKeyframeDefinition> boundaries =
        new List<RecordedMotionKeyframeDefinition>();

    private AnimationClip sourceClip;
    private GameObject sourceAvatar;
    private string actionId = "dead_bug_demo";
    private string displayName = "Dead Bug Demo";
    private int sampleFps = 30;
    private int setCount = 1;
    private int totalFrames;
    private Vector2 scroll;
    private string lastResult;
    private MessageType lastResultType = MessageType.Info;
    private bool editDefinitionDirty;
    private double nextAutoSaveAt;
    private string lastSavedDefinitionPath;
    private DateTime lastSavedAtLocal;
    private int currentFrame;
    private bool previewPlaying;
    private double lastPreviewTick;
    private double previewFrameAccumulator;
    private bool previewStartedAnimationMode;

    [MenuItem("IMUTrack/Process Latest Recording")]
    public static void Open()
    {
        var window = GetWindow<RecordedMotionPostProcessorWindow>("Recording Post Process");
        window.minSize = new Vector2(620f, 520f);
        window.TryUseRecorderAvatar();
        window.TryLoadLatestRecording(false);
        window.Show();
    }

    private void Update()
    {
        UpdatePreview();

        if (!editDefinitionDirty || sourceClip == null || EditorApplication.timeSinceStartup < nextAutoSaveAt)
            return;

        SaveEditDefinition(false);
    }

    private void OnDisable()
    {
        StopPreview();
        if (editDefinitionDirty && sourceClip != null) SaveEditDefinition(false);
    }

    private void OnGUI()
    {
        scroll = EditorGUILayout.BeginScrollView(scroll);

        EditorGUILayout.LabelField("Recorded motion", EditorStyles.boldLabel);
        using (new EditorGUILayout.HorizontalScope())
        {
            EditorGUI.BeginChangeCheck();
            AnimationClip selectedClip = (AnimationClip)EditorGUILayout.ObjectField("Source .anim", sourceClip,
                typeof(AnimationClip), false);
            if (EditorGUI.EndChangeCheck()) LoadClip(selectedClip);

            if (GUILayout.Button("Latest", GUILayout.Width(72f))) TryLoadLatestRecording(true);
            if (GUILayout.Button("Choose", GUILayout.Width(72f))) ChooseRecording();
        }

        using (new EditorGUILayout.HorizontalScope())
        {
            EditorGUI.BeginChangeCheck();
            GameObject selectedAvatar = (GameObject)EditorGUILayout.ObjectField("Recorded avatar", sourceAvatar,
                typeof(GameObject), true);
            if (EditorGUI.EndChangeCheck())
            {
                sourceAvatar = selectedAvatar;
                ApplyPreviewFrame();
            }
            if (GUILayout.Button("Use recorder", GUILayout.Width(150f))) TryUseRecorderAvatar();
        }

        EditorGUILayout.Space(8f);
        EditorGUILayout.LabelField("Action package", EditorStyles.boldLabel);
        EditorGUI.BeginChangeCheck();
        actionId = EditorGUILayout.TextField("actionId", actionId).Trim();
        displayName = EditorGUILayout.TextField("Display name", displayName);
        sampleFps = EditorGUILayout.IntSlider("JSON sample FPS", sampleFps, 15, 60);
        setCount = Mathf.Max(1, EditorGUILayout.IntField("Default set count", setCount));
        if (EditorGUI.EndChangeCheck()) MarkEditDefinitionDirty();

        EditorGUILayout.Space(8f);
        DrawPreviewControls();
        EditorGUILayout.Space(8f);
        DrawKeyframes();
        DrawEditDefinitionControls();

        EditorGUILayout.Space(10f);
        using (new EditorGUI.DisabledScope(sourceClip == null || sourceAvatar == null || boundaries.Count < 2))
        {
            if (GUILayout.Button("Export FBX + Motion JSON", GUILayout.Height(36f))) ExportPackage();
        }

        if (!string.IsNullOrEmpty(lastResult))
        {
            EditorGUILayout.Space(8f);
            EditorGUILayout.HelpBox(lastResult, lastResultType);
        }

        EditorGUILayout.EndScrollView();
    }

    private void DrawPreviewControls()
    {
        EditorGUILayout.LabelField("Timeline preview", EditorStyles.boldLabel);
        if (sourceClip == null)
        {
            EditorGUILayout.HelpBox("Load a recorded .anim file to preview its pose.", MessageType.Info);
            return;
        }

        EditorGUI.BeginChangeCheck();
        currentFrame = EditorGUILayout.IntSlider("Preview frame", currentFrame, 0, totalFrames);
        if (EditorGUI.EndChangeCheck())
        {
            StopPreview();
            ApplyPreviewFrame();
        }

        using (new EditorGUILayout.HorizontalScope())
        {
            using (new EditorGUI.DisabledScope(sourceAvatar == null))
            {
                if (GUILayout.Button(previewPlaying ? "Pause" : "Play", GUILayout.Width(82f)))
                {
                    if (previewPlaying) StopPreview();
                    else StartPreview();
                }

                if (GUILayout.Button("Previous", GUILayout.Width(82f)))
                {
                    StopPreview();
                    currentFrame = Mathf.Max(0, currentFrame - 1);
                    ApplyPreviewFrame();
                }

                if (GUILayout.Button("Next", GUILayout.Width(82f)))
                {
                    StopPreview();
                    currentFrame = Mathf.Min(totalFrames, currentFrame + 1);
                    ApplyPreviewFrame();
                }

            }

            using (new EditorGUI.DisabledScope(sourceClip == null || currentFrame <= 0 ||
                                               currentFrame >= totalFrames))
            {
                if (GUILayout.Button("Set current as keyframe", GUILayout.Width(160f)))
                    AddKeyframeAtFrame(currentFrame);
            }

            if (GUILayout.Button("Stop preview", GUILayout.Width(100f))) StopPreview();
            GUILayout.FlexibleSpace();
            GUILayout.Label($"{currentFrame} / {totalFrames}", EditorStyles.miniLabel);
        }

        if (sourceAvatar == null)
        {
            EditorGUILayout.HelpBox("Assign the recorded avatar above to preview poses.", MessageType.Warning);
        }
        else
        {
            EditorGUILayout.LabelField("Preview target", sourceAvatar.name, EditorStyles.miniLabel);
        }
    }

    private void StartPreview()
    {
        if (sourceClip == null || sourceAvatar == null) return;

        previewPlaying = true;
        lastPreviewTick = EditorApplication.timeSinceStartup;
        previewFrameAccumulator = 0d;
        ApplyPreviewFrame();
    }

    private void StopPreview()
    {
        previewPlaying = false;
        if (previewStartedAnimationMode && AnimationMode.InAnimationMode())
            AnimationMode.StopAnimationMode();
        previewStartedAnimationMode = false;
        Repaint();
    }

    private void UpdatePreview()
    {
        if (!previewPlaying || sourceClip == null || sourceAvatar == null) return;

        double now = EditorApplication.timeSinceStartup;
        double elapsed = Mathf.Clamp((float)(now - lastPreviewTick), 0f, 0.25f);
        lastPreviewTick = now;
        previewFrameAccumulator += elapsed * Mathf.Max(1f, sourceClip.frameRate);
        int frameAdvance = Mathf.FloorToInt((float)previewFrameAccumulator);
        previewFrameAccumulator -= frameAdvance;
        if (frameAdvance <= 0) return;
        currentFrame += frameAdvance;
        if (currentFrame > totalFrames) currentFrame = 0;
        ApplyPreviewFrame();
        Repaint();
    }

    private void ApplyPreviewFrame()
    {
        if (sourceClip == null || sourceAvatar == null) return;

        try
        {
            if (!AnimationMode.InAnimationMode())
            {
                AnimationMode.StartAnimationMode();
                previewStartedAnimationMode = true;
            }

            float frameRate = Mathf.Max(1f, sourceClip.frameRate);
            float time = Mathf.Clamp(currentFrame / frameRate, 0f, sourceClip.length);
            AnimationMode.BeginSampling();
            try
            {
                AnimationMode.SampleAnimationClip(sourceAvatar, sourceClip, time);
            }
            finally
            {
                AnimationMode.EndSampling();
            }
        }
        catch (Exception exception)
        {
            StopPreview();
            SetResult("Could not sample the recorded pose: " + exception.Message, MessageType.Warning);
        }
    }

    private void DrawKeyframes()
    {
        EditorGUILayout.LabelField("Keyframe boundaries", EditorStyles.boldLabel);
        EditorGUILayout.LabelField(
            "First and last frames are fixed. Each row names the segment from this frame to the next boundary.",
            EditorStyles.wordWrappedMiniLabel);

        if (sourceClip == null)
        {
            EditorGUILayout.HelpBox("Load the latest recording first.", MessageType.Info);
            return;
        }

        for (int i = 0; i < boundaries.Count; i++)
        {
            RecordedMotionKeyframeDefinition boundary = boundaries[i];
            bool endpoint = i == 0 || i == boundaries.Count - 1;

            using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
            {
                EditorGUI.BeginChangeCheck();
                using (new EditorGUILayout.HorizontalScope())
                {
                    GUILayout.Label(i == 0 ? "Start" : i == boundaries.Count - 1 ? "End" : $"Keyframe {i}",
                        EditorStyles.boldLabel, GUILayout.Width(88f));

                    using (new EditorGUI.DisabledScope(endpoint))
                    {
                        boundary.frame = EditorGUILayout.IntField("Frame", boundary.frame);
                    }

                    using (new EditorGUI.DisabledScope(endpoint))
                    {
                        if (GUILayout.Button("Remove", GUILayout.Width(70f)))
                        {
                            EditorGUI.EndChangeCheck();
                            boundaries[i - 1].followingSegmentDurationSeconds = Mathf.Max(0.1f,
                                boundaries[i - 1].followingSegmentDurationSeconds +
                                boundary.followingSegmentDurationSeconds);
                            boundaries.RemoveAt(i);
                            NormalizeBoundaries();
                            MarkEditDefinitionDirty();
                            i--;
                            continue;
                        }
                    }
                }

                boundary.name = EditorGUILayout.TextField("Keyframe name", boundary.name);
                int bodyPartMask = BodyPartsToMask(boundary.bodyParts);
                int updatedBodyPartMask = EditorGUILayout.MaskField("Body parts", bodyPartMask, BodyPartLabels);
                if (updatedBodyPartMask != bodyPartMask)
                    boundary.bodyParts = MaskToBodyParts(updatedBodyPartMask);
                boundary.holdSeconds = Mathf.Max(0f,
                    EditorGUILayout.FloatField("Hold seconds", boundary.holdSeconds));

                if (i < boundaries.Count - 1)
                {
                    boundary.followingSegmentName =
                        EditorGUILayout.TextField("Following segment", boundary.followingSegmentName);
                    boundary.followingSegmentDurationSeconds = Mathf.Max(0.1f,
                        EditorGUILayout.FloatField("Segment duration", boundary.followingSegmentDurationSeconds));
                }
                else
                {
                    EditorGUILayout.LabelField("This is the final boundary; it has no following segment.",
                        EditorStyles.miniLabel);
                }

                if (EditorGUI.EndChangeCheck()) MarkEditDefinitionDirty();
            }
        }

        NormalizeBoundaries();
        using (new EditorGUI.DisabledScope(totalFrames < 2))
        {
            if (GUILayout.Button("Add keyframe", GUILayout.Width(130f))) AddKeyframe();
        }
        EditorGUILayout.LabelField($"Recording length: {sourceClip.length:F3}s / frames 0-{totalFrames}",
            EditorStyles.miniLabel);
    }

    private void DrawEditDefinitionControls()
    {
        if (sourceClip == null) return;

        EditorGUILayout.Space(8f);
        EditorGUILayout.LabelField("Edit definition", EditorStyles.boldLabel);
        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Save now", GUILayout.Width(100f))) SaveEditDefinition(true);

            using (new EditorGUI.DisabledScope(!File.Exists(GetEditDefinitionPath(sourceClip))))
            {
                if (GUILayout.Button("Reload saved", GUILayout.Width(110f)))
                {
                    bool reload = !editDefinitionDirty || EditorUtility.DisplayDialog(
                        "Reload keyframe edit definition",
                        "Discard unsaved changes and reload the saved keyframe definition?",
                        "Reload", "Cancel");
                    if (reload) LoadEditDefinition(true);
                }
            }

            GUILayout.FlexibleSpace();
            GUILayout.Label(editDefinitionDirty ? "Unsaved changes" : "Saved", EditorStyles.miniLabel);
        }

        string definitionPath = GetEditDefinitionPath(sourceClip);
        EditorGUILayout.LabelField("File", ToProjectRelativePath(definitionPath), EditorStyles.miniLabel);
        if (!editDefinitionDirty && lastSavedAtLocal != default(DateTime))
            EditorGUILayout.LabelField("Last saved", lastSavedAtLocal.ToString("yyyy-MM-dd HH:mm:ss"),
                EditorStyles.miniLabel);
    }

    private void ChooseRecording()
    {
        string selected = EditorUtility.OpenFilePanel("Choose recorded AnimationClip", GetRecordingsDirectory(), "anim");
        if (string.IsNullOrEmpty(selected)) return;

        string assetPath = RecordedFbxPackageExporter.ToAssetPath(selected);
        if (string.IsNullOrEmpty(assetPath))
        {
            SetResult("The .anim file must be inside this IMUTrack project's Assets folder.", MessageType.Error);
            return;
        }

        LoadClip(AssetDatabase.LoadAssetAtPath<AnimationClip>(assetPath));
    }

    private void TryLoadLatestRecording(bool reportFailure)
    {
        string directory = GetRecordingsDirectory();
        if (!Directory.Exists(directory))
        {
            if (reportFailure) SetResult("Assets/Recordings does not exist yet.", MessageType.Warning);
            return;
        }

        FileInfo latest = new DirectoryInfo(directory).GetFiles("*.anim", SearchOption.AllDirectories)
            .OrderByDescending(file => file.LastWriteTimeUtc)
            .FirstOrDefault();
        if (latest == null)
        {
            if (reportFailure) SetResult("No .anim recording was found in Assets/Recordings.", MessageType.Warning);
            return;
        }

        string assetPath = RecordedFbxPackageExporter.ToAssetPath(latest.FullName);
        LoadClip(AssetDatabase.LoadAssetAtPath<AnimationClip>(assetPath));
    }

    private void LoadClip(AnimationClip clip)
    {
        StopPreview();
        if (editDefinitionDirty && sourceClip != null) SaveEditDefinition(false);

        sourceClip = clip;
        boundaries.Clear();
        editDefinitionDirty = false;
        lastSavedDefinitionPath = null;
        lastSavedAtLocal = default(DateTime);
        currentFrame = 0;
        if (sourceClip == null)
        {
            totalFrames = 0;
            return;
        }

        sampleFps = Mathf.Clamp(Mathf.RoundToInt(sourceClip.frameRate), 15, 60);
        totalFrames = Mathf.Max(1, Mathf.RoundToInt(sourceClip.length * sourceClip.frameRate));
        if (!LoadEditDefinition(false))
        {
            CreateDefaultEditDefinition();
            MarkEditDefinitionDirty();
            SetResult($"Loaded {AssetDatabase.GetAssetPath(sourceClip)} with a new keyframe definition.",
                MessageType.Info);
        }
    }

    private void TryUseRecorderAvatar()
    {
        PoseFbxRecorder recorder = FindObjectOfType<PoseFbxRecorder>();
        if (recorder != null && recorder.AvatarAnimator != null)
        {
            sourceAvatar = recorder.AvatarAnimator.gameObject;
            ApplyPreviewFrame();
        }
    }

    private void AddKeyframe()
    {
        NormalizeBoundaries();
        int largestGapIndex = 0;
        int largestGap = 0;
        for (int i = 0; i < boundaries.Count - 1; i++)
        {
            int gap = boundaries[i + 1].frame - boundaries[i].frame;
            if (gap > largestGap)
            {
                largestGap = gap;
                largestGapIndex = i;
            }
        }

        if (largestGap < 2)
        {
            SetResult("No room remains for another keyframe.", MessageType.Warning);
            return;
        }

        int frame = boundaries[largestGapIndex].frame + largestGap / 2;
        AddKeyframeAtFrame(frame);
    }

    private void AddKeyframeAtFrame(int frame)
    {
        NormalizeBoundaries();
        if (frame <= 0 || frame >= totalFrames)
        {
            SetResult("A keyframe must be inside the recording, between the first and last frame.",
                MessageType.Warning);
            return;
        }

        int segmentIndex = -1;
        for (int i = 0; i < boundaries.Count - 1; i++)
        {
            if (frame > boundaries[i].frame && frame < boundaries[i + 1].frame)
            {
                segmentIndex = i;
                break;
            }
        }

        if (segmentIndex < 0)
        {
            SetResult("A keyframe already exists at this frame.", MessageType.Warning);
            return;
        }

        RecordedMotionKeyframeDefinition previousBoundary = boundaries[segmentIndex];
        RecordedMotionKeyframeDefinition nextBoundary = boundaries[segmentIndex + 1];
        float originalDuration = Mathf.Max(0.1f, previousBoundary.followingSegmentDurationSeconds);
        float firstPartRatio = (float)(frame - previousBoundary.frame) /
                               (nextBoundary.frame - previousBoundary.frame);
        boundaries.Insert(segmentIndex + 1, new RecordedMotionKeyframeDefinition
        {
            name = GetUniqueKeyframeName(),
            frame = frame,
            bodyParts = new[] { "whole_body" },
            holdSeconds = 0f,
            followingSegmentName = GetUniqueSegmentName(),
            followingSegmentDurationSeconds = Mathf.Max(0.1f, originalDuration * (1f - firstPartRatio))
        });
        previousBoundary.followingSegmentDurationSeconds =
            Mathf.Max(0.1f, originalDuration * firstPartRatio);
        NormalizeBoundaries();
        currentFrame = frame;
        ApplyPreviewFrame();
        MarkEditDefinitionDirty();
    }

    private void NormalizeBoundaries()
    {
        if (boundaries.Count < 2) return;
        boundaries[0].frame = 0;
        boundaries[boundaries.Count - 1].frame = totalFrames;

        for (int i = 1; i < boundaries.Count - 1; i++)
        {
            boundaries[i].frame = Mathf.Clamp(boundaries[i].frame, 1, Mathf.Max(1, totalFrames - 1));
        }

        boundaries.Sort((left, right) => left.frame.CompareTo(right.frame));
        for (int i = 0; i < boundaries.Count; i++) NormalizeBoundary(boundaries[i], i);
    }

    private void CreateDefaultEditDefinition()
    {
        boundaries.Clear();
        boundaries.Add(new RecordedMotionKeyframeDefinition
        {
            name = "start",
            frame = 0,
            bodyParts = new[] { "whole_body" },
            holdSeconds = 0f,
            followingSegmentName = actionId,
            followingSegmentDurationSeconds = CalculateNaturalSegmentDuration(0, totalFrames)
        });
        boundaries.Add(new RecordedMotionKeyframeDefinition
        {
            name = "end",
            frame = totalFrames,
            bodyParts = new[] { "whole_body" },
            holdSeconds = 0f,
            followingSegmentName = string.Empty,
            followingSegmentDurationSeconds = 0f
        });
    }

    private void NormalizeBoundary(RecordedMotionKeyframeDefinition boundary, int index)
    {
        if (boundary == null) return;

        if (string.IsNullOrWhiteSpace(boundary.name))
        {
            boundary.name = index == 0
                ? "start"
                : index == boundaries.Count - 1
                    ? "end"
                    : $"keyframe_{index}";
        }

        boundary.bodyParts = NormalizeBodyParts(boundary.bodyParts);
        boundary.holdSeconds = Mathf.Max(0f, boundary.holdSeconds);
        if (index < boundaries.Count - 1)
        {
            if (string.IsNullOrWhiteSpace(boundary.followingSegmentName))
                boundary.followingSegmentName = index == 0 ? actionId : $"{actionId}_segment_{index + 1}";
            if (boundary.followingSegmentDurationSeconds <= 0f)
            {
                boundary.followingSegmentDurationSeconds =
                    CalculateNaturalSegmentDuration(boundary.frame, boundaries[index + 1].frame);
            }
        }
        else
        {
            boundary.followingSegmentName = string.Empty;
            boundary.followingSegmentDurationSeconds = 0f;
        }
    }

    private float CalculateNaturalSegmentDuration(int firstFrame, int lastFrame)
    {
        float frameRate = sourceClip == null || sourceClip.frameRate <= 0f ? 30f : sourceClip.frameRate;
        return Mathf.Max(0.1f, (lastFrame - firstFrame) / frameRate);
    }

    private string GetUniqueKeyframeName()
    {
        var usedNames = new HashSet<string>(boundaries.Select(boundary => boundary.name), StringComparer.Ordinal);
        for (int number = 1; ; number++)
        {
            string candidate = $"keyframe_{number}";
            if (!usedNames.Contains(candidate)) return candidate;
        }
    }

    private string GetUniqueSegmentName()
    {
        var usedNames = new HashSet<string>(
            boundaries.Select(boundary => boundary.followingSegmentName), StringComparer.Ordinal);
        for (int number = 1; ; number++)
        {
            string candidate = $"{actionId}_segment_{number}";
            if (!usedNames.Contains(candidate)) return candidate;
        }
    }

    private void MarkEditDefinitionDirty()
    {
        if (sourceClip == null) return;
        editDefinitionDirty = true;
        nextAutoSaveAt = EditorApplication.timeSinceStartup + AutoSaveDelaySeconds;
        Repaint();
    }

    private bool SaveEditDefinition(bool reportResult)
    {
        if (sourceClip == null) return false;

        try
        {
            NormalizeBoundaries();
            string definitionPath = GetEditDefinitionPath(sourceClip);
            var definition = new RecordedMotionEditDefinition
            {
                formatVersion = EditDefinitionFormatVersion,
                sourceClipAssetPath = AssetDatabase.GetAssetPath(sourceClip),
                sourceClipName = sourceClip.name,
                sourceClipLengthSeconds = sourceClip.length,
                sourceFrameRate = sourceClip.frameRate,
                totalFrames = totalFrames,
                actionId = actionId,
                displayName = displayName,
                sampleFps = sampleFps,
                setCount = setCount,
                savedAtUtc = DateTime.UtcNow.ToString("o"),
                keyframes = boundaries.Select(CloneBoundary).ToArray()
            };

            Directory.CreateDirectory(Path.GetDirectoryName(definitionPath) ?? GetRecordingsDirectory());
            File.WriteAllText(definitionPath, JsonUtility.ToJson(definition, true), new UTF8Encoding(false));
            string definitionAssetPath = RecordedFbxPackageExporter.ToAssetPath(definitionPath);
            if (!string.IsNullOrEmpty(definitionAssetPath))
                AssetDatabase.ImportAsset(definitionAssetPath, ImportAssetOptions.ForceUpdate);

            editDefinitionDirty = false;
            lastSavedDefinitionPath = definitionPath;
            lastSavedAtLocal = DateTime.Now;
            if (reportResult)
            {
                SetResult("Keyframe edit definition saved:\n" + ToProjectRelativePath(definitionPath),
                    MessageType.Info);
            }
            else
            {
                Repaint();
            }

            return true;
        }
        catch (Exception exception)
        {
            SetResult("Could not save keyframe edit definition: " + exception.Message, MessageType.Error);
            return false;
        }
    }

    private bool LoadEditDefinition(bool reportResult)
    {
        if (sourceClip == null) return false;

        string definitionPath = GetEditDefinitionPath(sourceClip);
        if (!File.Exists(definitionPath)) return false;

        try
        {
            RecordedMotionEditDefinition definition =
                JsonUtility.FromJson<RecordedMotionEditDefinition>(File.ReadAllText(definitionPath, Encoding.UTF8));
            if (definition == null || definition.formatVersion != EditDefinitionFormatVersion ||
                definition.keyframes == null || definition.keyframes.Length < 2)
            {
                SetResult("The saved keyframe edit definition is invalid. A new definition will be used.",
                    MessageType.Warning);
                return false;
            }

            actionId = string.IsNullOrWhiteSpace(definition.actionId) ? actionId : definition.actionId.Trim();
            displayName = string.IsNullOrWhiteSpace(definition.displayName) ? displayName : definition.displayName;
            sampleFps = Mathf.Clamp(definition.sampleFps, 15, 60);
            setCount = Mathf.Max(1, definition.setCount);
            boundaries.Clear();
            boundaries.AddRange(definition.keyframes.Where(keyframe => keyframe != null).Select(CloneBoundary));
            if (boundaries.Count < 2)
            {
                SetResult("The saved keyframe edit definition has fewer than two valid keyframes.",
                    MessageType.Warning);
                boundaries.Clear();
                return false;
            }

            NormalizeBoundaries();
            editDefinitionDirty = false;
            lastSavedDefinitionPath = definitionPath;
            lastSavedAtLocal = File.GetLastWriteTime(definitionPath);
            SetResult(
                $"Loaded {AssetDatabase.GetAssetPath(sourceClip)} and restored {boundaries.Count} keyframes.",
                MessageType.Info);
            if (reportResult) Repaint();
            return true;
        }
        catch (Exception exception)
        {
            SetResult("Could not load keyframe edit definition: " + exception.Message, MessageType.Warning);
            boundaries.Clear();
            return false;
        }
    }

    private static RecordedMotionKeyframeDefinition CloneBoundary(RecordedMotionKeyframeDefinition source)
    {
        return new RecordedMotionKeyframeDefinition
        {
            name = source.name,
            frame = source.frame,
            bodyParts = source.bodyParts == null ? Array.Empty<string>() : source.bodyParts.ToArray(),
            holdSeconds = source.holdSeconds,
            followingSegmentName = source.followingSegmentName,
            followingSegmentDurationSeconds = source.followingSegmentDurationSeconds
        };
    }

    private static int BodyPartsToMask(string[] bodyParts)
    {
        int mask = 0;
        if (bodyParts == null) return mask;
        foreach (string bodyPart in bodyParts)
        {
            int index = Array.IndexOf(BodyPartValues, bodyPart);
            if (index >= 0) mask |= 1 << index;
        }
        return mask;
    }

    private static string[] MaskToBodyParts(int mask)
    {
        if ((mask & 1) != 0) return new[] { BodyPartValues[0] };

        var bodyParts = new List<string>();
        for (int i = 1; i < BodyPartValues.Length; i++)
        {
            if ((mask & 1 << i) != 0) bodyParts.Add(BodyPartValues[i]);
        }
        return bodyParts.Count == 0 ? new[] { BodyPartValues[0] } : bodyParts.ToArray();
    }

    private static string[] NormalizeBodyParts(string[] bodyParts)
    {
        return MaskToBodyParts(BodyPartsToMask(bodyParts));
    }

    private static string GetEditDefinitionPath(AnimationClip clip)
    {
        if (clip == null) return null;
        string assetPath = AssetDatabase.GetAssetPath(clip);
        if (string.IsNullOrWhiteSpace(assetPath)) return null;
        string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        string fullPath = Path.GetFullPath(Path.Combine(projectRoot, assetPath));
        return Path.ChangeExtension(fullPath, ".keyframes.json");
    }

    private static string ToProjectRelativePath(string fullPath)
    {
        if (string.IsNullOrWhiteSpace(fullPath)) return string.Empty;
        string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."))
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        string normalizedPath = Path.GetFullPath(fullPath);
        return normalizedPath.StartsWith(projectRoot, StringComparison.OrdinalIgnoreCase)
            ? normalizedPath.Substring(projectRoot.Length).Replace('\\', '/')
            : normalizedPath;
    }

    private void ExportPackage()
    {
        if (!ValidateInput(out string error))
        {
            SetResult(error, MessageType.Error);
            return;
        }

        if (!SaveEditDefinition(false)) return;

        string outputDirectory = GetRecordingsDirectory();
        Directory.CreateDirectory(outputDirectory);
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string fbxPath = Path.Combine(outputDirectory, $"{actionId}_{timestamp}.fbx");
        string jsonPath = Path.Combine(outputDirectory, $"{actionId}.motion.json");

        if (!RecordedFbxPackageExporter.TryExportPackage(sourceAvatar, sourceClip, fbxPath, out string exportMessage))
        {
            SetResult("FBX export failed: " + exportMessage, MessageType.Error);
            return;
        }

        string[] segmentNames = boundaries.Take(boundaries.Count - 1)
            .Select(boundary => boundary.followingSegmentName).ToArray();
        int[] firstFrames = boundaries.Take(boundaries.Count - 1).Select(boundary => boundary.frame).ToArray();
        int[] lastFrames = boundaries.Skip(1).Select(boundary => boundary.frame).ToArray();

        if (!RecordedFbxPackageExporter.ConfigureClipRanges(
                fbxPath, segmentNames, firstFrames, lastFrames, out string rangeMessage))
        {
            SetResult("FBX slicing failed: " + rangeMessage, MessageType.Error);
            return;
        }

        if (!TryBuildPackage(fbxPath, segmentNames, out RecordedCoachMotionPackage package, out error))
        {
            SetResult(error, MessageType.Error);
            return;
        }

        File.WriteAllText(jsonPath, JsonUtility.ToJson(package, true));
        string jsonAssetPath = RecordedFbxPackageExporter.ToAssetPath(jsonPath);
        AssetDatabase.ImportAsset(jsonAssetPath, ImportAssetOptions.ForceUpdate);
        AssetDatabase.SaveAssets();

        var manifestSegments = new MotionPackageSegmentManifest[segmentNames.Length];
        for (int i = 0; i < segmentNames.Length; i++)
        {
            manifestSegments[i] = new MotionPackageSegmentManifest
            {
                name = segmentNames[i],
                firstFrame = firstFrames[i],
                lastFrame = lastFrames[i],
                durationSeconds = package.segments[i].scoringDuration
            };
        }

        RecordingController recordingController = UnityEngine.Object.FindObjectOfType<RecordingController>();
        string trackerJsonPath = recordingController?.GetLatestTrackerJsonPath();
        string rawImuJsonPath = recordingController?.GetLatestRawImuJsonPathForAction(actionId);
        string keyframesPath = GetEditDefinitionPath(sourceClip);
        if (!MotionPackageStorage.TryCreateVersionedPackage(
                fbxPath, jsonPath, trackerJsonPath, rawImuJsonPath, keyframesPath,
                actionId, package.displayName, manifestSegments,
                out MotionPackageManifest manifest, out string manifestPath, out string packageError))
        {
            SetResult("Shared package creation failed: " + packageError, MessageType.Error);
            return;
        }

        int frameCount = package.segments.Sum(segment => segment.frames.Length);
        SetResult(
            $"Done: {package.segments.Length} clips / {frameCount} sampled frames / v{manifest.version}" +
            $" / Raw IMU: {manifest.rawImuFrameCount} frames, {manifest.rawImuSensorCount} sensors" +
            $"\nFBX: {fbxPath}\nJSON: {jsonPath}" +
            $"\nKeyframes: {lastSavedDefinitionPath}\nManifest: {manifestPath}",
            MessageType.Info);
        Debug.Log($"[IMUTrack Post Process] {lastResult}");
    }

    private bool TryBuildPackage(string fbxPath, string[] segmentNames,
        out RecordedCoachMotionPackage package, out string error)
    {
        package = null;
        error = null;
        string fbxAssetPath = RecordedFbxPackageExporter.ToAssetPath(fbxPath);
        GameObject fbxAsset = AssetDatabase.LoadAssetAtPath<GameObject>(fbxAssetPath);
        if (fbxAsset == null)
        {
            error = "The exported FBX could not be loaded from the AssetDatabase.";
            return false;
        }

        var clips = new Dictionary<string, AnimationClip>(StringComparer.Ordinal);
        foreach (UnityEngine.Object asset in AssetDatabase.LoadAllAssetsAtPath(fbxAssetPath))
        {
            if (asset is AnimationClip clip && !clip.name.StartsWith("__preview__", StringComparison.Ordinal))
                clips[clip.name] = clip;
        }

        GameObject sampleTarget = Instantiate(fbxAsset);
        sampleTarget.name = "__IMUTrackMotionSampleTarget";
        Animator sampleAnimator = sampleTarget.GetComponentInChildren<Animator>();
        Animator sourceAnimator = sourceAvatar.GetComponentInChildren<Animator>();

        try
        {
            if (sampleAnimator == null)
            {
                error = "The exported FBX has no Animator.";
                return false;
            }

            if ((sampleAnimator.avatar == null || !sampleAnimator.avatar.isHuman) &&
                sourceAnimator != null && sourceAnimator.avatar != null && sourceAnimator.avatar.isHuman)
            {
                sampleAnimator.avatar = sourceAnimator.avatar;
            }

            if (sampleAnimator.avatar == null || !sampleAnimator.avatar.isHuman)
            {
                error = "The exported FBX is not configured as a valid Humanoid.";
                return false;
            }

            var exportedSegments = new List<RecordedCoachMotionSegment>();
            for (int i = 0; i < segmentNames.Length; i++)
            {
                if (!clips.TryGetValue(segmentNames[i], out AnimationClip clip))
                {
                    error = $"The imported FBX does not contain clip '{segmentNames[i]}'.";
                    return false;
                }

                EditorUtility.DisplayProgressBar("IMUTrack motion export",
                    $"Sampling {segmentNames[i]} ({i + 1}/{segmentNames.Length})",
                    (float)i / segmentNames.Length);

                RecordedCoachMotionFrame[] frames = RecordedCoachMotionSampler.SampleClip(
                    clip, sampleTarget, sampleAnimator, sampleFps);
                if (frames.Length == 0)
                {
                    error = $"Clip '{segmentNames[i]}' produced no Humanoid frames.";
                    return false;
                }

                exportedSegments.Add(new RecordedCoachMotionSegment
                {
                    label = segmentNames[i],
                    sourceStateName = segmentNames[i],
                    repeatCount = 1,
                    scoreLeniency = 1f,
                    scoringDuration = Mathf.Max(0.1f, boundaries[i].followingSegmentDurationSeconds),
                    // Segment i ends at boundary i + 1. The destination
                    // keyframe defines what the user must match and hold.
                    bodyParts = NormalizeBodyParts(boundaries[i + 1].bodyParts),
                    holdSeconds = Mathf.Max(0f, boundaries[i + 1].holdSeconds),
                    applyFormalActionScoreMapping = true,
                    voiceClipName = null,
                    frames = frames
                });
            }

            package = new RecordedCoachMotionPackage
            {
                formatVersion = 1,
                actionId = actionId,
                displayName = string.IsNullOrWhiteSpace(displayName) ? actionId : displayName,
                fps = sampleFps,
                sourceClipName = sourceClip.name,
                setCount = setCount,
                segments = exportedSegments.ToArray()
            };
            return true;
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            DestroyImmediate(sampleTarget);
        }
    }

    private bool ValidateInput(out string error)
    {
        NormalizeBoundaries();
        if (sourceClip == null)
        {
            error = "Choose a recorded .anim file.";
            return false;
        }

        Animator animator = sourceAvatar == null ? null : sourceAvatar.GetComponentInChildren<Animator>();
        if (animator == null || animator.avatar == null || !animator.avatar.isHuman)
        {
            error = "Choose the Humanoid avatar that was recorded.";
            return false;
        }

        if (!IsValidIdentifier(actionId))
        {
            error = "actionId may contain only lowercase letters, digits, and underscores.";
            return false;
        }

        var usedNames = new HashSet<string>(StringComparer.Ordinal);
        var usedKeyframeNames = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < boundaries.Count - 1; i++)
        {
            RecordedMotionKeyframeDefinition boundary = boundaries[i];
            if (!IsValidIdentifier(boundary.name) || !usedKeyframeNames.Add(boundary.name))
            {
                error = $"Keyframe {i + 1} needs a unique name using lowercase letters, digits, and underscores.";
                return false;
            }

            if (boundary.bodyParts == null || boundary.bodyParts.Length == 0)
            {
                error = $"Keyframe '{boundary.name}' must select at least one body part.";
                return false;
            }

            if (boundary.holdSeconds < 0f || boundary.followingSegmentDurationSeconds <= 0f)
            {
                error = $"Keyframe '{boundary.name}' has an invalid hold or segment duration.";
                return false;
            }

            string name = boundary.followingSegmentName;
            if (!IsValidIdentifier(name))
            {
                error = $"Segment {i + 1} needs a name using lowercase letters, digits, and underscores.";
                return false;
            }

            if (!usedNames.Add(name))
            {
                error = $"Segment name '{name}' is duplicated.";
                return false;
            }

            if (boundaries[i + 1].frame <= boundaries[i].frame)
            {
                error = "Every keyframe boundary must be later than the previous boundary.";
                return false;
            }
        }

        RecordedMotionKeyframeDefinition finalBoundary = boundaries[boundaries.Count - 1];
        if (!IsValidIdentifier(finalBoundary.name) || !usedKeyframeNames.Add(finalBoundary.name))
        {
            error = "The final keyframe needs a unique name using lowercase letters, digits, and underscores.";
            return false;
        }

        if (finalBoundary.bodyParts == null || finalBoundary.bodyParts.Length == 0 ||
            finalBoundary.holdSeconds < 0f)
        {
            error = $"Final keyframe '{finalBoundary.name}' has invalid body-part or hold settings.";
            return false;
        }

        error = null;
        return true;
    }

    private static bool IsValidIdentifier(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return false;
        foreach (char character in value)
        {
            bool lower = character >= 'a' && character <= 'z';
            bool digit = character >= '0' && character <= '9';
            if (!lower && !digit && character != '_') return false;
        }
        return true;
    }

    private static string GetRecordingsDirectory()
    {
        return Path.GetFullPath(Path.Combine(Application.dataPath, "Recordings"));
    }

    private void SetResult(string message, MessageType type)
    {
        lastResult = message;
        lastResultType = type;
        Repaint();
    }
}
#endif
