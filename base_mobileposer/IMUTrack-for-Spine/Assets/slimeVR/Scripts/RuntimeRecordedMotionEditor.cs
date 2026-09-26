using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using EVMC4U;
using SpineFlow.MotionEditing;
using SpineFlow.PoseRecording;
using UnityEngine;

namespace SpineFlow.RuntimeExport
{
    /// <summary>
    /// Lightweight in-player keyframe editor used by the standalone recorder Build.
    /// It intentionally uses IMGUI so the existing Recorder scene needs no extra prefab wiring.
    /// </summary>
    [DefaultExecutionOrder(9990)]
    public sealed class RuntimeRecordedMotionEditor : MonoBehaviour
    {
        private static RuntimeRecordedMotionEditor activeEditor;

        private static readonly string[] BodyPartValues =
        {
            "whole_body", "torso", "hips", "head_neck", "left_arm", "right_arm", "left_leg", "right_leg"
        };

        private static readonly string[] BodyPartLabels =
        {
            "全身", "躯干", "髋部", "头颈", "左臂", "右臂", "左腿", "右腿"
        };

        private RecordingController owner;
        private Animator animator;
        private RuntimePoseRecordingPackage recording;
        private RecordedMotionEditDefinition definition;
        private readonly List<RecordedMotionKeyframeDefinition> boundaries =
            new List<RecordedMotionKeyframeDefinition>();
        private HumanPoseHandler poseHandler;
        private HumanPose pose;
        private ExternalReceiver externalReceiver;
        private bool receiverWasFrozen;
        private string recordingPath;
        private string definitionPath;
        private string status;
        private Rect windowRect;
        private Vector2 keyframeScroll;
        private int currentFrame;
        private int selectedBoundary;
        private bool isPreviewPlaying;
        private float previewTime;
        private bool closing;

        /// <summary>
        /// Input.mousePosition uses a bottom-left origin while IMGUI uses a
        /// top-left origin. Camera controls use this method to avoid treating
        /// interaction with the editor window as a preview drag.
        /// </summary>
        public static bool IsPointerOverEditorWindow(Vector2 screenPosition)
        {
            if (activeEditor == null || !activeEditor.isActiveAndEnabled)
                return false;

            Vector2 guiPosition = new Vector2(
                screenPosition.x, Screen.height - screenPosition.y);
            return activeEditor.windowRect.Contains(guiPosition);
        }

        public static bool TryOpen(RecordingController owner, Animator animator, string recordingPath,
            string actionId, string displayName, out string error)
        {
            error = null;
            if (owner == null)
            {
                error = "RecordingController 未就绪";
                return false;
            }
            if (!RuntimePosePlaybackUtility.IsValidHumanoid(animator))
            {
                error = "未找到可用于预览的有效 Humanoid Animator";
                return false;
            }

            if (!RuntimePoseRecordingStorage.TryLoad(recordingPath,
                    out RuntimePoseRecordingPackage recording, out error))
                return false;
            if (recording.frames.Length < 2)
            {
                error = "录制帧数不足，无法编辑关键帧";
                return false;
            }

            RuntimeRecordedMotionEditor existing = FindObjectOfType<RuntimeRecordedMotionEditor>();
            if (existing != null) existing.Close(true);

            RuntimeRecordedMotionEditor editor = owner.gameObject.AddComponent<RuntimeRecordedMotionEditor>();
            if (!editor.Initialize(owner, animator, recordingPath, recording, actionId, displayName, out error))
            {
                Destroy(editor);
                return false;
            }

            return true;
        }

        private bool Initialize(RecordingController controller, Animator targetAnimator, string path,
            RuntimePoseRecordingPackage source, string actionId, string displayName, out string error)
        {
            error = null;
            owner = controller;
            activeEditor = this;
            animator = targetAnimator;
            recordingPath = Path.GetFullPath(path);
            definitionPath = RuntimePoseRecordingStorage.GetKeyframesPath(recordingPath);
            recording = source;
            windowRect = new Rect(
                Mathf.Max(8f, (Screen.width - 820f) * 0.5f),
                Mathf.Max(8f, (Screen.height - 680f) * 0.5f),
                Mathf.Min(820f, Screen.width - 16f),
                Mathf.Min(680f, Screen.height - 16f));

            if (File.Exists(definitionPath))
            {
                try
                {
                    definition = JsonUtility.FromJson<RecordedMotionEditDefinition>(
                        File.ReadAllText(definitionPath));
                }
                catch (Exception exception)
                {
                    status = "原关键帧文件无法读取，已创建新定义：" + exception.Message;
                }
            }

            if (definition == null || definition.keyframes == null || definition.keyframes.Length < 2)
                definition = RuntimeMotionPackageExporter.CreateDefaultEditDefinition(
                    recording, actionId, displayName);

            definition.actionId = actionId;
            definition.displayName = string.IsNullOrWhiteSpace(displayName) ? actionId : displayName.Trim();
            definition.sourceClipAssetPath = recordingPath;
            definition.sourceClipName = recording.trajectoryName;
            definition.sourceClipLengthSeconds = recording.durationSeconds;
            definition.sourceFrameRate = recording.sampleRate;
            definition.totalFrames = recording.frames.Length - 1;
            if (!RuntimeMotionPackageExporter.TryNormalizeAndValidate(
                    definition, recording.frames.Length - 1, actionId, out error))
            {
                definition = RuntimeMotionPackageExporter.CreateDefaultEditDefinition(
                    recording, actionId, displayName);
                status = "原关键帧定义无效，当前显示默认起点和终点：" + error;
                error = null;
            }

            boundaries.Clear();
            boundaries.AddRange(definition.keyframes);
            poseHandler = new HumanPoseHandler(animator.avatar, animator.transform);
            pose = new HumanPose { muscles = new float[HumanTrait.MuscleCount] };
            externalReceiver = FindObjectOfType<ExternalReceiver>();
            if (externalReceiver != null)
            {
                receiverWasFrozen = externalReceiver.Freeze;
                externalReceiver.Freeze = true;
            }

            currentFrame = 0;
            selectedBoundary = 0;
            ApplyCurrentFrame();
            if (string.IsNullOrWhiteSpace(status)) status = "可预览并编辑关键帧；保存后导出动作包会自动使用该定义。";
            return true;
        }

        private void OnDisable()
        {
            if (activeEditor == this) activeEditor = null;
        }

        private void Update()
        {
            if (!isPreviewPlaying || recording == null) return;
            previewTime += Time.unscaledDeltaTime;
            float duration = recording.frames[recording.frames.Length - 1].timeSeconds;
            if (previewTime >= duration)
            {
                previewTime = duration;
                isPreviewPlaying = false;
            }

            while (currentFrame + 1 < recording.frames.Length &&
                   recording.frames[currentFrame + 1].timeSeconds <= previewTime)
                currentFrame++;
        }

        private void LateUpdate()
        {
            if (recording == null || poseHandler == null) return;
            if (isPreviewPlaying)
            {
                int lowerFrame = currentFrame;
                RuntimePosePlaybackUtility.ApplyAtTime(recording, previewTime,
                    ref lowerFrame, poseHandler, ref pose);
                currentFrame = lowerFrame;
            }
            else
            {
                ApplyCurrentFrame();
            }
        }

        private void OnGUI()
        {
            GUI.depth = -10000;
            Color previousColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.65f);
            GUI.Box(new Rect(0f, 0f, Screen.width, Screen.height), GUIContent.none);
            GUI.color = previousColor;
            windowRect = GUI.Window(GetInstanceID(), windowRect, DrawWindow, "关键帧编辑（运行时）");
        }

        private void DrawWindow(int id)
        {
            GUILayout.Label("录制：" + Path.GetFileName(recordingPath));
            GUILayout.Label($"动作：{definition.actionId}    帧：0-{recording.frames.Length - 1}    " +
                            $"时长：{recording.durationSeconds:F3}s");

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("|<", GUILayout.Width(42f))) SetCurrentFrame(0);
            if (GUILayout.Button("<", GUILayout.Width(42f))) SetCurrentFrame(currentFrame - 1);
            if (GUILayout.Button(isPreviewPlaying ? "暂停" : "播放", GUILayout.Width(70f)))
            {
                if (!isPreviewPlaying && currentFrame >= recording.frames.Length - 1)
                    SetCurrentFrame(0);
                isPreviewPlaying = !isPreviewPlaying;
                previewTime = recording.frames[currentFrame].timeSeconds;
            }
            if (GUILayout.Button(">", GUILayout.Width(42f))) SetCurrentFrame(currentFrame + 1);
            if (GUILayout.Button(">|", GUILayout.Width(42f))) SetCurrentFrame(recording.frames.Length - 1);
            int sliderFrame = Mathf.RoundToInt(GUILayout.HorizontalSlider(
                currentFrame, 0f, recording.frames.Length - 1, GUILayout.ExpandWidth(true)));
            if (sliderFrame != currentFrame) SetCurrentFrame(sliderFrame);
            GUILayout.Label($"{currentFrame}/{recording.frames.Length - 1}", GUILayout.Width(82f));
            GUILayout.EndHorizontal();

            GUILayout.Space(6f);
            GUILayout.BeginHorizontal();
            GUILayout.BeginVertical(GUI.skin.box, GUILayout.Width(270f), GUILayout.ExpandHeight(true));
            GUILayout.Label("关键帧边界");
            keyframeScroll = GUILayout.BeginScrollView(keyframeScroll);
            for (int index = 0; index < boundaries.Count; index++)
            {
                RecordedMotionKeyframeDefinition boundary = boundaries[index];
                string label = index == 0 ? "起点" : index == boundaries.Count - 1 ? "终点" : "关键帧 " + index;
                bool selected = index == selectedBoundary;
                Color old = GUI.backgroundColor;
                if (selected) GUI.backgroundColor = new Color(0.45f, 0.75f, 1f);
                if (GUILayout.Button($"{label}  [{boundary.frame}]  {boundary.name}"))
                {
                    selectedBoundary = index;
                    SetCurrentFrame(boundary.frame);
                }
                GUI.backgroundColor = old;
            }
            GUILayout.EndScrollView();
            GUI.enabled = currentFrame > 0 && currentFrame < recording.frames.Length - 1 &&
                          boundaries.All(item => item.frame != currentFrame);
            if (GUILayout.Button("将当前帧设为关键帧")) AddBoundaryAtCurrentFrame();
            GUI.enabled = selectedBoundary > 0 && selectedBoundary < boundaries.Count - 1;
            if (GUILayout.Button("删除选中关键帧")) RemoveSelectedBoundary();
            GUI.enabled = true;
            GUILayout.EndVertical();

            GUILayout.BeginVertical(GUI.skin.box, GUILayout.ExpandWidth(true), GUILayout.ExpandHeight(true));
            DrawSelectedBoundary();
            GUILayout.EndVertical();
            GUILayout.EndHorizontal();

            GUILayout.Label(status ?? string.Empty, GUI.skin.box, GUILayout.Height(46f));
            GUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("保存", GUILayout.Width(110f), GUILayout.Height(32f))) Save(true);
            if (GUILayout.Button("保存并关闭", GUILayout.Width(130f), GUILayout.Height(32f))) Close(true);
            if (GUILayout.Button("取消关闭", GUILayout.Width(110f), GUILayout.Height(32f))) Close(false);
            GUILayout.EndHorizontal();
            GUI.DragWindow(new Rect(0f, 0f, windowRect.width, 24f));
        }

        private void DrawSelectedBoundary()
        {
            selectedBoundary = Mathf.Clamp(selectedBoundary, 0, boundaries.Count - 1);
            RecordedMotionKeyframeDefinition boundary = boundaries[selectedBoundary];
            bool endpoint = selectedBoundary == 0 || selectedBoundary == boundaries.Count - 1;
            GUILayout.Label(endpoint
                ? (selectedBoundary == 0 ? "起点属性（帧固定为 0）" : "终点属性（帧固定为最后一帧）")
                : "关键帧属性");

            GUILayout.BeginHorizontal();
            GUILayout.Label("名称", GUILayout.Width(105f));
            boundary.name = GUILayout.TextField(boundary.name ?? string.Empty);
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUILayout.Label("帧位置", GUILayout.Width(105f));
            GUI.enabled = !endpoint;
            string frameText = GUILayout.TextField(boundary.frame.ToString(CultureInfo.InvariantCulture));
            if (!endpoint && int.TryParse(frameText, out int frameValue))
                boundary.frame = Mathf.Clamp(frameValue, 1, recording.frames.Length - 2);
            GUI.enabled = true;
            GUILayout.EndHorizontal();

            GUILayout.Label("匹配身体部位（选择全身会覆盖其他选项）");
            int bodyMask = BodyPartsToMask(boundary.bodyParts);
            for (int row = 0; row < 2; row++)
            {
                GUILayout.BeginHorizontal();
                for (int column = 0; column < 4; column++)
                {
                    int partIndex = row * 4 + column;
                    bool before = (bodyMask & (1 << partIndex)) != 0;
                    bool after = GUILayout.Toggle(before, BodyPartLabels[partIndex], GUILayout.Width(105f));
                    if (after != before)
                    {
                        if (after) bodyMask |= 1 << partIndex;
                        else bodyMask &= ~(1 << partIndex);
                        if (partIndex == 0 && after) bodyMask = 1;
                        else if (partIndex > 0 && after) bodyMask &= ~1;
                    }
                }
                GUILayout.EndHorizontal();
            }
            boundary.bodyParts = MaskToBodyParts(bodyMask);

            DrawFloatField("保持秒数", ref boundary.holdSeconds, 0f);
            if (selectedBoundary < boundaries.Count - 1)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("后续片段名", GUILayout.Width(105f));
                boundary.followingSegmentName = GUILayout.TextField(boundary.followingSegmentName ?? string.Empty);
                GUILayout.EndHorizontal();
                DrawFloatField("片段目标时长", ref boundary.followingSegmentDurationSeconds, 0.1f);
            }
            else
            {
                GUILayout.Label("终点之后没有动作片段。", GUI.skin.box);
            }

            GUILayout.Space(8f);
            GUILayout.Label("名称仅允许小写字母、数字和下划线。身体部位与保持时间将用于动作评分。",
                GUI.skin.box);
        }

        private static void DrawFloatField(string label, ref float value, float minimum)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(105f));
            string text = GUILayout.TextField(value.ToString("0.###", CultureInfo.InvariantCulture));
            if (float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ||
                float.TryParse(text, NumberStyles.Float, CultureInfo.CurrentCulture, out parsed))
                value = Mathf.Max(minimum, parsed);
            GUILayout.EndHorizontal();
        }

        private void SetCurrentFrame(int frame)
        {
            currentFrame = Mathf.Clamp(frame, 0, recording.frames.Length - 1);
            previewTime = recording.frames[currentFrame].timeSeconds;
            isPreviewPlaying = false;
            ApplyCurrentFrame();
        }

        private void ApplyCurrentFrame()
        {
            RuntimePosePlaybackUtility.ApplyFrame(recording, currentFrame, poseHandler, ref pose);
        }

        private void AddBoundaryAtCurrentFrame()
        {
            if (currentFrame <= 0 || currentFrame >= recording.frames.Length - 1 ||
                boundaries.Any(item => item.frame == currentFrame))
                return;

            NormalizeBoundaryOrder();
            int insertAfter = boundaries.FindLastIndex(item => item.frame < currentFrame);
            RecordedMotionKeyframeDefinition previous = boundaries[insertAfter];
            RecordedMotionKeyframeDefinition next = boundaries[insertAfter + 1];
            float duration = Mathf.Max(0.1f, previous.followingSegmentDurationSeconds);
            float firstRatio = (float)(currentFrame - previous.frame) / (next.frame - previous.frame);
            previous.followingSegmentDurationSeconds = Mathf.Max(0.1f, duration * firstRatio);
            var boundary = new RecordedMotionKeyframeDefinition
            {
                name = GetUniqueName("keyframe"),
                frame = currentFrame,
                bodyParts = new[] { "whole_body" },
                followingSegmentName = GetUniqueName(definition.actionId + "_segment", true),
                followingSegmentDurationSeconds = Mathf.Max(0.1f, duration * (1f - firstRatio))
            };
            boundaries.Insert(insertAfter + 1, boundary);
            selectedBoundary = insertAfter + 1;
            status = "已添加关键帧；请编辑名称、身体部位和片段属性。";
        }

        private void RemoveSelectedBoundary()
        {
            if (selectedBoundary <= 0 || selectedBoundary >= boundaries.Count - 1) return;
            boundaries[selectedBoundary - 1].followingSegmentDurationSeconds = Mathf.Max(0.1f,
                boundaries[selectedBoundary - 1].followingSegmentDurationSeconds +
                boundaries[selectedBoundary].followingSegmentDurationSeconds);
            boundaries.RemoveAt(selectedBoundary);
            selectedBoundary = Mathf.Clamp(selectedBoundary - 1, 0, boundaries.Count - 1);
            status = "已删除关键帧。";
        }

        private void NormalizeBoundaryOrder()
        {
            boundaries[0].frame = 0;
            boundaries[boundaries.Count - 1].frame = recording.frames.Length - 1;
            boundaries.Sort((left, right) => left.frame.CompareTo(right.frame));
        }

        private string GetUniqueName(string prefix, bool segment = false)
        {
            for (int index = 1;; index++)
            {
                string candidate = prefix + "_" + index;
                bool exists = segment
                    ? boundaries.Any(item => item.followingSegmentName == candidate)
                    : boundaries.Any(item => item.name == candidate);
                if (!exists) return candidate;
            }
        }

        private bool Save(bool report)
        {
            NormalizeBoundaryOrder();
            definition.keyframes = boundaries.ToArray();
            definition.sourceClipAssetPath = recordingPath;
            definition.savedAtUtc = DateTime.UtcNow.ToString("o");
            if (!RuntimeMotionPackageExporter.TryNormalizeAndValidate(
                    definition, recording.frames.Length - 1, definition.actionId, out string validationError))
            {
                status = "保存失败：" + validationError;
                return false;
            }

            boundaries.Clear();
            boundaries.AddRange(definition.keyframes);
            if (!RuntimeMotionPackageExporter.TrySaveEditDefinition(definitionPath, definition, out string error))
            {
                status = error;
                return false;
            }

            status = "已保存：" + definitionPath;
            if (report && owner != null) owner.ShowOperationStatus("关键帧已保存：" + definitionPath);
            return true;
        }

        private void Close(bool save)
        {
            if (closing) return;
            if (save && !Save(true)) return;
            closing = true;
            Destroy(this);
        }

        private void OnDestroy()
        {
            poseHandler?.Dispose();
            poseHandler = null;
            if (externalReceiver != null) externalReceiver.Freeze = receiverWasFrozen;
        }

        private static int BodyPartsToMask(string[] values)
        {
            int mask = 0;
            if (values != null)
            {
                foreach (string value in values)
                {
                    int index = Array.IndexOf(BodyPartValues, value);
                    if (index >= 0) mask |= 1 << index;
                }
            }
            return mask == 0 ? 1 : mask;
        }

        private static string[] MaskToBodyParts(int mask)
        {
            if ((mask & 1) != 0 || mask == 0) return new[] { "whole_body" };
            var values = new List<string>();
            for (int index = 1; index < BodyPartValues.Length; index++)
                if ((mask & (1 << index)) != 0) values.Add(BodyPartValues[index]);
            return values.Count == 0 ? new[] { "whole_body" } : values.ToArray();
        }
    }
}
