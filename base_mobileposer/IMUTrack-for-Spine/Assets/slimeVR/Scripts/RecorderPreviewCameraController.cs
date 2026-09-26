using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using SpineFlow.RuntimeExport;

/// <summary>
/// Mouse camera controls for the Recorder scene's central avatar viewport.
/// A drag must begin inside the viewport so surrounding UI remains unaffected.
/// </summary>
[DefaultExecutionOrder(10100)]
[DisallowMultipleComponent]
public sealed class RecorderPreviewCameraController : MonoBehaviour
{
    [SerializeField, Min(0.01f)] private float rotationDegreesPerPixel = 0.18f;
    [SerializeField, Min(0.01f)] private float panSensitivity = 1f;
    [SerializeField] private Vector2 pitchLimits = new Vector2(-80f, 80f);
    [SerializeField, Min(0.1f)] private float fallbackOrbitDistance = 2f;
    [SerializeField] private bool followAvatarMotion = true;
    [SerializeField] private bool followVerticalAvatarMotion = false;
    [SerializeField, Min(0.1f)] private float resetResponse = 6f;

    private Camera controlledCamera;
    private RectTransform inputArea;
    private Canvas inputCanvas;
    private Vector3 orbitCenter;
    private float orbitDistance;
    private float yaw;
    private float pitch;
    private int activeMouseButton = -1;
    private Vector2 previousPointerPosition;
    private Transform avatarMotionTarget;
    private Vector3 previousAvatarPosition;
    private bool hasAvatarPosition;
    private Vector3 defaultOrbitCenter;
    private float defaultOrbitDistance;
    private float defaultYaw;
    private float defaultPitch;
    private Vector3 defaultAvatarPosition;
    private bool hasDefaultAvatarPosition;
    private bool isResettingView;
    private bool initialized;
    private Transform avatarViewRoot;
    private string activeModelKey;
    private Vector3 initialCameraPosition;
    private Quaternion initialCameraRotation;
    private bool hasInitialCameraPose;
    private readonly Dictionary<string, ModelViewState> modelViews =
        new Dictionary<string, ModelViewState>(StringComparer.OrdinalIgnoreCase);

    private sealed class ModelViewState
    {
        public Vector3 orbitCenter;
        public Vector3 orbitCenterOffset;
        public bool centerIsRelative;
        public float orbitDistance;
        public float yaw;
        public float pitch;
        public Vector3 defaultOrbitCenter;
        public Vector3 defaultOrbitCenterOffset;
        public bool defaultCenterIsRelative;
        public float defaultOrbitDistance;
        public float defaultYaw;
        public float defaultPitch;
    }

    public void Initialize(Camera cameraToControl, RectTransform viewportInputArea,
        Canvas canvas, Transform avatarRoot, string modelKey = null)
    {
        if (cameraToControl == null || viewportInputArea == null || canvas == null)
        {
            initialized = false;
            activeMouseButton = -1;
            return;
        }

        if (initialized) SaveActiveModelView();

        controlledCamera = cameraToControl;
        inputArea = viewportInputArea;
        inputCanvas = canvas;
        if (!hasInitialCameraPose)
        {
            initialCameraPosition = controlledCamera.transform.position;
            initialCameraRotation = controlledCamera.transform.rotation;
            hasInitialCameraPose = true;
        }

        avatarViewRoot = avatarRoot;
        avatarMotionTarget = ResolveMotionTarget(avatarRoot);
        activeModelKey = ResolveModelKey(modelKey, avatarRoot);
        if (modelViews.TryGetValue(activeModelKey, out ModelViewState savedView))
        {
            RestoreModelView(savedView);
        }
        else
        {
            // A model seen for the first time starts from the scene-authored
            // camera pose, never from the view last used by another model.
            controlledCamera.transform.SetPositionAndRotation(
                initialCameraPosition, initialCameraRotation);
            CaptureCurrentView(avatarRoot);
            CaptureDefaultView();
            SaveActiveModelView();
        }

        CaptureAvatarPosition();
        isResettingView = false;
        activeMouseButton = -1;
        initialized = true;
    }

    private void SaveActiveModelView()
    {
        if (string.IsNullOrWhiteSpace(activeModelKey)) return;
        var state = new ModelViewState
        {
            orbitCenter = orbitCenter,
            orbitDistance = orbitDistance,
            yaw = yaw,
            pitch = pitch,
            defaultOrbitCenter = defaultOrbitCenter,
            defaultOrbitDistance = defaultOrbitDistance,
            defaultYaw = defaultYaw,
            defaultPitch = defaultPitch
        };

        if (avatarViewRoot != null && IsFinite(avatarViewRoot.position))
        {
            state.centerIsRelative = true;
            state.orbitCenterOffset = orbitCenter - avatarViewRoot.position;
            state.defaultCenterIsRelative = true;
            state.defaultOrbitCenterOffset = defaultOrbitCenter - avatarViewRoot.position;
        }
        modelViews[activeModelKey] = state;
    }

    private void RestoreModelView(ModelViewState state)
    {
        bool hasRootPosition = avatarViewRoot != null && IsFinite(avatarViewRoot.position);
        orbitCenter = state.centerIsRelative && hasRootPosition
            ? avatarViewRoot.position + state.orbitCenterOffset
            : state.orbitCenter;
        orbitDistance = state.orbitDistance;
        yaw = state.yaw;
        pitch = state.pitch;
        defaultOrbitCenter = state.defaultCenterIsRelative && hasRootPosition
            ? avatarViewRoot.position + state.defaultOrbitCenterOffset
            : state.defaultOrbitCenter;
        defaultOrbitDistance = state.defaultOrbitDistance;
        defaultYaw = state.defaultYaw;
        defaultPitch = state.defaultPitch;
        hasDefaultAvatarPosition = avatarMotionTarget != null &&
                                   IsFinite(avatarMotionTarget.position);
        if (hasDefaultAvatarPosition) defaultAvatarPosition = avatarMotionTarget.position;
        ApplyOrbitView();
    }

    private static string ResolveModelKey(string modelKey, Transform avatarRoot)
    {
        if (!string.IsNullOrWhiteSpace(modelKey)) return modelKey.Trim();
        if (avatarRoot != null && !string.IsNullOrWhiteSpace(avatarRoot.name))
            return avatarRoot.name;
        return "__default_avatar__";
    }

    private void Update()
    {
        if (!initialized || controlledCamera == null || inputArea == null) return;

        if (activeMouseButton < 0)
        {
            if (Input.GetMouseButtonDown(0) && CanStartPreviewDrag())
                BeginDrag(0);
            else if (Input.GetMouseButtonDown(1) && CanStartPreviewDrag())
                BeginDrag(1);
            return;
        }

        if (!IsPointerInsideViewport() || IsPointerBlockedByOverlay())
        {
            activeMouseButton = -1;
            return;
        }

        if (!Input.GetMouseButton(activeMouseButton))
        {
            activeMouseButton = -1;
            return;
        }

        Vector2 pointerPosition = Input.mousePosition;
        Vector2 pointerDelta = pointerPosition - previousPointerPosition;
        previousPointerPosition = pointerPosition;
        if (pointerDelta.sqrMagnitude <= Mathf.Epsilon) return;

        if (activeMouseButton == 0)
            Orbit(pointerDelta);
        else
            Pan(pointerDelta);
    }

    private void LateUpdate()
    {
        if (!initialized || controlledCamera == null)
        {
            return;
        }

        if (followAvatarMotion && avatarMotionTarget != null)
        {
            Vector3 currentAvatarPosition = avatarMotionTarget.position;
            if (!IsFinite(currentAvatarPosition))
            {
                hasAvatarPosition = false;
            }
            else if (!hasAvatarPosition)
            {
                previousAvatarPosition = currentAvatarPosition;
                hasAvatarPosition = true;
            }
            else
            {
                // Move the camera and its orbit centre by exactly the same amount as
                // the avatar. This keeps the apparent size stable while preserving
                // any pan offset deliberately introduced with the right mouse button.
                Vector3 avatarDelta = currentAvatarPosition - previousAvatarPosition;
                previousAvatarPosition = currentAvatarPosition;
                if (!followVerticalAvatarMotion)
                {
                    avatarDelta.y = 0f;
                }
                if (avatarDelta.sqrMagnitude > 0.00000001f)
                {
                    orbitCenter += avatarDelta;
                    controlledCamera.transform.position += avatarDelta;
                }
            }
        }

        if (isResettingView)
        {
            UpdateSmoothReset();
        }
    }

    public void ResetViewSmoothly()
    {
        if (!initialized || controlledCamera == null) return;
        activeMouseButton = -1;
        isResettingView = true;
    }

    private void UpdateSmoothReset()
    {
        Vector3 targetCenter = GetDefaultOrbitCenter();
        float deltaTime = Mathf.Clamp(Time.unscaledDeltaTime, 1f / 240f, 0.1f);
        float blend = 1f - Mathf.Exp(-resetResponse * deltaTime);

        orbitCenter = Vector3.Lerp(orbitCenter, targetCenter, blend);
        orbitDistance = Mathf.Lerp(orbitDistance, defaultOrbitDistance, blend);
        yaw = Mathf.LerpAngle(yaw, defaultYaw, blend);
        pitch = Mathf.Lerp(pitch, defaultPitch, blend);

        bool finished = Vector3.SqrMagnitude(orbitCenter - targetCenter) < 0.000001f &&
                        Mathf.Abs(Mathf.DeltaAngle(yaw, defaultYaw)) < 0.05f &&
                        Mathf.Abs(pitch - defaultPitch) < 0.05f &&
                        Mathf.Abs(orbitDistance - defaultOrbitDistance) < 0.001f;
        if (finished)
        {
            orbitCenter = targetCenter;
            orbitDistance = defaultOrbitDistance;
            yaw = defaultYaw;
            pitch = defaultPitch;
            isResettingView = false;
        }

        ApplyOrbitView();
    }

    private void BeginDrag(int mouseButton)
    {
        isResettingView = false;
        activeMouseButton = mouseButton;
        previousPointerPosition = Input.mousePosition;
    }

    private bool IsPointerInsideViewport()
    {
        Camera eventCamera = inputCanvas != null && inputCanvas.renderMode != RenderMode.ScreenSpaceOverlay
            ? inputCanvas.worldCamera
            : null;
        return RectTransformUtility.RectangleContainsScreenPoint(
            inputArea, Input.mousePosition, eventCamera);
    }

    private bool CanStartPreviewDrag()
    {
        return IsPointerInsideViewport() && !IsPointerBlockedByOverlay() &&
               (EventSystem.current == null || !EventSystem.current.IsPointerOverGameObject());
    }

    private static bool IsPointerBlockedByOverlay()
    {
        return RuntimeRecordedMotionEditor.IsPointerOverEditorWindow(Input.mousePosition);
    }

    private void CaptureCurrentView(Transform avatarRoot)
    {
        Transform cameraTransform = controlledCamera.transform;
        orbitDistance = Mathf.Max(0.1f, fallbackOrbitDistance);

        if (avatarRoot != null)
        {
            Vector3 horizontalForward = Vector3.ProjectOnPlane(cameraTransform.forward, Vector3.up);
            if (horizontalForward.sqrMagnitude > 0.0001f)
            {
                horizontalForward.Normalize();
                var targetDepthPlane = new Plane(horizontalForward, avatarRoot.position);
                var viewRay = new Ray(cameraTransform.position, cameraTransform.forward);
                if (targetDepthPlane.Raycast(viewRay, out float distance) && distance > 0.1f)
                    orbitDistance = distance;
            }
        }

        orbitCenter = cameraTransform.position + cameraTransform.forward * orbitDistance;
        Vector3 euler = cameraTransform.eulerAngles;
        yaw = euler.y;
        pitch = NormalizeSignedAngle(euler.x);
    }

    private void CaptureAvatarPosition()
    {
        hasAvatarPosition = avatarMotionTarget != null &&
                            IsFinite(avatarMotionTarget.position);
        if (hasAvatarPosition)
        {
            previousAvatarPosition = avatarMotionTarget.position;
        }
    }

    private void CaptureDefaultView()
    {
        defaultOrbitCenter = orbitCenter;
        defaultOrbitDistance = orbitDistance;
        defaultYaw = yaw;
        defaultPitch = pitch;
        hasDefaultAvatarPosition = avatarMotionTarget != null &&
                                   IsFinite(avatarMotionTarget.position);
        if (hasDefaultAvatarPosition)
        {
            defaultAvatarPosition = avatarMotionTarget.position;
        }
    }

    private Vector3 GetDefaultOrbitCenter()
    {
        Vector3 targetCenter = defaultOrbitCenter;
        if (!followAvatarMotion || !hasDefaultAvatarPosition ||
            avatarMotionTarget == null || !IsFinite(avatarMotionTarget.position))
        {
            return targetCenter;
        }

        Vector3 avatarDelta = avatarMotionTarget.position - defaultAvatarPosition;
        if (!followVerticalAvatarMotion)
        {
            avatarDelta.y = 0f;
        }
        return targetCenter + avatarDelta;
    }

    private static Transform ResolveMotionTarget(Transform avatarRoot)
    {
        if (avatarRoot == null)
        {
            return null;
        }

        Animator animator = avatarRoot.GetComponent<Animator>();
        if (animator == null)
        {
            animator = avatarRoot.GetComponentInChildren<Animator>();
        }

        if (animator != null && animator.isHuman)
        {
            Transform hips = animator.GetBoneTransform(HumanBodyBones.Hips);
            if (hips != null)
            {
                return hips;
            }
        }

        return avatarRoot;
    }

    private void Orbit(Vector2 pointerDelta)
    {
        yaw += pointerDelta.x * rotationDegreesPerPixel;
        pitch = Mathf.Clamp(pitch - pointerDelta.y * rotationDegreesPerPixel,
            Mathf.Min(pitchLimits.x, pitchLimits.y), Mathf.Max(pitchLimits.x, pitchLimits.y));
        ApplyOrbitView();
    }

    private void Pan(Vector2 pointerDelta)
    {
        float viewportPixelHeight = Mathf.Max(1f,
            inputArea.rect.height * (inputCanvas == null ? 1f : inputCanvas.scaleFactor));
        float visibleWorldHeight = controlledCamera.orthographic
            ? controlledCamera.orthographicSize * 2f
            : 2f * orbitDistance * Mathf.Tan(controlledCamera.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float worldUnitsPerPixel = visibleWorldHeight / viewportPixelHeight;
        Vector3 translation = (-controlledCamera.transform.right * pointerDelta.x -
                               controlledCamera.transform.up * pointerDelta.y) *
                              (worldUnitsPerPixel * panSensitivity);
        orbitCenter += translation;
        controlledCamera.transform.position += translation;
    }

    private void ApplyOrbitView()
    {
        Quaternion rotation = Quaternion.Euler(pitch, yaw, 0f);
        Transform cameraTransform = controlledCamera.transform;
        cameraTransform.SetPositionAndRotation(
            orbitCenter - rotation * Vector3.forward * orbitDistance, rotation);
    }

    private void OnApplicationFocus(bool hasFocus)
    {
        if (!hasFocus) activeMouseButton = -1;
    }

    private static float NormalizeSignedAngle(float angle)
    {
        return angle > 180f ? angle - 360f : angle;
    }

    private static bool IsFinite(Vector3 value)
    {
        return !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
               !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
               !float.IsNaN(value.z) && !float.IsInfinity(value.z);
    }
}
