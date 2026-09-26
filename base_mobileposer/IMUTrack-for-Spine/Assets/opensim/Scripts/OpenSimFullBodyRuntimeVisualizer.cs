using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
public class OpenSimFullBodyRuntimeVisualizer : MonoBehaviour
{
    [Header("OpenSim Files")]
    public string maleOsimFile = "opensim/Data/MaleFullBodyModel_v2.0_OS4_BU.osim";
    public string femaleOsimFile = "opensim/Data/FemaleFullBodyModel_v2.0_OS4_BU.osim";
    public string maleMotionFile = "opensim/Data/T1_to_L5_synthetic_spine_motion_heatmap_clean.mot";
    public string femaleMotionFile = "opensim/Data/T1_to_L5_synthetic_spine_motion_heatmap_clean.mot";
    public string geometryDirectory = "opensim/Data/Geometry";

    [Header("Playback")]
    public bool playOnStart = true;
    public bool loop = true;
    public float playbackSpeed = 1.0f;

    [Header("Layout")]
    public Vector3 maleOffset = new Vector3(-0.75f, 0.0f, 0.0f);
    public Vector3 femaleOffset = new Vector3(0.75f, 0.0f, 0.0f);
    public float modelScale = 1.0f;
    public bool loadOpenSimMeshes = true;
    public bool showDebugJointSkeleton = false;
    public bool showSpineHeatOverlay = false;
    public float bodyNodeRadius = 0.012f;
    public float boneLineWidth = 0.008f;
    public float spineSegmentRadius = 0.018f;

    [Header("Spine Heatmap")]
    public bool useFrameRelativeContrast = true;
    public float maxRotationForFullHeat = 3.25f;
    public Color lowRotationColor = new Color(0.05f, 0.18f, 0.95f, 1.0f);
    public Color middleRotationColor = new Color(1.0f, 0.86f, 0.05f, 1.0f);
    public Color highRotationColor = new Color(0.95f, 0.04f, 0.02f, 1.0f);
    public Color skeletonColor = new Color(0.72f, 0.76f, 0.80f, 1.0f);

    [Header("Debug")]
    public bool showOverlay = true;

    [Header("Editor")]
    public bool buildInEditMode = true;
    public bool animateInEditMode = false;

    private static readonly string[] SpineLevels =
    {
        "T1_T2", "T2_T3", "T3_T4", "T4_T5", "T5_T6", "T6_T7",
        "T7_T8", "T8_T9", "T9_T10", "T10_T11", "T11_T12", "T12_L1",
        "L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"
    };

    private static readonly Dictionary<string, string> SpineLevelToColoredBody = new Dictionary<string, string>
    {
        { "T1_T2", "thoracic2" },
        { "T2_T3", "thoracic3" },
        { "T3_T4", "thoracic4" },
        { "T4_T5", "thoracic5" },
        { "T5_T6", "thoracic6" },
        { "T6_T7", "thoracic7" },
        { "T7_T8", "thoracic8" },
        { "T8_T9", "thoracic9" },
        { "T9_T10", "thoracic10" },
        { "T10_T11", "thoracic11" },
        { "T11_T12", "thoracic12" },
        { "T12_L1", "lumbar1" },
        { "L1_L2", "lumbar2" },
        { "L2_L3", "lumbar3" },
        { "L3_L4", "lumbar4" },
        { "L4_L5", "lumbar5" },
        { "L5_S1", "sacrum" }
    };

    private readonly List<ModelRuntime> models = new List<ModelRuntime>();
    private Material skeletonMaterial;
    private Material bodyMaterial;
    private float playTime;
    private bool isPlaying;
    private string statusMessage = "Not initialized";
    private static readonly Dictionary<string, Mesh> SharedMeshCache = new Dictionary<string, Mesh>(StringComparer.OrdinalIgnoreCase);

    private void OnEnable()
    {
        if (!Application.isPlaying && buildInEditMode)
        {
            BuildScene();
        }
    }

    private void Start()
    {
        if (Application.isPlaying)
        {
            BuildScene();
        }
    }

    private void Update()
    {
        if (!Application.isPlaying && !animateInEditMode)
        {
            return;
        }

        if (models.Count == 0)
        {
            return;
        }

        if (isPlaying)
        {
            playTime += Time.deltaTime * playbackSpeed;
        }

        float maxDuration = 0.0f;
        for (int i = 0; i < models.Count; i++)
        {
            maxDuration = Mathf.Max(maxDuration, models[i].Motion.Duration);
        }

        if (loop && maxDuration > 0.0f)
        {
            playTime = Mathf.Repeat(playTime, maxDuration);
        }
        else
        {
            playTime = Mathf.Min(playTime, maxDuration);
        }

        for (int i = 0; i < models.Count; i++)
        {
            UpdateModel(models[i], playTime);
        }
    }

    [ContextMenu("Rebuild OpenSim Visualizers")]
    public void BuildScene()
    {
        ClearGeneratedChildren();

        skeletonMaterial = CreateMaterial("OpenSim Skeleton", skeletonColor);
        bodyMaterial = CreateMaterial("OpenSim Body Nodes", new Color(0.86f, 0.86f, 0.80f, 1.0f));

        models.Clear();
        playTime = 0.0f;
        isPlaying = playOnStart;

        try
        {
            AddModel("Male", maleOsimFile, maleMotionFile, maleOffset);
            AddModel("Female", femaleOsimFile, femaleMotionFile, femaleOffset);
            int loadedMeshes = 0;
            int skippedMeshes = 0;
            for (int i = 0; i < models.Count; i++)
            {
                loadedMeshes += models[i].LoadedMeshCount;
                skippedMeshes += models[i].SkippedMeshCount;
            }
            statusMessage = "Loaded male and female OpenSim full-body meshes (" + loadedMeshes + " meshes, " + skippedMeshes + " skipped). Rotation heatmap is computed live each frame.";
            Debug.Log(statusMessage, this);
        }
        catch (Exception ex)
        {
            statusMessage = "OpenSim visualizer error: " + ex.Message;
            Debug.LogException(ex, this);
        }
    }

    private void AddModel(string displayName, string osimFile, string motionFile, Vector3 offset)
    {
        string osimPath = ResolvePath(osimFile);
        string motionPath = ResolvePath(motionFile);
        string geometryPath = ResolvePath(geometryDirectory);

        OpenSimModel parsedModel = OpenSimModel.Load(osimPath);
        MotionStorage motion = MotionStorage.Load(motionPath);

        GameObject rootObject = new GameObject(displayName + " OpenSim FullBody");
        rootObject.transform.SetParent(transform, false);
        rootObject.transform.localPosition = offset;
        rootObject.transform.localRotation = Quaternion.identity;
        rootObject.transform.localScale = Vector3.one * modelScale;

        ModelRuntime runtime = new ModelRuntime
        {
            DisplayName = displayName,
            Root = rootObject.transform,
            ParsedModel = parsedModel,
            Motion = motion
        };

        CreateBodyTransforms(runtime);
        if (loadOpenSimMeshes)
        {
            CreateOpenSimMeshes(runtime, geometryPath, Path.GetDirectoryName(osimPath));
        }
        CreateSkeletonLines(runtime);
        CreateSpineHeatSegments(runtime);

        ApplyPose(runtime, runtime.Motion.StartTime);
        CaptureReferenceSpineRotations(runtime);
        UpdateHeatmap(runtime);

        models.Add(runtime);
    }

    private void CreateBodyTransforms(ModelRuntime runtime)
    {
        foreach (string bodyName in runtime.ParsedModel.BodyNames)
        {
            GameObject bodyObject = new GameObject(bodyName);
            Transform bodyTransform = bodyObject.transform;
            bodyTransform.SetParent(runtime.Root, false);
            runtime.BodyTransforms[bodyName] = bodyTransform;

            if (!showDebugJointSkeleton)
            {
                continue;
            }

            GameObject node = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            node.name = bodyName + "_node";
            node.transform.SetParent(bodyTransform, false);
            node.transform.localScale = Vector3.one * bodyNodeRadius * 2.0f;

            Renderer renderer = node.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = bodyMaterial;
            }

            Collider collider = node.GetComponent<Collider>();
            if (collider != null)
            {
                Destroy(collider);
            }
        }

        foreach (JointInfo joint in runtime.ParsedModel.Joints)
        {
            if (string.IsNullOrEmpty(joint.ChildBody) || !runtime.BodyTransforms.ContainsKey(joint.ChildBody))
            {
                continue;
            }

            Transform child = runtime.BodyTransforms[joint.ChildBody];
            Transform parent = runtime.Root;
            if (!string.IsNullOrEmpty(joint.ParentBody) && runtime.BodyTransforms.ContainsKey(joint.ParentBody))
            {
                parent = runtime.BodyTransforms[joint.ParentBody];
            }

            child.SetParent(parent, false);
            ComposeJointLocalTransform(joint, Quaternion.identity, Vector3.zero, out Vector3 localPosition, out Quaternion localRotation);
            child.localPosition = localPosition;
            child.localRotation = localRotation;
            child.localScale = Vector3.one;

            joint.BaseLocalPosition = child.localPosition;
            joint.BaseLocalRotation = child.localRotation;
            runtime.JointsByChildBody[joint.ChildBody] = joint;

            for (int i = 0; i < joint.Coordinates.Count; i++)
            {
                runtime.JointsByCoordinate[joint.Coordinates[i]] = joint;
            }
        }
    }

    private void CreateOpenSimMeshes(ModelRuntime runtime, string geometryPath, string osimDirectory)
    {
        foreach (BodyGeometryInfo geometryInfo in runtime.ParsedModel.BodyGeometry)
        {
            Transform bodyTransform;
            if (!runtime.BodyTransforms.TryGetValue(geometryInfo.BodyName, out bodyTransform))
            {
                continue;
            }

            string meshPath = ResolveMeshPath(geometryInfo.MeshFile, geometryPath, osimDirectory);
            if (string.IsNullOrEmpty(meshPath) || !File.Exists(meshPath))
            {
                Debug.LogWarning("OpenSim mesh not found: " + geometryInfo.MeshFile + " for body " + geometryInfo.BodyName, this);
                continue;
            }

            Mesh mesh;
            if (!SharedMeshCache.TryGetValue(meshPath, out mesh))
            {
                try
                {
                    mesh = OpenSimMeshLoader.Load(meshPath);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("Skipped OpenSim mesh '" + meshPath + "' for body '" + geometryInfo.BodyName + "': " + ex.Message, this);
                    runtime.SkippedMeshCount++;
                    continue;
                }
                SharedMeshCache[meshPath] = mesh;
            }

            GameObject meshObject = new GameObject(geometryInfo.Name);
            meshObject.transform.SetParent(bodyTransform, false);
            meshObject.transform.localPosition = geometryInfo.LocalTranslation;
            meshObject.transform.localRotation = geometryInfo.LocalRotation;
            meshObject.transform.localScale = geometryInfo.Scale;

            MeshFilter meshFilter = meshObject.AddComponent<MeshFilter>();
            meshFilter.sharedMesh = mesh;

            MeshRenderer meshRenderer = meshObject.AddComponent<MeshRenderer>();
            Material material = GetOrCreateBodyMaterial(runtime, geometryInfo.BodyName, geometryInfo.Color, geometryInfo.Opacity);
            meshRenderer.sharedMaterial = material;

            runtime.BodyMeshRenderers.Add(meshRenderer);
            runtime.LoadedMeshCount++;
            if (!runtime.BodyMeshMaterials.ContainsKey(geometryInfo.BodyName))
            {
                runtime.BodyMeshMaterials[geometryInfo.BodyName] = new List<Material>();
            }
            runtime.BodyMeshMaterials[geometryInfo.BodyName].Add(material);
        }
    }

    private Material GetOrCreateBodyMaterial(ModelRuntime runtime, string bodyName, Color color, float opacity)
    {
        Material material;
        if (runtime.BodyPrimaryMaterials.TryGetValue(bodyName, out material))
        {
            return material;
        }

        Color finalColor = color;
        finalColor.a = opacity;
        material = CreateMaterial(runtime.DisplayName + "_" + bodyName + "_mesh", finalColor);
        runtime.BodyPrimaryMaterials[bodyName] = material;
        return material;
    }

    private string ResolveMeshPath(string meshFile, string geometryPath, string osimDirectory)
    {
        if (string.IsNullOrWhiteSpace(meshFile))
        {
            return string.Empty;
        }

        string trimmed = meshFile.Trim().Replace('\\', Path.DirectorySeparatorChar).Replace('/', Path.DirectorySeparatorChar);
        if (Path.IsPathRooted(trimmed))
        {
            return trimmed;
        }

        string candidate = Path.Combine(geometryPath, Path.GetFileName(trimmed));
        if (File.Exists(candidate))
        {
            return candidate;
        }

        candidate = Path.Combine(geometryPath, trimmed);
        if (File.Exists(candidate))
        {
            return candidate;
        }

        candidate = Path.Combine(osimDirectory ?? string.Empty, trimmed);
        if (File.Exists(candidate))
        {
            return candidate;
        }

        candidate = Path.Combine(osimDirectory ?? string.Empty, "Geometry", Path.GetFileName(trimmed));
        return candidate;
    }

    private void CreateSkeletonLines(ModelRuntime runtime)
    {
        if (!showDebugJointSkeleton)
        {
            return;
        }

        foreach (JointInfo joint in runtime.ParsedModel.Joints)
        {
            if (string.IsNullOrEmpty(joint.ParentBody) || string.IsNullOrEmpty(joint.ChildBody))
            {
                continue;
            }

            if (!runtime.BodyTransforms.ContainsKey(joint.ParentBody) || !runtime.BodyTransforms.ContainsKey(joint.ChildBody))
            {
                continue;
            }

            GameObject lineObject = new GameObject(joint.Name + "_line");
            lineObject.transform.SetParent(runtime.Root, false);

            LineRenderer line = lineObject.AddComponent<LineRenderer>();
            line.sharedMaterial = skeletonMaterial;
            line.useWorldSpace = true;
            line.positionCount = 2;
            line.startWidth = boneLineWidth;
            line.endWidth = boneLineWidth;
            line.numCapVertices = 4;

            runtime.BoneLines.Add(new BoneLine
            {
                Renderer = line,
                Parent = runtime.BodyTransforms[joint.ParentBody],
                Child = runtime.BodyTransforms[joint.ChildBody]
            });
        }
    }

    private void CreateSpineHeatSegments(ModelRuntime runtime)
    {
        foreach (string level in SpineLevels)
        {
            JointInfo joint;
            if (!runtime.ParsedModel.JointsByName.TryGetValue(level + "_IVDjnt", out joint))
            {
                continue;
            }

            if (string.IsNullOrEmpty(joint.ParentBody) || string.IsNullOrEmpty(joint.ChildBody))
            {
                continue;
            }

            if (!runtime.BodyTransforms.ContainsKey(joint.ParentBody) || !runtime.BodyTransforms.ContainsKey(joint.ChildBody))
            {
                continue;
            }

            Transform segmentTransform = null;
            Material overlayMaterial = null;
            if (showSpineHeatOverlay)
            {
                GameObject segment = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                segment.name = level + "_heat_segment";
                segment.transform.SetParent(runtime.Root, false);
                segmentTransform = segment.transform;

                Renderer renderer = segment.GetComponent<Renderer>();
                overlayMaterial = CreateMaterial(runtime.DisplayName + "_" + level + "_heat", lowRotationColor);
                if (renderer != null)
                {
                    renderer.sharedMaterial = overlayMaterial;
                }

                Collider collider = segment.GetComponent<Collider>();
                if (collider != null)
                {
                    Destroy(collider);
                }
            }

            string coloredBody = SpineLevelToColoredBody.ContainsKey(level) ? SpineLevelToColoredBody[level] : joint.ChildBody;
            List<Material> meshMaterials = new List<Material>();
            List<Material> foundMaterials;
            if (runtime.BodyMeshMaterials.TryGetValue(coloredBody, out foundMaterials))
            {
                meshMaterials.AddRange(foundMaterials);
            }

            runtime.SpineSegments[level] = new SpineSegment
            {
                Level = level,
                Joint = joint,
                Parent = runtime.BodyTransforms[joint.ParentBody],
                Child = runtime.BodyTransforms[joint.ChildBody],
                ColoredBody = coloredBody,
                SegmentTransform = segmentTransform,
                OverlayMaterial = overlayMaterial,
                MeshMaterials = meshMaterials
            };
        }
    }

    private void ApplyPose(ModelRuntime runtime, float time)
    {
        Dictionary<string, float> sample = runtime.Motion.Sample(time);

        foreach (JointInfo joint in runtime.ParsedModel.Joints)
        {
            if (string.IsNullOrEmpty(joint.ChildBody))
            {
                continue;
            }

            Transform child;
            if (!runtime.BodyTransforms.TryGetValue(joint.ChildBody, out child))
            {
                continue;
            }

            Vector3 translation = Vector3.zero;
            Quaternion rotation = Quaternion.identity;

            for (int i = 0; i < joint.TransformAxes.Count; i++)
            {
                TransformAxis axis = joint.TransformAxes[i];
                float coordinateValue = CoordinateValueForFunction(runtime, joint, axis, sample);
                float displacement = axis.Function == null ? coordinateValue : axis.Function.Evaluate(coordinateValue);

                if (axis.IsRotation)
                {
                    rotation *= Quaternion.AngleAxis(displacement * Mathf.Rad2Deg, ToUnity(axis.Axis).normalized);
                }
                else
                {
                    translation += ToUnity(axis.Axis) * displacement;
                }
            }

            ComposeJointLocalTransform(joint, rotation, translation, out Vector3 localPosition, out Quaternion localRotation);
            child.localPosition = localPosition;
            child.localRotation = localRotation;
        }

        UpdateSkeletonLines(runtime);
    }

    private static float CoordinateValueForFunction(ModelRuntime runtime, JointInfo joint, TransformAxis axis, Dictionary<string, float> sample)
    {
        if (string.IsNullOrEmpty(axis.Coordinate))
        {
            return 0.0f;
        }

        float value;
        bool fromMotion = sample.TryGetValue(axis.Coordinate, out value);
        if (!fromMotion && !joint.CoordinateDefaults.TryGetValue(axis.Coordinate, out value))
        {
            value = 0.0f;
        }

        if (fromMotion && runtime.Motion.InDegrees && joint.RotationalCoordinates.Contains(axis.Coordinate))
        {
            value *= Mathf.Deg2Rad;
        }

        return value;
    }

    private void UpdateModel(ModelRuntime runtime, float time)
    {
        ApplyPose(runtime, time);
        UpdateHeatmap(runtime);
    }

    private void CaptureReferenceSpineRotations(ModelRuntime runtime)
    {
        runtime.ReferenceSpineRotations.Clear();
        foreach (KeyValuePair<string, SpineSegment> pair in runtime.SpineSegments)
        {
            runtime.ReferenceSpineRotations[pair.Key] = pair.Value.Joint.ChildTransformLocalRotation(runtime);
        }
    }

    private void UpdateHeatmap(ModelRuntime runtime)
    {
        float minAngle = float.PositiveInfinity;
        float maxAngle = 0.0f;

        foreach (SpineSegment segment in runtime.SpineSegments.Values)
        {
            Quaternion reference;
            if (!runtime.ReferenceSpineRotations.TryGetValue(segment.Level, out reference))
            {
                reference = segment.Joint.ChildTransformLocalRotation(runtime);
                runtime.ReferenceSpineRotations[segment.Level] = reference;
            }

            Quaternion current = segment.Joint.ChildTransformLocalRotation(runtime);
            float angle = Quaternion.Angle(reference, current);
            segment.CurrentRotationDegrees = angle;
            minAngle = Mathf.Min(minAngle, angle);
            maxAngle = Mathf.Max(maxAngle, angle);
        }

        if (float.IsNaN(minAngle) || float.IsInfinity(minAngle))
        {
            minAngle = 0.0f;
        }

        float span = Mathf.Max(maxAngle - minAngle, 0.0001f);
        float temporalGate = Mathf.Sqrt(Mathf.Clamp01(maxAngle / Mathf.Max(maxRotationForFullHeat, 0.0001f)));

        foreach (SpineSegment segment in runtime.SpineSegments.Values)
        {
            float normalized;
            if (useFrameRelativeContrast)
            {
                normalized = ((segment.CurrentRotationDegrees - minAngle) / span) * temporalGate;
            }
            else
            {
                normalized = segment.CurrentRotationDegrees / Mathf.Max(maxRotationForFullHeat, 0.0001f);
            }

            normalized = Mathf.Clamp01(normalized);
            Color color = EvaluateHeatColor(normalized);
            for (int i = 0; i < segment.MeshMaterials.Count; i++)
            {
                segment.MeshMaterials[i].color = color;
            }

            if (segment.OverlayMaterial != null)
            {
                segment.OverlayMaterial.color = color;
            }

            if (segment.SegmentTransform != null)
            {
                UpdateCylinder(segment.SegmentTransform, segment.Parent.position, segment.Child.position, spineSegmentRadius);
            }
        }
    }

    private void UpdateSkeletonLines(ModelRuntime runtime)
    {
        for (int i = 0; i < runtime.BoneLines.Count; i++)
        {
            BoneLine line = runtime.BoneLines[i];
            if (line.Renderer == null || line.Parent == null || line.Child == null)
            {
                continue;
            }

            line.Renderer.SetPosition(0, line.Parent.position);
            line.Renderer.SetPosition(1, line.Child.position);
        }
    }

    private void UpdateCylinder(Transform cylinder, Vector3 start, Vector3 end, float radius)
    {
        Vector3 delta = end - start;
        float length = delta.magnitude;
        if (length < 0.0001f)
        {
            cylinder.gameObject.SetActive(false);
            return;
        }

        cylinder.gameObject.SetActive(true);
        cylinder.position = (start + end) * 0.5f;
        cylinder.rotation = Quaternion.FromToRotation(Vector3.up, delta / length);
        cylinder.localScale = new Vector3(radius, length * 0.5f, radius);
    }

    private void ClearGeneratedChildren()
    {
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            Transform child = transform.GetChild(i);
            if (Application.isPlaying)
            {
                Destroy(child.gameObject);
            }
            else
            {
                DestroyImmediate(child.gameObject);
            }
        }
    }

    public void ClearGeneratedVisualizers()
    {
        ClearGeneratedChildren();
        models.Clear();
        playTime = 0.0f;
        statusMessage = "Cleared OpenSim visualizers.";
    }

    private string ResolvePath(string configuredPath)
    {
        if (Path.IsPathRooted(configuredPath))
        {
            return configuredPath;
        }

        string normalized = configuredPath.Replace('\\', '/');
        if (normalized.StartsWith("Assets/", StringComparison.OrdinalIgnoreCase))
        {
            normalized = normalized.Substring("Assets/".Length);
        }

        return Path.Combine(Application.dataPath, normalized);
    }

    private Material CreateMaterial(string materialName, Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }
        if (shader == null)
        {
            shader = Shader.Find("Sprites/Default");
        }

        Material material = new Material(shader);
        material.name = materialName;
        material.color = color;
        return material;
    }

    private Color EvaluateHeatColor(float value)
    {
        value = Mathf.Clamp01(value);
        if (value < 0.5f)
        {
            return Color.Lerp(lowRotationColor, middleRotationColor, value * 2.0f);
        }

        return Color.Lerp(middleRotationColor, highRotationColor, (value - 0.5f) * 2.0f);
    }

    private static Vector3 ToUnity(Vector3 value)
    {
        return value;
    }

    private static Quaternion EulerRadians(Vector3 radians)
    {
        return Quaternion.AngleAxis(radians.z * Mathf.Rad2Deg, Vector3.forward)
            * Quaternion.AngleAxis(radians.y * Mathf.Rad2Deg, Vector3.up)
            * Quaternion.AngleAxis(radians.x * Mathf.Rad2Deg, Vector3.right);
    }

    private static void ComposeJointLocalTransform(JointInfo joint, Quaternion jointRotation, Vector3 jointTranslation, out Vector3 localPosition, out Quaternion localRotation)
    {
        Quaternion parentFrameRotation = EulerRadians(joint.ParentFrameOrientation);
        Quaternion childFrameRotation = EulerRadians(joint.ChildFrameOrientation);

        localRotation = parentFrameRotation * jointRotation * Quaternion.Inverse(childFrameRotation);
        localPosition = ToUnity(joint.ParentFrameTranslation)
            + parentFrameRotation * jointTranslation
            - localRotation * ToUnity(joint.ChildFrameTranslation);
    }

    private static TransformFunction ParseTransformFunction(XElement axisElement)
    {
        XElement functionElement = FirstFunctionElement(axisElement);
        return functionElement == null ? null : ParseFunctionElement(functionElement);
    }

    private static XElement FirstFunctionElement(XElement parent)
    {
        if (parent == null)
        {
            return null;
        }

        foreach (XElement child in parent.Elements())
        {
            if (IsFunctionElement(child))
            {
                return child;
            }
        }

        return null;
    }

    private static bool IsFunctionElement(XElement element)
    {
        string name = LocalName(element);
        return name == "Constant"
            || name == "LinearFunction"
            || name == "SimmSpline"
            || name == "MultiplierFunction"
            || name == "PolynomialFunction";
    }

    private static TransformFunction ParseFunctionElement(XElement element)
    {
        if (element == null)
        {
            return null;
        }

        string name = LocalName(element);
        if (name == "Constant")
        {
            return new ConstantTransformFunction(ParseFloat(ChildText(element, "value")));
        }

        if (name == "LinearFunction")
        {
            float[] coefficients = ParseFloatList(ChildText(element, "coefficients"));
            float slope = coefficients.Length > 0 ? coefficients[0] : 1.0f;
            float intercept = coefficients.Length > 1 ? coefficients[1] : 0.0f;
            return new LinearTransformFunction(slope, intercept);
        }

        if (name == "SimmSpline")
        {
            return new SimmSplineTransformFunction(ParseFloatList(ChildText(element, "x")), ParseFloatList(ChildText(element, "y")));
        }

        if (name == "MultiplierFunction")
        {
            string scaleText = ChildText(element, "scale");
            float scale = string.IsNullOrWhiteSpace(scaleText) ? 1.0f : ParseFloat(scaleText);
            XElement wrapper = element.Elements().FirstOrDefault(e => LocalName(e) == "function");
            TransformFunction inner = ParseFunctionElement(FirstFunctionElement(wrapper));
            return new MultiplierTransformFunction(inner, scale);
        }

        if (name == "PolynomialFunction")
        {
            return new PolynomialTransformFunction(ParseFloatList(ChildText(element, "coefficients")));
        }

        return null;
    }

    private void OnGUI()
    {
        if (!showOverlay)
        {
            return;
        }

        GUILayout.BeginArea(new Rect(12, 12, 520, 180), GUI.skin.box);
        GUILayout.Label("OpenSim FullBody Runtime Visualizer");
        GUILayout.Label(statusMessage);
        GUILayout.Label("Time: " + playTime.ToString("0.000", CultureInfo.InvariantCulture) + " s");
        GUILayout.Label("Space: play/pause   R: restart   C: toggle frame-relative contrast");

        if (models.Count > 0)
        {
            ModelRuntime model = models[0];
            string line = "Male spine angles: ";
            int shown = 0;
            foreach (string level in SpineLevels)
            {
                SpineSegment segment;
                if (model.SpineSegments.TryGetValue(level, out segment))
                {
                    if (shown > 0)
                    {
                        line += "  ";
                    }
                    line += level + "=" + segment.CurrentRotationDegrees.ToString("0.00", CultureInfo.InvariantCulture);
                    shown++;
                    if (shown >= 4)
                    {
                        break;
                    }
                }
            }
            GUILayout.Label(line);
        }

        GUILayout.EndArea();
    }

    private void OnValidate()
    {
        playbackSpeed = Mathf.Max(0.0f, playbackSpeed);
        modelScale = Mathf.Max(0.001f, modelScale);
        bodyNodeRadius = Mathf.Max(0.001f, bodyNodeRadius);
        boneLineWidth = Mathf.Max(0.001f, boneLineWidth);
        spineSegmentRadius = Mathf.Max(0.001f, spineSegmentRadius);
        maxRotationForFullHeat = Mathf.Max(0.001f, maxRotationForFullHeat);
    }

    private void OnApplicationFocus(bool hasFocus)
    {
        if (!hasFocus)
        {
            return;
        }
    }

    private void LateUpdate()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isPlaying = !isPlaying;
        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            playTime = 0.0f;
        }
        if (Input.GetKeyDown(KeyCode.C))
        {
            useFrameRelativeContrast = !useFrameRelativeContrast;
        }
    }

    private class ModelRuntime
    {
        public string DisplayName;
        public Transform Root;
        public OpenSimModel ParsedModel;
        public MotionStorage Motion;
        public readonly Dictionary<string, Transform> BodyTransforms = new Dictionary<string, Transform>();
        public readonly Dictionary<string, JointInfo> JointsByChildBody = new Dictionary<string, JointInfo>();
        public readonly Dictionary<string, JointInfo> JointsByCoordinate = new Dictionary<string, JointInfo>();
        public readonly List<BoneLine> BoneLines = new List<BoneLine>();
        public readonly List<MeshRenderer> BodyMeshRenderers = new List<MeshRenderer>();
        public readonly Dictionary<string, Material> BodyPrimaryMaterials = new Dictionary<string, Material>();
        public readonly Dictionary<string, List<Material>> BodyMeshMaterials = new Dictionary<string, List<Material>>();
        public readonly Dictionary<string, SpineSegment> SpineSegments = new Dictionary<string, SpineSegment>();
        public readonly Dictionary<string, Quaternion> ReferenceSpineRotations = new Dictionary<string, Quaternion>();
        public int LoadedMeshCount;
        public int SkippedMeshCount;
    }

    private class BoneLine
    {
        public LineRenderer Renderer;
        public Transform Parent;
        public Transform Child;
    }

    private class SpineSegment
    {
        public string Level;
        public JointInfo Joint;
        public Transform Parent;
        public Transform Child;
        public string ColoredBody;
        public Transform SegmentTransform;
        public Material OverlayMaterial;
        public List<Material> MeshMaterials = new List<Material>();
        public float CurrentRotationDegrees;
    }

    private class OpenSimModel
    {
        public readonly List<string> BodyNames = new List<string>();
        public readonly List<BodyGeometryInfo> BodyGeometry = new List<BodyGeometryInfo>();
        public readonly List<JointInfo> Joints = new List<JointInfo>();
        public readonly Dictionary<string, JointInfo> JointsByName = new Dictionary<string, JointInfo>();

        public static OpenSimModel Load(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("OpenSim model not found", path);
            }

            XDocument document = XDocument.Load(path);
            OpenSimModel model = new OpenSimModel();

            XElement bodySet = document.Descendants().FirstOrDefault(e => LocalName(e) == "BodySet");
            if (bodySet != null)
            {
                XElement objects = bodySet.Elements().FirstOrDefault(e => LocalName(e) == "objects");
                if (objects != null)
                {
                    foreach (XElement body in objects.Elements().Where(e => LocalName(e) == "Body"))
                    {
                        XAttribute name = body.Attribute("name");
                        if (name != null && !string.IsNullOrEmpty(name.Value))
                        {
                            model.BodyNames.Add(name.Value);
                            ParseBodyGeometry(name.Value, body, model.BodyGeometry);
                        }
                    }
                }
            }

            XElement jointSet = document.Descendants().FirstOrDefault(e => LocalName(e) == "JointSet");
            XElement jointObjects = jointSet == null ? null : jointSet.Elements().FirstOrDefault(e => LocalName(e) == "objects");
            if (jointObjects != null)
            {
                foreach (XElement jointElement in jointObjects.Elements())
                {
                    JointInfo joint = ParseJoint(jointElement);
                    if (string.IsNullOrEmpty(joint.Name))
                    {
                        continue;
                    }

                    model.Joints.Add(joint);
                    model.JointsByName[joint.Name] = joint;
                }
            }

            return model;
        }

        private static void ParseBodyGeometry(string bodyName, XElement bodyElement, List<BodyGeometryInfo> bodyGeometry)
        {
            XElement attached = bodyElement.Elements().FirstOrDefault(e => LocalName(e) == "attached_geometry");
            if (attached != null)
            {
                ParseAttachedGeometry(bodyName, attached, Vector3.zero, Quaternion.identity, bodyGeometry);
            }

            XElement components = bodyElement.Elements().FirstOrDefault(e => LocalName(e) == "components");
            if (components == null)
            {
                return;
            }

            foreach (XElement frameElement in components.Elements().Where(e => LocalName(e) == "PhysicalOffsetFrame"))
            {
                ParsePhysicalOffsetFrameGeometry(bodyName, frameElement, Vector3.zero, Quaternion.identity, bodyGeometry);
            }
        }

        private static void ParsePhysicalOffsetFrameGeometry(string bodyName, XElement frameElement, Vector3 parentTranslation, Quaternion parentRotation, List<BodyGeometryInfo> bodyGeometry)
        {
            Vector3 frameTranslation = parentTranslation + parentRotation * ToUnity(ParseVector(ChildText(frameElement, "translation")));
            Quaternion frameRotation = parentRotation * EulerRadians(ParseVector(ChildText(frameElement, "orientation")));

            XElement attached = frameElement.Elements().FirstOrDefault(e => LocalName(e) == "attached_geometry");
            if (attached != null)
            {
                ParseAttachedGeometry(bodyName, attached, frameTranslation, frameRotation, bodyGeometry);
            }

            XElement components = frameElement.Elements().FirstOrDefault(e => LocalName(e) == "components");
            if (components == null)
            {
                return;
            }

            foreach (XElement childFrame in components.Elements().Where(e => LocalName(e) == "PhysicalOffsetFrame"))
            {
                ParsePhysicalOffsetFrameGeometry(bodyName, childFrame, frameTranslation, frameRotation, bodyGeometry);
            }
        }

        private static void ParseAttachedGeometry(string bodyName, XElement attached, Vector3 localTranslation, Quaternion localRotation, List<BodyGeometryInfo> bodyGeometry)
        {
            foreach (XElement meshElement in attached.Elements().Where(e => LocalName(e) == "Mesh"))
            {
                string meshFile = ChildText(meshElement, "mesh_file").Trim();
                if (string.IsNullOrEmpty(meshFile))
                {
                    continue;
                }

                XElement appearance = meshElement.Elements().FirstOrDefault(e => LocalName(e) == "Appearance");
                Color color = new Color(0.91f, 0.85f, 0.78f, 1.0f);
                float opacity = 1.0f;
                if (appearance != null)
                {
                    Vector3 colorVector = ParseVector(ChildText(appearance, "color"));
                    if (colorVector.sqrMagnitude > 0.000001f)
                    {
                        color = new Color(colorVector.x, colorVector.y, colorVector.z, 1.0f);
                    }
                    string opacityText = ChildText(appearance, "opacity");
                    if (!string.IsNullOrWhiteSpace(opacityText))
                    {
                        opacity = Mathf.Clamp01(ParseFloat(opacityText));
                    }
                }

                Vector3 scale = ParseVector(ChildText(meshElement, "scale_factors"));
                if (scale.sqrMagnitude < 0.000001f)
                {
                    scale = Vector3.one;
                }

                bodyGeometry.Add(new BodyGeometryInfo
                {
                    BodyName = bodyName,
                    Name = AttributeValue(meshElement, "name"),
                    MeshFile = meshFile,
                    Scale = scale,
                    LocalTranslation = localTranslation,
                    LocalRotation = localRotation,
                    Color = color,
                    Opacity = opacity
                });
            }
        }

        private static JointInfo ParseJoint(XElement jointElement)
        {
            JointInfo joint = new JointInfo
            {
                Name = AttributeValue(jointElement, "name"),
                JointType = LocalName(jointElement)
            };

            Dictionary<string, FrameInfo> frames = new Dictionary<string, FrameInfo>();
            XElement framesElement = jointElement.Elements().FirstOrDefault(e => LocalName(e) == "frames");
            if (framesElement != null)
            {
                foreach (XElement frameElement in framesElement.Elements().Where(e => LocalName(e) == "PhysicalOffsetFrame"))
                {
                    string frameName = AttributeValue(frameElement, "name");
                    if (string.IsNullOrEmpty(frameName))
                    {
                        continue;
                    }

                    frames[frameName] = new FrameInfo
                    {
                        BodyName = BodyFromPath(ChildText(frameElement, "socket_parent")),
                        Translation = ParseVector(ChildText(frameElement, "translation")),
                        Orientation = ParseVector(ChildText(frameElement, "orientation"))
                    };
                }
            }

            FrameInfo parentFrame = ResolveFrame(ChildText(jointElement, "socket_parent_frame"), frames);
            FrameInfo childFrame = ResolveFrame(ChildText(jointElement, "socket_child_frame"), frames);

            joint.ParentBody = parentFrame.BodyName;
            joint.ChildBody = childFrame.BodyName;
            joint.ParentFrameTranslation = parentFrame.Translation;
            joint.ChildFrameTranslation = childFrame.Translation;
            joint.ParentFrameOrientation = parentFrame.Orientation;
            joint.ChildFrameOrientation = childFrame.Orientation;

            XElement coordinatesElement = jointElement.Elements().FirstOrDefault(e => LocalName(e) == "coordinates");
            if (coordinatesElement != null)
            {
                foreach (XElement coordinate in coordinatesElement.Elements().Where(e => LocalName(e) == "Coordinate"))
                {
                    string name = AttributeValue(coordinate, "name");
                    if (!string.IsNullOrEmpty(name))
                    {
                        joint.Coordinates.Add(name);
                        joint.CoordinateDefaults[name] = ParseFloat(ChildText(coordinate, "default_value"));
                    }
                }
            }

            XElement spatialTransform = jointElement.Elements().FirstOrDefault(e => LocalName(e) == "SpatialTransform");
            if (spatialTransform != null)
            {
                foreach (XElement axisElement in spatialTransform.Elements().Where(e => LocalName(e) == "TransformAxis"))
                {
                    string axisName = AttributeValue(axisElement, "name");
                    string coordinate = ChildText(axisElement, "coordinates").Trim();
                    Vector3 axis = ParseVector(ChildText(axisElement, "axis"));
                    if (axis.sqrMagnitude < 0.000001f)
                    {
                        continue;
                    }

                    joint.TransformAxes.Add(new TransformAxis
                    {
                        Name = axisName,
                        Coordinate = coordinate,
                        Axis = axis,
                        IsRotation = axisName.StartsWith("rotation", StringComparison.OrdinalIgnoreCase),
                        Function = ParseTransformFunction(axisElement)
                    });

                    if (axisName.StartsWith("rotation", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(coordinate))
                    {
                        joint.RotationalCoordinates.Add(coordinate);
                    }
                }
            }

            return joint;
        }

        private static FrameInfo ResolveFrame(string socket, Dictionary<string, FrameInfo> frames)
        {
            string trimmed = (socket ?? string.Empty).Trim();
            if (frames.ContainsKey(trimmed))
            {
                return frames[trimmed];
            }

            if (trimmed == "ground" || trimmed == "/ground")
            {
                return new FrameInfo();
            }

            return new FrameInfo
            {
                BodyName = BodyFromPath(trimmed)
            };
        }
    }

    private class JointInfo
    {
        public string Name;
        public string JointType;
        public string ParentBody;
        public string ChildBody;
        public Vector3 ParentFrameTranslation;
        public Vector3 ChildFrameTranslation;
        public Vector3 ParentFrameOrientation;
        public Vector3 ChildFrameOrientation;
        public Vector3 BaseLocalPosition;
        public Quaternion BaseLocalRotation = Quaternion.identity;
        public readonly List<string> Coordinates = new List<string>();
        public readonly Dictionary<string, float> CoordinateDefaults = new Dictionary<string, float>();
        public readonly HashSet<string> RotationalCoordinates = new HashSet<string>();
        public readonly List<TransformAxis> TransformAxes = new List<TransformAxis>();

        public Quaternion ChildTransformLocalRotation(ModelRuntime runtime)
        {
            Transform child;
            if (runtime.BodyTransforms.TryGetValue(ChildBody, out child))
            {
                return child.localRotation;
            }

            return Quaternion.identity;
        }
    }

    private struct FrameInfo
    {
        public string BodyName;
        public Vector3 Translation;
        public Vector3 Orientation;
    }

    private struct TransformAxis
    {
        public string Name;
        public string Coordinate;
        public Vector3 Axis;
        public bool IsRotation;
        public TransformFunction Function;
    }

    private abstract class TransformFunction
    {
        public abstract float Evaluate(float x);
    }

    private class ConstantTransformFunction : TransformFunction
    {
        private readonly float value;

        public ConstantTransformFunction(float value)
        {
            this.value = value;
        }

        public override float Evaluate(float x)
        {
            return value;
        }
    }

    private class LinearTransformFunction : TransformFunction
    {
        private readonly float slope;
        private readonly float intercept;

        public LinearTransformFunction(float slope, float intercept)
        {
            this.slope = slope;
            this.intercept = intercept;
        }

        public override float Evaluate(float x)
        {
            return slope * x + intercept;
        }
    }

    private class MultiplierTransformFunction : TransformFunction
    {
        private readonly TransformFunction inner;
        private readonly float scale;

        public MultiplierTransformFunction(TransformFunction inner, float scale)
        {
            this.inner = inner;
            this.scale = scale;
        }

        public override float Evaluate(float x)
        {
            return scale * (inner == null ? 0.0f : inner.Evaluate(x));
        }
    }

    private class SimmSplineTransformFunction : TransformFunction
    {
        private readonly float[] xs;
        private readonly float[] ys;

        public SimmSplineTransformFunction(float[] xs, float[] ys)
        {
            this.xs = xs ?? Array.Empty<float>();
            this.ys = ys ?? Array.Empty<float>();
        }

        public override float Evaluate(float x)
        {
            int count = Mathf.Min(xs.Length, ys.Length);
            if (count == 0)
            {
                return 0.0f;
            }

            if (count == 1 || x <= xs[0])
            {
                return ys[0];
            }

            if (x >= xs[count - 1])
            {
                return ys[count - 1];
            }

            for (int i = 1; i < count; i++)
            {
                if (x <= xs[i])
                {
                    float span = Mathf.Max(xs[i] - xs[i - 1], 0.000001f);
                    float t = Mathf.Clamp01((x - xs[i - 1]) / span);
                    return Mathf.Lerp(ys[i - 1], ys[i], t);
                }
            }

            return ys[count - 1];
        }
    }

    private class PolynomialTransformFunction : TransformFunction
    {
        private readonly float[] coefficients;

        public PolynomialTransformFunction(float[] coefficients)
        {
            this.coefficients = coefficients ?? Array.Empty<float>();
        }

        public override float Evaluate(float x)
        {
            float result = 0.0f;
            float power = 1.0f;
            for (int i = 0; i < coefficients.Length; i++)
            {
                result += coefficients[i] * power;
                power *= x;
            }

            return result;
        }
    }

    private class BodyGeometryInfo
    {
        public string BodyName;
        public string Name;
        public string MeshFile;
        public Vector3 Scale = Vector3.one;
        public Vector3 LocalTranslation;
        public Quaternion LocalRotation = Quaternion.identity;
        public Color Color = Color.white;
        public float Opacity = 1.0f;
    }

    private static class OpenSimMeshLoader
    {
        public static Mesh Load(string path)
        {
            string extension = Path.GetExtension(path).ToLowerInvariant();
            if (extension == ".vtp")
            {
                return LoadVtp(path);
            }
            if (extension == ".obj")
            {
                return LoadObj(path);
            }

            throw new NotSupportedException("Unsupported OpenSim mesh extension: " + extension);
        }

        private static Mesh LoadVtp(string path)
        {
            XDocument document = XDocument.Load(path);
            XElement piece = document.Descendants().FirstOrDefault(e => LocalName(e) == "Piece");
            if (piece == null)
            {
                throw new InvalidDataException("VTP file has no Piece element: " + path);
            }

            XElement pointsElement = piece.Elements().FirstOrDefault(e => LocalName(e) == "Points");
            XElement pointData = pointsElement == null
                ? null
                : pointsElement.Descendants().FirstOrDefault(e => LocalName(e) == "DataArray");
            if (pointData == null)
            {
                throw new InvalidDataException("VTP file has no point DataArray: " + path);
            }

            float[] pointValues = ParseFloatArray(pointData.Value);
            Vector3[] vertices = new Vector3[pointValues.Length / 3];
            for (int i = 0; i < vertices.Length; i++)
            {
                vertices[i] = ToUnity(new Vector3(pointValues[i * 3], pointValues[i * 3 + 1], pointValues[i * 3 + 2]));
            }

            XElement polysElement = piece.Elements().FirstOrDefault(e => LocalName(e) == "Polys");
            if (polysElement == null)
            {
                throw new InvalidDataException("VTP file has no Polys element: " + path);
            }

            XElement connectivityElement = polysElement.Elements()
                .FirstOrDefault(e => LocalName(e) == "DataArray" && AttributeValue(e, "Name").Equals("connectivity", StringComparison.OrdinalIgnoreCase));
            XElement offsetsElement = polysElement.Elements()
                .FirstOrDefault(e => LocalName(e) == "DataArray" && AttributeValue(e, "Name").Equals("offsets", StringComparison.OrdinalIgnoreCase));

            if (connectivityElement == null || offsetsElement == null)
            {
                throw new InvalidDataException("VTP file has no polygon connectivity/offsets: " + path);
            }

            int[] connectivity = ParseIntArray(connectivityElement.Value);
            int[] offsets = ParseIntArray(offsetsElement.Value);
            List<int> triangles = new List<int>(connectivity.Length);
            int start = 0;
            for (int i = 0; i < offsets.Length; i++)
            {
                int end = Mathf.Clamp(offsets[i], 0, connectivity.Length);
                int count = end - start;
                if (count >= 3)
                {
                    int first = connectivity[start];
                    for (int j = 1; j < count - 1; j++)
                    {
                        triangles.Add(first);
                        triangles.Add(connectivity[start + j]);
                        triangles.Add(connectivity[start + j + 1]);
                    }
                }
                start = end;
            }

            Mesh mesh = new Mesh();
            mesh.name = Path.GetFileNameWithoutExtension(path);
            if (vertices.Length > 65535)
            {
                mesh.indexFormat = IndexFormat.UInt32;
            }
            mesh.vertices = vertices;
            mesh.triangles = triangles.ToArray();
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static Mesh LoadObj(string path)
        {
            List<Vector3> sourceVertices = new List<Vector3>();
            List<Vector3> vertices = new List<Vector3>();
            List<int> triangles = new List<int>();

            string[] lines = File.ReadAllLines(path);
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                {
                    continue;
                }

                if (line.StartsWith("v ", StringComparison.Ordinal))
                {
                    string[] parts = SplitValues(line.Substring(2));
                    if (parts.Length >= 3)
                    {
                        sourceVertices.Add(ToUnity(new Vector3(ParseFloat(parts[0]), ParseFloat(parts[1]), ParseFloat(parts[2]))));
                    }
                }
                else if (line.StartsWith("f ", StringComparison.Ordinal))
                {
                    string[] parts = SplitValues(line.Substring(2));
                    if (parts.Length < 3)
                    {
                        continue;
                    }

                    int[] face = new int[parts.Length];
                    for (int j = 0; j < parts.Length; j++)
                    {
                        face[j] = ResolveObjVertexIndex(parts[j], sourceVertices.Count);
                    }

                    for (int j = 1; j < face.Length - 1; j++)
                    {
                        AddObjTriangle(sourceVertices, vertices, triangles, face[0], face[j], face[j + 1]);
                    }
                }
            }

            Mesh mesh = new Mesh();
            mesh.name = Path.GetFileNameWithoutExtension(path);
            if (vertices.Count > 65535)
            {
                mesh.indexFormat = IndexFormat.UInt32;
            }
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static void AddObjTriangle(List<Vector3> sourceVertices, List<Vector3> vertices, List<int> triangles, int a, int b, int c)
        {
            if (!IsValidIndex(sourceVertices, a) || !IsValidIndex(sourceVertices, b) || !IsValidIndex(sourceVertices, c))
            {
                return;
            }

            int start = vertices.Count;
            vertices.Add(sourceVertices[a]);
            vertices.Add(sourceVertices[b]);
            vertices.Add(sourceVertices[c]);
            triangles.Add(start);
            triangles.Add(start + 1);
            triangles.Add(start + 2);
        }

        private static bool IsValidIndex(List<Vector3> vertices, int index)
        {
            return index >= 0 && index < vertices.Count;
        }

        private static int ResolveObjVertexIndex(string token, int vertexCount)
        {
            string vertexPart = token.Split('/')[0];
            int objIndex;
            if (!int.TryParse(vertexPart, NumberStyles.Integer, CultureInfo.InvariantCulture, out objIndex))
            {
                return -1;
            }

            if (objIndex > 0)
            {
                return objIndex - 1;
            }

            if (objIndex < 0)
            {
                return vertexCount + objIndex;
            }

            return -1;
        }

        private static float[] ParseFloatArray(string text)
        {
            string[] parts = SplitValues(text);
            float[] values = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                values[i] = ParseFloat(parts[i]);
            }
            return values;
        }

        private static int[] ParseIntArray(string text)
        {
            string[] parts = SplitValues(text);
            int[] values = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                int.TryParse(parts[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out values[i]);
            }
            return values;
        }
    }

    private class MotionStorage
    {
        public bool InDegrees = true;
        public float StartTime;
        public float EndTime;
        public float Duration;
        private readonly List<string> labels = new List<string>();
        private readonly List<float[]> rows = new List<float[]>();
        private readonly Dictionary<string, int> labelToIndex = new Dictionary<string, int>();

        public static MotionStorage Load(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Motion file not found", path);
            }

            MotionStorage storage = new MotionStorage();
            string[] lines = File.ReadAllLines(path);
            int headerEnd = -1;
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (line.StartsWith("inDegrees", StringComparison.OrdinalIgnoreCase))
                {
                    storage.InDegrees = line.IndexOf("yes", StringComparison.OrdinalIgnoreCase) >= 0;
                }
                if (line.Equals("endheader", StringComparison.OrdinalIgnoreCase))
                {
                    headerEnd = i;
                    break;
                }
            }

            if (headerEnd < 0 || headerEnd + 1 >= lines.Length)
            {
                throw new InvalidDataException("Motion file is missing an OpenSim storage header: " + path);
            }

            storage.labels.AddRange(SplitValues(lines[headerEnd + 1]));
            for (int i = 0; i < storage.labels.Count; i++)
            {
                storage.labelToIndex[storage.labels[i]] = i;
            }

            for (int i = headerEnd + 2; i < lines.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(lines[i]))
                {
                    continue;
                }

                string[] parts = SplitValues(lines[i]);
                if (parts.Length < storage.labels.Count)
                {
                    continue;
                }

                float[] row = new float[storage.labels.Count];
                for (int j = 0; j < row.Length; j++)
                {
                    row[j] = ParseFloat(parts[j]);
                }

                storage.rows.Add(row);
            }

            if (storage.rows.Count == 0)
            {
                throw new InvalidDataException("Motion file has no samples: " + path);
            }

            storage.StartTime = storage.rows[0][0];
            storage.EndTime = storage.rows[storage.rows.Count - 1][0];
            storage.Duration = Mathf.Max(0.0f, storage.EndTime - storage.StartTime);
            return storage;
        }

        public Dictionary<string, float> Sample(float time)
        {
            float clampedTime = Mathf.Clamp(time, StartTime, EndTime);
            int next = 0;
            while (next < rows.Count && rows[next][0] < clampedTime)
            {
                next++;
            }

            if (next <= 0)
            {
                return RowToDictionary(rows[0]);
            }

            if (next >= rows.Count)
            {
                return RowToDictionary(rows[rows.Count - 1]);
            }

            float[] a = rows[next - 1];
            float[] b = rows[next];
            float span = Mathf.Max(b[0] - a[0], 0.000001f);
            float u = Mathf.Clamp01((clampedTime - a[0]) / span);

            Dictionary<string, float> sample = new Dictionary<string, float>(labels.Count);
            for (int i = 0; i < labels.Count; i++)
            {
                sample[labels[i]] = Mathf.Lerp(a[i], b[i], u);
            }

            return sample;
        }

        private Dictionary<string, float> RowToDictionary(float[] row)
        {
            Dictionary<string, float> sample = new Dictionary<string, float>(labels.Count);
            for (int i = 0; i < labels.Count; i++)
            {
                sample[labels[i]] = row[i];
            }
            return sample;
        }
    }

    private static string LocalName(XElement element)
    {
        return element.Name.LocalName;
    }

    private static string AttributeValue(XElement element, string attributeName)
    {
        XAttribute attribute = element.Attribute(attributeName);
        return attribute == null ? string.Empty : attribute.Value;
    }

    private static string ChildText(XElement element, string childName)
    {
        XElement child = element.Elements().FirstOrDefault(e => LocalName(e) == childName);
        return child == null ? string.Empty : child.Value;
    }

    private static string BodyFromPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return string.Empty;
        }

        string normalized = path.Trim().Replace('\\', '/');
        int bodySetIndex = normalized.IndexOf("/bodyset/", StringComparison.OrdinalIgnoreCase);
        if (bodySetIndex >= 0)
        {
            return normalized.Substring(bodySetIndex + "/bodyset/".Length).Split('/')[0];
        }

        if (normalized.StartsWith("bodyset/", StringComparison.OrdinalIgnoreCase))
        {
            return normalized.Substring("bodyset/".Length).Split('/')[0];
        }

        if (!normalized.Contains("/") && normalized != "ground")
        {
            return normalized;
        }

        return string.Empty;
    }

    private static Vector3 ParseVector(string text)
    {
        string[] values = SplitValues(text);
        if (values.Length < 3)
        {
            return Vector3.zero;
        }

        return new Vector3(ParseFloat(values[0]), ParseFloat(values[1]), ParseFloat(values[2]));
    }

    private static float[] ParseFloatList(string text)
    {
        string[] values = SplitValues(text);
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = ParseFloat(values[i]);
        }

        return result;
    }

    private static float ParseFloat(string text)
    {
        float value;
        if (float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
        {
            return value;
        }

        return 0.0f;
    }

    private static string[] SplitValues(string text)
    {
        return (text ?? string.Empty)
            .Trim()
            .Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
    }
}
