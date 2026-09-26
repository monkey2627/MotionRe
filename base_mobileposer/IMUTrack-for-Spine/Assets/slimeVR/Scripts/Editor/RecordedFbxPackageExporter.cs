#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Animations;
using UnityEditor.Formats.Fbx.Exporter;
using UnityEngine;

/// <summary>
/// Exports one recorded Humanoid clip together with the avatar as an FBX.
/// The class is outside an Editor folder so IMUTrack editor tools can share it
/// without changing any scene or runtime behaviour.
/// </summary>
public static class RecordedFbxPackageExporter
{
    public static bool TryExportPackage(GameObject sourceAvatar, AnimationClip clip, string fbxPath, out string message)
    {
        if (sourceAvatar == null)
        {
            message = "No source avatar GameObject was supplied.";
            return false;
        }

        if (clip == null)
        {
            message = "No AnimationClip was supplied.";
            return false;
        }

        string directory = Path.GetDirectoryName(fbxPath);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        GameObject clone = null;
        string controllerPath = null;

        try
        {
            clone = UnityEngine.Object.Instantiate(sourceAvatar);
            // Copy-from-other Humanoid import matches the hierarchy by name.
            // Keep the original root name or the exported FBX loses its Avatar mapping.
            clone.name = sourceAvatar.name;
            clone.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            clone.transform.localScale = Vector3.one;

            var animator = clone.GetComponentInChildren<Animator>() ?? clone.AddComponent<Animator>();
            string assetFbxPath = ToAssetPath(fbxPath);
            string assetDirectory = string.IsNullOrEmpty(assetFbxPath)
                ? "Assets/Recordings"
                : Path.GetDirectoryName(assetFbxPath)?.Replace("\\", "/");
            if (string.IsNullOrEmpty(assetDirectory)) assetDirectory = "Assets/Recordings";

            Directory.CreateDirectory(Path.GetFullPath(Path.Combine(Application.dataPath, "..", assetDirectory)));
            controllerPath = AssetDatabase.GenerateUniqueAssetPath(
                $"{assetDirectory}/{Path.GetFileNameWithoutExtension(fbxPath)}_Export.controller");
            AnimatorController controller = AnimatorController.CreateAnimatorControllerAtPath(controllerPath);
            AnimatorState state = controller.layers[0].stateMachine.AddState(clip.name);
            state.motion = clip;
            controller.layers[0].stateMachine.defaultState = state;
            animator.runtimeAnimatorController = controller;

            var exportOptions = new ExportModelOptions
            {
                ModelAnimIncludeOption = Include.ModelAndAnim,
                AnimateSkinnedMesh = true,
                ObjectPosition = ObjectPosition.WorldAbsolute,
                ExportUnrendered = true,
                UseMayaCompatibleNames = false
            };

            ModelExporter.ExportObject(fbxPath, clone, exportOptions);
            ConfigureHumanoidImport(fbxPath, sourceAvatar);
            message = $"Exported FBX: {fbxPath}";
            return true;
        }
        catch (Exception exception)
        {
            message = exception.InnerException?.Message ?? exception.Message;
            return false;
        }
        finally
        {
            if (clone != null) UnityEngine.Object.DestroyImmediate(clone);
            if (!string.IsNullOrEmpty(controllerPath)) AssetDatabase.DeleteAsset(controllerPath);
        }
    }

    /// <summary>
    /// Makes the one FBX take expose one imported AnimationClip per keyframe
    /// interval. The boundary frame is shared by adjacent clips.
    /// </summary>
    public static bool ConfigureClipRanges(string fbxPath, string[] names, int[] firstFrames, int[] lastFrames,
        out string message)
    {
        message = null;
        string assetPath = ToAssetPath(fbxPath);
        if (string.IsNullOrEmpty(assetPath) || names == null || firstFrames == null || lastFrames == null ||
            names.Length == 0 || names.Length != firstFrames.Length || names.Length != lastFrames.Length)
        {
            message = "Invalid FBX clip range configuration.";
            return false;
        }

        AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
        var importer = AssetImporter.GetAtPath(assetPath) as ModelImporter;
        if (importer == null)
        {
            message = "The exported file is not a ModelImporter asset.";
            return false;
        }

        ModelImporterClipAnimation[] defaults = importer.defaultClipAnimations;
        string takeName = defaults != null && defaults.Length > 0 ? defaults[0].takeName : string.Empty;
        var clips = new ModelImporterClipAnimation[names.Length];
        for (int i = 0; i < clips.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(names[i]) || firstFrames[i] < 0 || lastFrames[i] <= firstFrames[i])
            {
                message = $"Invalid range for clip {i + 1}.";
                return false;
            }

            clips[i] = new ModelImporterClipAnimation
            {
                name = names[i],
                takeName = takeName,
                firstFrame = firstFrames[i],
                lastFrame = lastFrames[i],
                loop = false,
                loopPose = false,
                keepOriginalOrientation = true,
                keepOriginalPositionY = true,
                keepOriginalPositionXZ = true,
                heightFromFeet = false,
                lockRootRotation = false,
                lockRootHeightY = false,
                lockRootPositionXZ = false
            };
        }

        importer.importAnimation = true;
        importer.clipAnimations = clips;
        importer.SaveAndReimport();
        message = $"Configured {clips.Length} FBX clips.";
        return true;
    }

    public static string ToAssetPath(string absolutePath)
    {
        string projectPath = Path.GetFullPath(Path.Combine(Application.dataPath, ".."))
            .Replace("\\", "/");
        string normalized = Path.GetFullPath(absolutePath).Replace("\\", "/");
        if (!normalized.StartsWith(projectPath + "/Assets/", StringComparison.OrdinalIgnoreCase)) return string.Empty;
        return normalized.Substring(projectPath.Length + 1);
    }

    private static void ConfigureHumanoidImport(string fbxPath, GameObject sourceAvatar)
    {
        string assetPath = ToAssetPath(fbxPath);
        if (string.IsNullOrEmpty(assetPath)) return;

        AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceSynchronousImport);
        var importer = AssetImporter.GetAtPath(assetPath) as ModelImporter;
        if (importer == null) return;

        importer.importAnimation = true;
        importer.animationType = ModelImporterAnimationType.Human;
        importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
        importer.optimizeGameObjects = false;
        importer.resampleCurves = true;

        Animator sourceAnimator = sourceAvatar.GetComponentInChildren<Animator>();
        if (sourceAnimator != null && sourceAnimator.avatar != null &&
            sourceAnimator.avatar.isValid && sourceAnimator.avatar.isHuman)
        {
            importer.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
            importer.humanDescription = sourceAnimator.avatar.humanDescription;
        }

        importer.SaveAndReimport();
    }
}
#endif
