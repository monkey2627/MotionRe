using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

[CustomEditor(typeof(OpenSimFullBodyRuntimeVisualizer))]
public class OpenSimFullBodyRuntimeVisualizerEditor : Editor
{
    private static bool rebuildWhenBackInEditMode;

    static OpenSimFullBodyRuntimeVisualizerEditor()
    {
        EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
        EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
    }

    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        GUILayout.Space(8);
        OpenSimFullBodyRuntimeVisualizer visualizer = (OpenSimFullBodyRuntimeVisualizer)target;

        if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
        {
            EditorGUILayout.HelpBox("You are in Play Mode. Meshes created now will be discarded when Play Mode stops.", MessageType.Warning);
            if (GUILayout.Button("Exit Play Mode And Rebuild"))
            {
                rebuildWhenBackInEditMode = true;
                EditorApplication.isPlaying = false;
            }

            return;
        }

        if (GUILayout.Button("Rebuild OpenSim Meshes"))
        {
            EditorApplication.delayCall += () => RebuildInEditLoop(visualizer);
        }

        if (GUILayout.Button("Clear Generated Meshes"))
        {
            EditorApplication.delayCall += () => ClearInEditLoop(visualizer);
        }
    }

    private static void OnPlayModeStateChanged(PlayModeStateChange state)
    {
        if (state != PlayModeStateChange.EnteredEditMode || !rebuildWhenBackInEditMode)
        {
            return;
        }

        rebuildWhenBackInEditMode = false;
        EditorApplication.delayCall += RebuildCurrentSceneVisualizer;
    }

    private static void RebuildInEditLoop(OpenSimFullBodyRuntimeVisualizer visualizer)
    {
        if (visualizer == null)
        {
            return;
        }

        Undo.RegisterFullObjectHierarchyUndo(visualizer.gameObject, "Rebuild OpenSim Meshes");
        visualizer.BuildScene();
        EditorUtility.SetDirty(visualizer);
        EditorSceneManager.MarkSceneDirty(visualizer.gameObject.scene);
        EditorApplication.QueuePlayerLoopUpdate();
        SceneView.RepaintAll();
    }

    private static void ClearInEditLoop(OpenSimFullBodyRuntimeVisualizer visualizer)
    {
        if (visualizer == null)
        {
            return;
        }

        Undo.RegisterFullObjectHierarchyUndo(visualizer.gameObject, "Clear OpenSim Meshes");
        visualizer.ClearGeneratedVisualizers();
        EditorUtility.SetDirty(visualizer);
        EditorSceneManager.MarkSceneDirty(visualizer.gameObject.scene);
        EditorApplication.QueuePlayerLoopUpdate();
        SceneView.RepaintAll();
    }

    [MenuItem("Tools/OpenSim/Rebuild Visualizer In Current Scene")]
    private static void RebuildCurrentSceneVisualizer()
    {
        OpenSimFullBodyRuntimeVisualizer visualizer = Object.FindObjectOfType<OpenSimFullBodyRuntimeVisualizer>();
        RebuildInEditLoop(visualizer);
    }

    [MenuItem("Tools/OpenSim/Clear Visualizer In Current Scene")]
    private static void ClearCurrentSceneVisualizer()
    {
        OpenSimFullBodyRuntimeVisualizer visualizer = Object.FindObjectOfType<OpenSimFullBodyRuntimeVisualizer>();
        ClearInEditLoop(visualizer);
    }
}
