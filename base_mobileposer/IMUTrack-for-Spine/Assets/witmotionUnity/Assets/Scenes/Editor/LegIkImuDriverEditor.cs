using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(LegIkImuDriver))]
public class LegIkImuDriverEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        EditorGUILayout.Space();

        LegIkImuDriver driver = (LegIkImuDriver)target;

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Start UDP"))
            {
                driver.StartUdp();
            }

            if (GUILayout.Button("Send Loc"))
            {
                driver.SendLoc();
            }

            if (GUILayout.Button("Stop UDP"))
            {
                driver.StopUdp();
            }
        }

        using (new EditorGUI.DisabledScope(!Application.isPlaying))
        {
            if (GUILayout.Button("Calibrate Pose"))
            {
                driver.CalibratePose();
            }
        }

        if (!Application.isPlaying)
        {
            EditorGUILayout.HelpBox("Calibrate Pose samples each assigned IMU for 1 second in Play Mode and stores a per-IMU mounting offset against the current LegIK target pose.", MessageType.Info);
        }
    }
}
