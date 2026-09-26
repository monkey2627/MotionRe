using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Leaves the independent IMUTrack module and returns focus to SpineFlowAdmin.
/// Recording and export are completed by IMUTrack before this method is called.
/// </summary>
public sealed class ReturnToAdminController : MonoBehaviour
{
    [Header("Admin Window")]
    [SerializeField] private string adminProcessName = "SpineFlowAdmin";
    [SerializeField] private string adminWindowTitleHint = "SpineFlowAdmin";
    [SerializeField] private Text statusText;

    public void FinishAndReturn()
    {
        SetStatus("Returning to SpineFlowAdmin.");

#if UNITY_EDITOR
        string processName = adminProcessName;
        string titleHint = adminWindowTitleHint;
        UnityEditor.EditorApplication.delayCall += delegate
        {
            FocusAdminWindow(processName, titleHint);
        };
        UnityEditor.EditorApplication.isPlaying = false;
#else
        FocusAdminWindow(adminProcessName, adminWindowTitleHint);
        Application.Quit();
#endif
    }

    private void SetStatus(string message)
    {
        if (statusText != null)
        {
            statusText.text = message;
        }
        UnityEngine.Debug.Log("[Return To Admin] " + message, this);
    }

    private static bool FocusAdminWindow(string processName, string titleHint)
    {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        if (!string.IsNullOrWhiteSpace(processName))
        {
            foreach (var process in Process.GetProcessesByName(processName))
            {
                if (TryFocusProcess(process)) return true;
            }
        }

        foreach (var process in Process.GetProcesses())
        {
            try
            {
                if (string.IsNullOrEmpty(process.MainWindowTitle)) continue;
                if (process.MainWindowTitle.IndexOf(titleHint, StringComparison.OrdinalIgnoreCase) < 0) continue;
                if (TryFocusProcess(process)) return true;
            }
            catch (Exception)
            {
                // Some system processes deny access. Keep searching.
            }
        }
#endif
        return false;
    }

    private static bool TryFocusProcess(Process process)
    {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        try
        {
            if (process.MainWindowHandle == IntPtr.Zero) return false;
            ShowWindow(process.MainWindowHandle, 9);
            SetForegroundWindow(process.MainWindowHandle);
            return true;
        }
        catch (Exception)
        {
            return false;
        }
#else
        return false;
#endif
    }

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr windowHandle);

    [DllImport("user32.dll")]
    private static extern bool ShowWindow(IntPtr windowHandle, int command);
#endif
}
