using System;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace SpineFlow.MobilePoser
{
    /// <summary>
    /// Connects to the local mobileposer realtime bridge
    /// (mobileposer/realtime/run.py + stream_out.py, ws://127.0.0.1:21200 by
    /// default) and drives a Humanoid Animator's bones directly from its SMPL
    /// pose stream. This is an alternative pose source alongside the existing
    /// SlimeVR VMC/EVMC4U path, not a replacement for it -- SlimeVR-Server
    /// keeps running unmodified, so both can be compared side by side (see the
    /// implementation plan's M6).
    ///
    /// Attach to the GameObject holding the Animator for the avatar to drive
    /// (e.g. wherever mesh_edit1.vrm's Animator lives), or assign `animator`
    /// explicitly in the inspector.
    /// </summary>
    public sealed class MobilePoserPoseSource : MonoBehaviour
    {
        [SerializeField] private Animator animator;
        [SerializeField] private string host = "127.0.0.1";
        [SerializeField] private int port = 21200;
        [Tooltip("If true, applies the received rotation as a delta on top of this " +
                 "avatar's own cached rest pose (t.rest * received). Turn off only if " +
                 "you have verified this rig's bone-local axes already match SMPL's.")]
        [SerializeField] private bool applyAsRestPoseDelta = true;

        // SMPL joint index -> HumanBodyBones, per mobileposer/fit_ours_smpl.py's
        // joint-index comment: 0 pelvis, 1/2 L/R hip, 3 spine1, 4/5 L/R knee,
        // 6 spine2, 7/8 L/R ankle, 9 spine3, 10/11 L/R foot, 12 neck, 13/14 L/R
        // collar, 15 head, 16/17 L/R shoulder, 18/19 L/R elbow, 20/21 L/R wrist,
        // 22/23 L/R hand. HumanBodyBones has no direct "hand tip" bone distinct
        // from LeftHand/RightHand, so 22/23 are skipped (LastBone sentinel).
        private static readonly HumanBodyBones[] JointToBone =
        {
            HumanBodyBones.Hips,               // 0  pelvis
            HumanBodyBones.LeftUpperLeg,        // 1  L hip
            HumanBodyBones.RightUpperLeg,       // 2  R hip
            HumanBodyBones.Spine,               // 3  spine1
            HumanBodyBones.LeftLowerLeg,        // 4  L knee
            HumanBodyBones.RightLowerLeg,       // 5  R knee
            HumanBodyBones.Chest,               // 6  spine2
            HumanBodyBones.LeftFoot,            // 7  L ankle
            HumanBodyBones.RightFoot,           // 8  R ankle
            HumanBodyBones.UpperChest,          // 9  spine3
            HumanBodyBones.LeftToes,            // 10 L foot
            HumanBodyBones.RightToes,           // 11 R foot
            HumanBodyBones.Neck,                // 12 neck
            HumanBodyBones.LeftShoulder,        // 13 L collar
            HumanBodyBones.RightShoulder,       // 14 R collar
            HumanBodyBones.Head,                // 15 head
            HumanBodyBones.LeftUpperArm,        // 16 L shoulder
            HumanBodyBones.RightUpperArm,       // 17 R shoulder
            HumanBodyBones.LeftLowerArm,        // 18 L elbow
            HumanBodyBones.RightLowerArm,       // 19 R elbow
            HumanBodyBones.LeftHand,            // 20 L wrist
            HumanBodyBones.RightHand,           // 21 R wrist
            HumanBodyBones.LastBone,            // 22 L hand -- no distinct Humanoid bone
            HumanBodyBones.LastBone,            // 23 R hand -- no distinct Humanoid bone
        };

        private const int JointCount = 24;

        private readonly Transform[] boneTransforms = new Transform[JointCount];
        private readonly Quaternion[] restLocalRotation = new Quaternion[JointCount];

        private CancellationTokenSource cancellation;
        private Task worker;
        private readonly object sync = new object();
        private float[] latestJoints; // flat, JointCount * 4, guarded by sync
        private string lastError = "Connecting to mobileposer realtime bridge...";

        public string CurrentStatus
        {
            get { lock (sync) return lastError ?? "Streaming mobileposer pose"; }
        }

        private void Awake()
        {
            if (animator == null) animator = GetComponent<Animator>();
        }

        private void Start()
        {
            CacheBoneTransformsAndRestPose();
            cancellation = new CancellationTokenSource();
            worker = Task.Run(() => RunConnectionLoopAsync(cancellation.Token));
        }

        private void OnDestroy()
        {
            cancellation?.Cancel();
            cancellation?.Dispose();
        }

        private void CacheBoneTransformsAndRestPose()
        {
            if (animator == null)
            {
                Debug.LogError("MobilePoserPoseSource: no Animator found/assigned.", this);
                return;
            }
            for (int i = 0; i < JointCount; i++)
            {
                if (JointToBone[i] == HumanBodyBones.LastBone) continue;
                Transform t = animator.GetBoneTransform(JointToBone[i]);
                boneTransforms[i] = t;
                if (t != null) restLocalRotation[i] = t.localRotation;
            }
        }

        private void Update()
        {
            float[] joints;
            lock (sync)
            {
                joints = latestJoints;
            }
            if (joints == null || joints.Length != JointCount * 4) return;

            // SMPL local rotations are deviations from the SMPL zero pose (all
            // identity at rest); this avatar's Humanoid bone localRotation is a
            // deviation from whatever rest/bind pose its own rig defines.
            // Applying the received rotation as a delta on top of this avatar's
            // cached rest pose (restLocalRotation[i] * received) is the standard
            // lightweight retarget for this situation -- it assumes SMPL's and
            // this rig's per-bone local axis conventions agree, which is NOT
            // guaranteed for every bone (shoulders/hips are the usual
            // troublemakers). Validate visually; if a specific bone looks
            // twisted, that bone's axis convention needs a per-bone correction
            // here, not a change to the whole scheme.
            for (int i = 0; i < JointCount; i++)
            {
                Transform t = boneTransforms[i];
                if (t == null) continue;

                int o = i * 4;
                float x = joints[o], y = joints[o + 1], z = joints[o + 2], w = joints[o + 3];
                float magnitudeSquared = x * x + y * y + z * z + w * w;
                if (float.IsNaN(magnitudeSquared) || float.IsInfinity(magnitudeSquared) ||
                    magnitudeSquared < 0.000001f)
                    continue;

                var received = new Quaternion(x, y, z, w);
                received.Normalize();
                t.localRotation = applyAsRestPoseDelta ? restLocalRotation[i] * received : received;
            }
        }

        private async Task RunConnectionLoopAsync(CancellationToken token)
        {
            string url = $"ws://{host}:{port}";
            while (!token.IsCancellationRequested)
            {
                try
                {
                    using (var socket = new ClientWebSocket())
                    {
                        await socket.ConnectAsync(new Uri(url), token).ConfigureAwait(false);
                        SetError(null);
                        await ReceiveLoopAsync(socket, token).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (token.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception exception)
                {
                    SetError("mobileposer realtime bridge connection failed: " + exception.Message);
                }

                try
                {
                    await Task.Delay(1000, token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    return;
                }
            }
        }

        private async Task ReceiveLoopAsync(ClientWebSocket socket, CancellationToken token)
        {
            var buffer = new byte[65536];
            while (!token.IsCancellationRequested && socket.State == WebSocketState.Open)
            {
                using (var message = new MemoryStream())
                {
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), token)
                            .ConfigureAwait(false);
                        if (result.MessageType == WebSocketMessageType.Close) return;
                        message.Write(buffer, 0, result.Count);
                    } while (!result.EndOfMessage);

                    if (result.MessageType == WebSocketMessageType.Text)
                        ProcessTextMessage(Encoding.UTF8.GetString(message.ToArray()));
                }
            }
        }

        private void ProcessTextMessage(string json)
        {
            PoseFrame frame;
            try
            {
                frame = JsonUtility.FromJson<PoseFrame>(json);
            }
            catch (Exception)
            {
                return; // malformed/partial frame; drop it and keep the connection alive
            }
            if (frame?.joints == null || frame.joints.Length != JointCount * 4) return;

            lock (sync)
            {
                latestJoints = frame.joints;
            }
        }

        private void SetError(string error)
        {
            lock (sync) lastError = error;
        }

        // Wire format: {"frame": int, "layout": str, "joints": [96 floats]}
        // ("joints" is a FLAT array -- JsonUtility cannot deserialize a nested
        // array-of-arrays). Keep this in sync with
        // mobileposer/realtime/stream_out.py::PoseBroadcaster.broadcast --
        // update both sides together if the payload shape ever changes.
        [Serializable]
        private class PoseFrame
        {
            public int frame;
            public string layout;
            public float[] joints;
        }
    }
}
