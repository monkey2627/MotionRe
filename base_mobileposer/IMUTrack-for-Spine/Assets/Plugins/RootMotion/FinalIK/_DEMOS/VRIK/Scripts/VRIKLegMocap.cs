using UnityEngine;
using RootMotion.FinalIK;

namespace RootMotion.Demos
{
    /// <summary>
    /// Makes VRIK work with thigh and calf targets at a cost of reduced foot accuracy.
    /// </summary>
    public class VRIKLegMocap : MonoBehaviour
    {

        public VRIK ik;

        [Range(0f, 1f)] public float thighWeight = 1f;
        [Range(0f, 1f)] public float calfWeight = 1f;

        public Transform leftThighTarget;
        public Transform leftCalfTarget;
        public Transform rightThighTarget;
        public Transform rightCalfTarget;

        void Start()
        {
            if (ik != null) ik.solver.OnPostUpdate += AfterVRIK;
        }

        void AfterVRIK()
        {
            if (ik == null) return;

            UpdateLeg(ik.references.leftThigh, ik.references.leftCalf, ik.references.leftFoot, leftThighTarget, leftCalfTarget);
            UpdateLeg(ik.references.rightThigh, ik.references.rightCalf, ik.references.rightFoot, rightThighTarget, rightCalfTarget);
        }

        private void UpdateLeg(Transform thigh, Transform calf, Transform foot, Transform thighTarget, Transform calfTarget)
        {
            if (thighTarget != null && thigh != null && calf != null) RotateTowards(thigh, calf.position - thigh.position, thighTarget.position - thigh.position, thighWeight);
            if (calfTarget != null && calf != null && foot != null) RotateTowards(calf, foot.position - calf.position, calfTarget.position - calf.position, calfWeight);
        }

        private static void RotateTowards(Transform bone, Vector3 currentDirection, Vector3 targetDirection, float weight)
        {
            if (weight <= 0f) return;
            if (currentDirection == Vector3.zero) return;
            if (targetDirection == Vector3.zero) return;

            Quaternion rotation = Quaternion.FromToRotation(currentDirection, targetDirection);
            if (weight < 1f) rotation = Quaternion.Slerp(Quaternion.identity, rotation, weight);

            bone.rotation = rotation * bone.rotation;
        }

        void OnDestroy()
        {
            if (ik != null) ik.solver.OnPostUpdate -= AfterVRIK;
        }
    }
}
