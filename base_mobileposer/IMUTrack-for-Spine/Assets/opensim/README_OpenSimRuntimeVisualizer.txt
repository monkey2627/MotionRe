OpenSim Unity runtime visualizer

Scene
Assets/opensim/opensimskeleton.unity

Main script
Assets/opensim/Scripts/OpenSimFullBodyRuntimeVisualizer.cs

Data loaded by default
Assets/opensim/Data/MaleFullBodyModel_v2.0_OS4_BU.osim
Assets/opensim/Data/FemaleFullBodyModel_v2.0_OS4_BU.osim
Assets/opensim/Data/T1_to_L5_synthetic_spine_motion_heatmap_clean.mot
Assets/opensim/Data/Geometry/*.vtp

How it works
1. At Play, the script parses each .osim BodySet and JointSet.
2. It loads each body's attached_geometry/Mesh .vtp file and parents the mesh
   under the corresponding OpenSim body transform.
3. It reads the .mot file and applies the current frame's coordinates to the
   parsed joint transform axes.
4. For every frame, it computes each T1-L5 joint's local rotation deviation
   from the first frame using Quaternion.Angle(reference, current).
5. The spine heatmap colors are applied directly to the corresponding vertebra
   mesh materials (thoracic2 through sacrum). No precomputed color table is used.

Default color mapping
The script uses frame-relative contrast by default:
relative_segment_value * sqrt(frame_max_rotation / maxRotationForFullHeat)

This makes the color difference between spine levels clearer while keeping
very early low-motion frames from saturating to red.

Controls
Space: play/pause
R: restart playback
C: toggle frame-relative contrast

Inspector knobs
maxRotationForFullHeat controls the temporal gate/saturation point.
useFrameRelativeContrast can be disabled to use absolute angle coloring.
loadOpenSimMeshes should stay enabled if you want OpenSim-like bone geometry.
showDebugJointSkeleton can be enabled to display the old line/sphere debug view.
showSpineHeatOverlay can be enabled to add colored cylinders on top of the mesh.
