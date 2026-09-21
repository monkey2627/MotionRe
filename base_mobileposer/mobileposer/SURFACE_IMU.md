# SMPL surface IMU experiments

The six original layouts share 11 fixed surface attachments. `wrists` means the
SlimeVR **forearm** tracker, not the hand tracker. The configuration is frozen in
`configs/no_head_surface_imu.json`, including exact triangles, barycentric weights,
bone rotations, mounting rotations and a SHA256 identifying the SMPL pickle.

View the positions:

```bash
python -m mobileposer.surface_imu
```

This produces `results/surface_imu/placement_preview.png` (front, left side,
back). Back-view hollow markers are see-through annotations, not posterior
attachments. SMPL is used because that is the actual training model; its vertex
and face IDs must not be reused for SMPL-X.

The default locations follow the supplied diagram: lateral forearm 75% elbow to
wrist; lateral upper arm 82% shoulder to elbow; anterior thigh 86% hip to knee;
lateral shank 88% knee to ankle; dorsal foot; anterior waist at spine1 height.
These are explicit anatomical approximations, not measurements inferred from
pixels. Foot trackers are unassigned in the screenshot; feet are synthetic
candidate locations. Left and right refer to anatomy.

## Synthesis and compatibility

Surface position is the weighted sum of three skinned vertices. Mesh computation
is chunked; acceleration is calculated after joining the point trajectory so
chunk boundaries do not introduce artifacts. Existing 30 FPS finite differences,
smoothing, shape coefficients and disabled pose blendshapes are retained.
Signals are world-frame linear acceleration (m/s², no gravity) and bone-calibrated
rotation matrices. Mounting rotation is represented explicitly and removed by
calibration. This does not model strap slip or soft-tissue dynamics.

Files retain `acc [T,6,3]` and `ori [T,6,3,3]`; only the first five slots feed the
60-dimensional network input. The sixth slot remains the pelvis joint signal.
`--imu-mode joint` remains the default for existing callers; `surface` chooses
new default paths. Metadata mismatches fail rather than reuse an incompatible
cache. Sample-limited datasets are marked and must not be used as full caches.

## Full experiment

```bash
python -m mobileposer.run_no_head_5imu_experiments \
  --imu-mode surface --eval-dataset dip --gpus 1 2 3 --no-video
```

Default outputs:

- `data/no_head_5imu_surface_processed/<layout>`: AMASS training files.
- `data/no_head_5imu_surface_dip_test/<layout>/dip_test.pt`: DIP s09/s10.
- `data/no_head_5imu_surface_dip_test/<layout>/train/dip_train.pt`: DIP s01–s08,
  generated separately, never consumed by the test loader or AMASS training.
- `checkpoints/no_head_5imu_surface/<layout>`: new model weights.
- `results/no_head_5imu_surface`: inference and evaluation outputs.

DIP lacks global translation in this generator and uses the existing all-ones
shape coefficients. DIP results are synthetic evaluations, not real sensor
validation. Compare old and new models on the *same* surface test files:

```bash
python -m mobileposer.evaluate_no_head_5imu \
  --checkpoint-root checkpoints/no_head_5imu \
  --data-root data/no_head_5imu_surface_dip_test \
  --output-root results/no_head_5imu_surface/baseline_on_surface \
  --device cuda:0 --no-video
```

## Checks

```bash
python -m unittest discover -s tests -p test_surface_imu.py
python -m mobileposer.no_head_layouts --imu-mode surface \
  --layout wrists_shanks_waist --source dip-test --max-sequences 1 \
  --output-root data/surface_imu_smoke
```

For a genuine tiny training check use the generated file directly with the
training entrypoint and set `MOBILEPOSER_PROCESSED_DATASETS` to its layout folder,
`MOBILEPOSER_TRAIN_COMBOS=all_5imu`, and an isolated checkpoint folder. Pass
`--fast-dev-run` to `mobileposer.train`: it uses batch size one and zero workers
only for this diagnostic mode. It does not save weights; do not request combine
or inference for this mode. The full experiment runner rejects limited-data
caches on purpose.
