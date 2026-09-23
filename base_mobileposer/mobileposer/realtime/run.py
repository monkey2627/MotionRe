"""End-to-end realtime bridge: SlimeVR-Server (SolarXR) -> mobileposer
inference -> local WebSocket broadcast for the Unity receiver script.

Does not modify SlimeVR-Server -- it only subscribes to its existing SolarXR
data feed (see solarxr_client.py) as an ordinary client. SlimeVR-Server's own
VMC/OSC output keeps running unmodified in parallel (useful as a comparison
reference, plan M6).

Orientation needs no calibration (rotation_reference_adjusted is already
usable directly, see calibration.py::bone_orientation). Acceleration DOES
need a short startup calibration window with rotational motion -- this is not
avoidable, see calibration.py's module docstring for why (SlimeVR-Server
composes the raw<->reference-adjusted relationship from four private,
unexposed quaternions; only the delta-trajectory trick used here can recover
the one recoverable component). Once fitted, the ongoing per-frame cost is a
single fixed matrix multiply -- streaming is real-time after that.

Usage (from repo root, base_mobileposer/), with SlimeVR-Server running,
trackers bound, and the matching no_head_5imu_surface checkpoint synced
locally to checkpoints/no_head_5imu_surface/<layout>/1/base_model.pth::

    python -m mobileposer.realtime.run

NOTE: this is the M1-M4 version per the implementation plan. It resolves the
layout once at startup and does not yet hot-swap if trackers connect/disconnect
mid-session (plan M5) -- restart the script if your tracker set changes.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from mobileposer.realtime.calibration import UnobservableHeadingError, calibrate_heading
from mobileposer.realtime.infer import DEFAULT_CHECKPOINT_ROOT, InferenceSession, load_model
from mobileposer.realtime.layout import LayoutMismatchError, ResolvedLayout, detect_layout
from mobileposer.realtime.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from mobileposer.realtime.solarxr_client import TrackerSample, stream_trackers
from mobileposer.realtime.stream_out import PoseBroadcaster


async def _wait_for_layout(tracker_stream, latest: Dict[str, TrackerSample]) -> ResolvedLayout:
    print("Waiting for a recognized tracker layout (bind trackers in SlimeVR-Server)...", file=sys.stderr)
    last_error = None
    async for update in tracker_stream:
        latest.update(update)
        try:
            resolved = detect_layout(latest)
        except LayoutMismatchError as exc:
            if str(exc) != last_error:
                print(f"  ... {exc}", file=sys.stderr)
                last_error = str(exc)
            continue
        print(f"Resolved layout: {resolved.name} (labels: {resolved.labels})", file=sys.stderr)
        return resolved
    raise RuntimeError("SolarXR stream ended before a layout was resolved")


async def _collect_heading_window(
    tracker_stream, latest: Dict[str, TrackerSample], resolved: ResolvedLayout, frames: int
):
    """Collect `frames` samples per label while the user moves the trackers
    around (any rotation works -- twisting the torso, swinging limbs, etc.).
    A perfectly still hold will not work: see calibration.py."""
    raw: Dict[str, List] = {label: [] for label in resolved.labels}
    adjusted: Dict[str, List] = {label: [] for label in resolved.labels}
    print(
        f"Calibrating: rotate/move the trackers for the next {frames} frames "
        f"(twist your torso, swing your arms/legs -- holding still will NOT work)...",
        file=sys.stderr,
    )
    async for update in tracker_stream:
        latest.update(update)
        for label in resolved.labels:
            key = resolved.label_to_tracker_key[label]
            sample = latest.get(key)
            if sample is not None and sample.online:
                raw[label].append(sample.raw_quat_xyzw)
                adjusted[label].append(sample.quat_xyzw)
        if all(len(v) >= frames for v in raw.values()):
            break
    raw_arrays = {label: np.asarray(values[:frames], dtype=np.float32) for label, values in raw.items()}
    adjusted_arrays = {label: np.asarray(values[:frames], dtype=np.float32) for label, values in adjusted.items()}
    return raw_arrays, adjusted_arrays


async def main_async(args: argparse.Namespace) -> None:
    broadcaster = PoseBroadcaster()
    await broadcaster.serve(args.host, args.port)
    print(f"Broadcasting SMPL pose on ws://{args.host}:{args.port}", file=sys.stderr)

    latest: Dict[str, TrackerSample] = {}
    tracker_stream = stream_trackers()

    resolved = await _wait_for_layout(tracker_stream, latest)

    while True:
        raw_arrays, adjusted_arrays = await _collect_heading_window(
            tracker_stream, latest, resolved, args.calibration_frames
        )
        try:
            heading = calibrate_heading(resolved.labels, raw_arrays, adjusted_arrays)
            break
        except UnobservableHeadingError as exc:
            print(f"  Calibration failed, retrying: {exc}", file=sys.stderr)
    for label, diag in heading.diagnostics.items():
        print(f"  {label}: yaw={diag['yaw_deg']:+.1f}deg residual_p95={diag['delta_residual_p95_deg']:.2f}deg",
              file=sys.stderr)

    model = load_model(resolved.name, args.checkpoint_root)
    session = InferenceSession(layout=resolved, model=model, heading=heading)
    print("Calibrated. Streaming live pose...", file=sys.stderr)

    frame_index = 0
    async for update in tracker_stream:
        latest.update(update)
        try:
            quats = np.stack(
                [latest[resolved.label_to_tracker_key[label]].quat_xyzw for label in resolved.labels]
            )
            accel = np.stack(
                [latest[resolved.label_to_tracker_key[label]].accel_xyz for label in resolved.labels]
            )
        except KeyError:
            continue  # a required tracker dropped out; wait for it to come back

        output = session.step(quats, accel)
        quats_out = batch_smpl_matrix_to_unity_quat_xyzw(output.detach().numpy())
        await broadcaster.broadcast(frame_index, resolved.name, quats_out)
        frame_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="host to broadcast the SMPL pose stream on")
    parser.add_argument("--port", type=int, default=21200, help="port to broadcast the SMPL pose stream on")
    parser.add_argument("--calibration-frames", type=int, default=150,
                         help="frames to collect (while moving/rotating) for the startup heading fit, ~5s at 30Hz")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT,
                         help="root directory containing <layout>/1/base_model.pth checkpoints")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
