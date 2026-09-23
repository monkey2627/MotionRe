"""M1 diagnostic: connect to a running SlimeVR-Server and print live tracker data.

Run this while SlimeVR-Server is up and trackers are bound, to sanity-check
that the SolarXR feed is reachable and that BodyPart/rotation/acceleration
look right before wiring up the actual inference pipeline.

Usage (from repo root, base_mobileposer/)::

    "E:/SoftWare/Conda/python.exe" -m mobileposer.realtime.probe_solarxr
    "E:/SoftWare/Conda/python.exe" -m mobileposer.realtime.probe_solarxr --hz 5
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from mobileposer.realtime.solarxr_client import stream_trackers, TrackerSample


def _format_sample(s: TrackerSample) -> str:
    qx, qy, qz, qw = s.quat_xyzw
    ax, ay, az = s.accel_xyz
    status = "OK" if s.online else f"status={s.status}"
    return (
        f"  {s.tracker_key:>6}  {s.body_part_name:<18} {status:<10} "
        f"quat=({qx:+.3f},{qy:+.3f},{qz:+.3f},{qw:+.3f})  "
        f"accel=({ax:+.2f},{ay:+.2f},{az:+.2f}) m/s^2"
    )


async def _main(print_hz: float) -> None:
    latest: dict[str, TrackerSample] = {}
    last_print = 0.0
    interval = 1.0 / print_hz if print_hz > 0 else 0.0

    print(f"Connecting to SlimeVR-Server SolarXR feed...", file=sys.stderr)
    async for update in stream_trackers():
        latest.update(update)
        now = time.monotonic()
        if now - last_print < interval:
            continue
        last_print = now
        print(f"--- {len(latest)} tracker(s) known ---")
        for key in sorted(latest):
            print(_format_sample(latest[key]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hz", type=float, default=2.0, help="console print rate (data itself streams as fast as the server sends it)")
    args = parser.parse_args()
    try:
        asyncio.run(_main(args.hz))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
