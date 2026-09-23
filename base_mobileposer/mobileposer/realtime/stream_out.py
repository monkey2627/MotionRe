"""Local WebSocket broadcaster: streams per-frame SMPL joint quaternions to
Unity, replacing SlimeVR's VMC output as the pose source for the avatar.

Plain JSON over WebSocket -- not SolarXR's flatbuffers protocol. This is a
new, project-local channel between our own two endpoints (this bridge and
the Unity receiver script), so there is no reason to take on flatbuffers
complexity here; SlimeVR-Server's own VMC/OSC output is untouched and can
keep running in parallel as a comparison reference (plan M6).
"""
from __future__ import annotations

import asyncio
import json
from typing import Set

import numpy as np
import websockets

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 21200  # arbitrary, distinct from SlimeVR-Server's 21110


class PoseBroadcaster:
    """Holds the set of connected Unity clients and fans out each frame."""

    def __init__(self) -> None:
        self._clients: Set = set()

    async def _handler(self, websocket) -> None:
        self._clients.add(websocket)
        try:
            async for _ in websocket:  # send-only channel; ignore any inbound data
                pass
        finally:
            self._clients.discard(websocket)

    async def serve(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        return await websockets.serve(self._handler, host, port)

    async def broadcast(self, frame_index: int, layout_name: str, joints_xyzw: np.ndarray) -> None:
        """joints_xyzw: (24, 4) array of (x, y, z, w) local-rotation quaternions,
        already converted to Unity's left-handed convention (see
        rotmat.batch_smpl_matrix_to_unity_quat_xyzw), in mobileposer's fixed
        24-joint SMPL order (0=pelvis, 1/2=hip, ..., see mobileposer/fit_ours_smpl.py's
        joint-index comment for the full list).

        "joints" is sent as a flat 96-float array (24 * 4), not a nested
        array-of-arrays: Unity's JsonUtility (used on the receiving end, see
        Assets/slimeVR/Scripts/MobilePoserPoseSource.cs) cannot deserialize
        nested JSON arrays, only flat arrays of a primitive type.
        """
        if not self._clients:
            return
        payload = json.dumps({
            "frame": frame_index,
            "layout": layout_name,
            "joints": joints_xyzw.reshape(-1).tolist(),
        })
        stale = []
        for client in self._clients:
            try:
                await client.send(payload)
            except websockets.ConnectionClosed:
                stale.append(client)
        for client in stale:
            self._clients.discard(client)
