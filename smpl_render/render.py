"""Render Android SMPL axis-angle outputs as a three-view mesh video."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/home/duanyuhan/dyh/motion/MotionRe/base_mobileposer')
sys.path.insert(0, str(ROOT))
from mobileposer.articulate.model import ParametricModel
from mobileposer.verify_android_output import axis_angle_matrix

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'smpl.jsonl.json'
rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
torch.set_num_threads(2)
model = ParametricModel(str(ROOT / 'mobileposer/smpl/basicmodel_m.pkl'), use_pose_blendshape=True)
rotations = torch.tensor(np.stack([axis_angle_matrix(r['global_orient'] + r['body_pose']) for r in rows]), dtype=torch.float32)
shape = torch.tensor([r['betas'] for r in rows], dtype=torch.float32)
translation = torch.tensor([r['transl'] for r in rows], dtype=torch.float32)
with torch.no_grad():
    _, joints, vertices = model.forward_kinematics(rotations, shape, translation, calc_mesh=True)
vertices = vertices.numpy()
faces = np.asarray(model.face, dtype=np.int32)
assert np.isfinite(vertices).all()
# One fixed display offset for the entire clip, never per-frame foot locking.
floor_offset = -float(vertices[:, :, 1].min())
vertices[:, :, 1] += floor_offset

W, H, SS = 1440, 800, 2
views = [('FRONT', 0.0), ('SIDE', np.pi / 2), ('THREE-QUARTER', np.pi / 6)]
height = float(vertices[:, :, 1].max())
horizontal = max(float(np.abs(vertices[:, :, 0] * np.cos(a) - vertices[:, :, 2] * np.sin(a)).max()) for _, a in views)
scale = min(580 / height, 200 / max(horizontal, 0.1))
light = np.array([-0.35, 0.65, 0.68]); light /= np.linalg.norm(light)
video = OUT / 'smpl_three_views.mp4'
cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
previews = []

def text(img, label, xy, size=0.65, color=(185, 197, 210)):
    cv2.putText(img, label, tuple(int(v * SS) for v in xy), cv2.FONT_HERSHEY_SIMPLEX, size * SS, color, SS, cv2.LINE_AA)

try:
    for i, verts in enumerate(vertices):
        canvas = np.full((H * SS, W * SS, 3), (29, 24, 20), dtype=np.uint8)
        text(canvas, 'SMPL | Android output', (32, 40), 0.85, (245, 239, 230))
        text(canvas, f'wrists_thighs_waist   |   frame {i:03d}/179   |   playback {i/30:.2f}s   |   pose {rows[i]["pose_time_seconds"]:.2f}s', (32, 72), 0.52)
        for k, (label, angle) in enumerate(views):
            ca, sa = np.cos(angle), np.sin(angle)
            basis = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
            v = verts @ basis.T
            center = 240 + 480 * k
            text(canvas, label, (center - 90, 115), 0.58)
            for offset in np.arange(-1.0, 1.01, 0.25):
                px = int((center + offset * scale) * SS)
                cv2.line(canvas, (px, 138 * SS), (px, 715 * SS), (43, 36, 30), 1, cv2.LINE_AA)
            for y in np.arange(0, height + 0.05, 0.25):
                py = int((715 - y * scale) * SS)
                cv2.line(canvas, ((k * 480 + 20) * SS, py), ((k * 480 + 460) * SS, py), (43, 36, 30), 1, cv2.LINE_AA)
            cv2.line(canvas, ((k * 480 + 20) * SS, 715 * SS), ((k * 480 + 460) * SS, 715 * SS), (105, 88, 65), SS, cv2.LINE_AA)
            triangles = v[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            shade = 0.38 + 0.62 * np.maximum(normals @ light, 0)
            colors = np.clip(shade[:, None] * np.array([220, 175, 95]), 0, 255).astype(np.uint8)
            pixels = np.stack([center + v[:, 0] * scale, 715 - v[:, 1] * scale], axis=1)
            pixels = np.rint(pixels * SS).astype(np.int32)[faces]
            for f in np.argsort(triangles[:, :, 2].mean(axis=1)):
                cv2.fillConvexPoly(canvas, pixels[f], tuple(int(c) for c in colors[f]), lineType=cv2.LINE_AA)
        text(canvas, '30 FPS  |  bundled synthetic IMU  |  zero translation / mean body shape', (32, 764), 0.56)
        frame = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
        encoder.stdin.write(frame.tobytes())
        if i == 90:
            cv2.imwrite(str(OUT / 'preview.png'), frame)
        if i in [0, 45, 90, 135, 179]:
            previews.append(frame[:, 960:1440])
        if i % 30 == 0:
            print(f'Rendered {i + 1}/{len(rows)}', flush=True)
finally:
    encoder.stdin.close()
    code = encoder.wait()
if code:
    raise RuntimeError(f'ffmpeg failed: {code}')
cv2.imwrite(str(OUT / 'contact_sheet.jpg'), np.concatenate(previews, axis=1))
(OUT / 'render_info.json').write_text(json.dumps({'source': str(SOURCE), 'frames': len(rows), 'fps': 30, 'duration_seconds': len(rows)/30, 'resolution': [W, H], 'views': [x[0] for x in views], 'model': 'basicmodel_m.pkl', 'pose_blendshapes': True, 'display_floor_offset_m': floor_offset, 'pose_source': 'global_orient + body_pose (axis-angle)', 'translation': 'original exported values; fixed display offset only'}, indent=2) + '\n')
print(video, flush=True)
