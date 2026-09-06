import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from mobileposer.helpers import *
from mobileposer.config import *
from mobileposer.constants import NUM_VERTICES
import mobileposer.articulate as art

PRED_COLOR = [0.78, 0.49, 0.49]  # reddish
GT_COLOR   = [0.49, 0.65, 0.78]  # blueish


class SMPLViewer:
    def __init__(self, fps: int=25):
        self.fps = fps
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=model_config.device)

    def _render_subject(self, verts, faces, color):
        """Render a single mesh sequence; return list of uint8 RGB frames [H,W,3]."""
        import open3d as o3d

        n_frames = verts.shape[0]
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=540, height=720)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts[0])
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        vis.add_geometry(mesh)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.15, 0.15, 0.15])
        opt.light_on = True

        frames = []
        for i in range(n_frames):
            mesh.vertices = o3d.utility.Vector3dVector(verts[i])
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frames.append((frame * 255).astype(np.uint8))

        vis.destroy_window()
        return frames

    def _add_label(self, frame, label):
        """Overlay a text label on the top-left of a frame."""
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
        except Exception:
            font = ImageFont.load_default()
        draw.rectangle([8, 8, 8 + len(label) * 17, 44], fill=(30, 30, 30))
        draw.text((12, 12), label, fill=(255, 255, 255), font=font)
        return np.array(img)

    def _save_video(self, frames, output_path):
        import imageio
        try:
            imageio.mimwrite(output_path, frames, fps=self.fps, macro_block_size=1)
            print(f'Saved: {output_path}')
        except Exception as e:
            gif_path = output_path.rsplit('.', 1)[0] + '.gif'
            imageio.mimwrite(gif_path, frames[::2], fps=max(self.fps // 2, 1))
            print(f'MP4 failed ({e}), saved GIF: {gif_path}')

    def _get_verts(self, pose, tran):
        tran = tran - tran[:1]
        _, _, v = self.bodymodel.forward_kinematics(pose, tran=tran, calc_mesh=True)
        return v.cpu().numpy()

    def view(self, pose_p, tran_p, pose_t, tran_t, with_tran: bool=False):
        if not with_tran:
            tran_p = torch.zeros(pose_p.shape[0], 3, device=pose_p.device)
            tran_t = torch.zeros(pose_t.shape[0], 3, device=pose_t.device)

        pose_p = pose_p.view(-1, 24, 3, 3)
        tran_p = tran_p.view(-1, 3)
        pose_t = pose_t.view(-1, 24, 3, 3)
        tran_t = tran_t.view(-1, 3)

        faces = self.bodymodel.face

        print('Rendering prediction...')
        verts_p  = self._get_verts(pose_p, tran_p)
        frames_p = self._render_subject(verts_p, faces, PRED_COLOR)

        print('Rendering ground truth...')
        verts_t  = self._get_verts(pose_t, tran_t)
        frames_t = self._render_subject(verts_t, faces, GT_COLOR)

        # stitch side by side with labels
        n = min(len(frames_p), len(frames_t))
        combined = []
        for i in tqdm(range(n), desc='Compositing'):
            left  = self._add_label(frames_p[i], 'Prediction')
            right = self._add_label(frames_t[i], 'Ground Truth')
            combined.append(np.concatenate([left, right], axis=1))

        self._save_video(combined, 'output.mp4')
