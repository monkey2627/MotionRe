import tempfile
import unittest
from pathlib import Path
import torch
from mobileposer.articulate import math
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.surface_imu import load_attachments, surface_kinematics, attachment_points
from mobileposer.no_head_layouts import _syn_acc, synthesis_metadata, validate_cached_dataset

class SurfaceIMUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model=ParametricModel(paths.smpl_file)
        cls.config=load_attachments()
        cls.labels=list(cls.config['attachments'])

    def test_static_and_calibration(self):
        pose=torch.eye(3).repeat(24,24,1,1)
        r,j,p=surface_kinematics(self.model,pose,torch.zeros(10),torch.zeros(24,3),self.labels,self.config,7)
        self.assertLess(float(_syn_acc(p).abs().max()),1e-4)
        torch.testing.assert_close(r,torch.eye(3).expand_as(r),atol=1e-6,rtol=1e-6)
        self.assertEqual(p.shape,(24,12,3))
        self.assertGreater(float((p[:,:-1]-j[:,[self.config['attachments'][x]['joint'] for x in self.labels]]).norm(dim=-1).min()),.02)

    def test_rotating_point_chunk_equivalence(self):
        aa=torch.zeros(35,24,3); aa[:,0,1]=torch.arange(35)*.025
        pose=math.axis_angle_to_rotation_matrix(aa).view(35,24,3,3)
        args=(self.model,pose,torch.zeros(10),torch.zeros(35,3),self.labels,self.config)
        one=surface_kinematics(*args,chunk_size=35)
        chunks=surface_kinematics(*args,chunk_size=8)
        for a,b in zip(one,chunks): torch.testing.assert_close(a,b)
        self.assertGreater(float(_syn_acc(one[2])[:,0].norm(dim=-1).max()),.1)
        torch.testing.assert_close(_syn_acc(one[2])[:,-1],torch.zeros(35,3),atol=1e-4,rtol=0)
        r,j,v=self.model.forward_kinematics(pose,torch.zeros(10),torch.zeros(35,3),calc_mesh=True)
        torch.testing.assert_close(one[2][:,:-1],attachment_points(v,self.config,self.labels))
        torch.testing.assert_close(one[0][:,:-1],r[:,[self.config['attachments'][x]['joint'] for x in self.labels]],atol=1e-6,rtol=1e-6)

    def test_cache_rejects_joint_surface_mix(self):
        from mobileposer.surface_imu import DEFAULT_CONFIG
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.pt'; torch.save({'acc':[]},p)
            meta=synthesis_metadata('surface',DEFAULT_CONFIG)
            with self.assertRaises(ValueError): validate_cached_dataset(p,meta)
            torch.save({'synthesis':meta},p)
            self.assertTrue(validate_cached_dataset(p,meta))

if __name__=='__main__': unittest.main()
