import unittest
import numpy as np
import torch
from eventhold.adapters import eventhold_features
from eventhold.baseline import CausalBaseline,geodesic
from test_adapters import fixture


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3);torch.set_num_threads(2)
        self.model=CausalBaseline(hidden=16).eval()
        self.x=torch.tensor(eventhold_features(fixture(t=31)),dtype=torch.float32)[None]

    def test_chunked_equals_continuous(self):
        with torch.no_grad():
            full,_=self.model(self.x)
            a,h=self.model(self.x[:,:12]);b,_=self.model(self.x[:,12:],h)
        torch.testing.assert_close(full,torch.cat([a,b],dim=1),atol=1e-6,rtol=1e-5)

    def test_future_invariance(self):
        changed=self.x.clone();changed[:,15:]=100
        with torch.no_grad():a,_=self.model(self.x);b,_=self.model(changed)
        torch.testing.assert_close(a[:,:15],b[:,:15],atol=0,rtol=0)

    def test_reset_isolation(self):
        with torch.no_grad():
            a,_=self.model(self.x);self.model(self.x*2);b,_=self.model(self.x)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_valid_rotations_and_measured_root(self):
        p,_=self.model(self.x)
        torch.testing.assert_close(p[...,0,:,:],self.x[...,:9].reshape(1,31,3,3))
        torch.testing.assert_close(torch.linalg.det(p),torch.ones(p.shape[:-2]),atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.isfinite(geodesic(p,p)).all())

    def test_single_optimizer_step_finite(self):
        self.model.train();opt=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        p,_=self.model(self.x);target=torch.eye(3).expand_as(p)
        loss=geodesic(p[:,:,1:],target[:,:,1:]).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in self.model.parameters() if x.grad is not None))
        opt.step()


if __name__=='__main__':unittest.main()
