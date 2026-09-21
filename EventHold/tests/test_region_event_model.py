import unittest,torch
from eventhold.region_event_model import RegionEventWrite

class RegionEventModelTests(unittest.TestCase):
 def test_shapes_gates_and_root_copy(self):
  torch.manual_seed(1);m=RegionEventWrite(96);x=torch.randn(2,17,66);p,s,gl,gu=m(x,return_gates=True)
  self.assertEqual(tuple(p.shape),(2,17,24,3,3));self.assertEqual(tuple(s.shape),(2,96));self.assertEqual(tuple(gl.shape),(2,17,32));self.assertTrue(torch.all((gl>=0)&(gl<=1)));self.assertTrue(torch.all((gu>=0)&(gu<=1)));torch.testing.assert_close(p[:,:,0],x[:,:,:9].view(2,17,3,3))
 def test_future_perturbation_and_chunk_state(self):
  torch.manual_seed(2);m=RegionEventWrite(96).eval();x=torch.randn(1,31,66);a,_=m(x);y=x.clone();y[:,20:]+=10;b,_=m(y);torch.testing.assert_close(a[:,:20],b[:,:20]);s=None;parts=[]
  # explicitly use ordinary sequential chunks to verify state persistence
  s=None;parts=[]
  for i in range(0,31,7):q,s=m(x[:,i:i+7],s);parts.append(q)
  torch.testing.assert_close(a,torch.cat(parts,1),atol=1e-6,rtol=1e-6)
 if __name__=='__main__':unittest.main()
