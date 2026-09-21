import unittest
from pathlib import Path
import numpy as np


class SyntheticContractTests(unittest.TestCase):
    def test_pilot_has_five_named_channels_and_boundary_mask(self):
        root=Path('reports/g0/five_imu_synthetic_pilot')
        manifest=(root/'manifest.json')
        self.assertTrue(manifest.exists())
        import json
        meta=json.loads(manifest.read_text())
        self.assertEqual(meta['files'],8);self.assertFalse(meta['head_input'])
        for row in meta['manifest']:
            d=np.load(root/row['id'])
            self.assertEqual(d['orientation'].shape[1:],(5,3,3))
            self.assertEqual(d['acceleration'].shape[1:],(5,3))
            self.assertEqual(d['target'].shape[1:],(24,3,3))
            self.assertEqual(int((~d['valid']).sum()),2)
            r=d['orientation'][d['valid']]
            np.testing.assert_allclose(np.linalg.det(r),1,atol=2e-4)
            np.testing.assert_allclose(np.swapaxes(r,-1,-2)@r,np.broadcast_to(np.eye(3),r.shape),atol=2e-4)
            self.assertTrue(np.isfinite(d['acceleration']).all())


if __name__=='__main__':unittest.main()
