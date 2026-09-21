import unittest, json
from pathlib import Path

class SmokeSyntheticTests(unittest.TestCase):
    def test_smoke_report_is_complete(self):
        p=Path('reports/g0/five_imu_synthetic_smoke/summary.json')
        self.assertTrue(p.exists())
        d=json.loads(p.read_text());self.assertEqual(d['files'],8);self.assertFalse(d['performance_claim'])
        for row in d['rows']:
            self.assertEqual(row['feature_dim'],66);self.assertEqual(row['valid_frames'],row['frames']-2)
            self.assertLessEqual(row['chunk_parity_max_abs'],2e-5);self.assertLessEqual(row['root_copy_max_abs'],1e-6)

if __name__=='__main__':unittest.main()
