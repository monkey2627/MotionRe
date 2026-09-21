import unittest,json
from pathlib import Path
class SyntheticSplitTests(unittest.TestCase):
 def test_split_groups_do_not_cross(self):
  d=json.loads(Path('reports/g0/five_imu_synthetic_split/split.json').read_text());seen={}
  for row in d['rows']:
   old=seen.setdefault(row['subject_key'],row['split']);self.assertEqual(old,row['split'])
  self.assertTrue(d['no_test_claim']);self.assertGreater(sum(x['split']=='holdout_exploration' for x in d['rows']),0)
if __name__=='__main__':unittest.main()
