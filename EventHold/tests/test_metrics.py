import unittest
import numpy as np
from eventhold.holds import mine
from eventhold.metrics import retention


class MetricTests(unittest.TestCase):
    def test_stably_wrong_is_not_success(self):
        r=retention(np.full(600,45.),np.ones(600,bool),60)
        self.assertEqual(r['retention_delta_deg'],0.)
        self.assertEqual(r['mean_error_deg'],45.)
        self.assertTrue(r['failure_all'])
        self.assertIsNone(r['retention_failure_given_correct_entry'])

    def test_correct_then_forgetting(self):
        r=retention(np.r_[np.full(300,5.),np.full(300,25.)],np.ones(600,bool),60)
        self.assertTrue(r['retention_failure_given_correct_entry'])
        self.assertEqual(r['retention_delta_deg'],20.)

    def test_absent_reference_not_zero_error(self):
        r=retention(np.zeros(600),np.zeros(600,bool),60)
        self.assertEqual(r['status'],'no_valid_reference')
        self.assertNotIn('mean_error_deg',r)

    def test_candidates_not_semantic_truth(self):
        r=mine(np.zeros((601,72)),np.ones((601,5),bool))
        self.assertEqual(len(r),2)
        self.assertTrue(all(x['pose_semantics']=='unknown' for x in r))
        self.assertTrue(all(x['annotation_status']=='candidate_not_reviewed' for x in r))


if __name__=='__main__':unittest.main()
