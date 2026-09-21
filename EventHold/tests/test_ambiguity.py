import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from eventhold.diagnose_ambiguity import eligible_frames, eligible_sampled_frames, pair_distances, select_matches


class AmbiguityTests(unittest.TestCase):
    def test_sparse_history_never_uses_missing_sampled_values(self):
        valid=np.ones((200,5),bool);valid[42,3]=False
        np.testing.assert_array_equal(eligible_sampled_frames(valid,[0,30,60,120]),[120,150,180])
        valid[90,2]=False
        np.testing.assert_array_equal(eligible_sampled_frames(valid,[0,30,60,120]),[180])

    def test_missing_interior_excludes_history_and_future_does_not(self):
        valid=np.ones((200,5),bool)
        np.testing.assert_array_equal(eligible_frames(valid),[120,150,180])
        valid[42,3]=False
        np.testing.assert_array_equal(eligible_frames(valid),[180])
        valid[195]=False
        np.testing.assert_array_equal(eligible_frames(valid),[180])

    def test_physical_distance_units(self):
        r=Rotation.from_euler('z',[0,90],degrees=True).as_matrix()[:,None]
        a=np.array([[[0.,0,0]],[[0,3,4]]])
        angle,acc=pair_distances(r,a)
        self.assertAlmostEqual(angle[0,1,0],90)
        self.assertAlmostEqual(acc[0,1,0],5)
        np.testing.assert_allclose(angle,angle.swapaxes(0,1))

    def test_history_changes_choice_with_identical_current_pool(self):
        current=np.array([[0,1,2],[1,0,3],[2,3,0]],dtype=float)[...,None]
        past=np.array([[0,9,.2],[9,0,8],[.2,8,0]],dtype=float)[...,None]
        distances=[(current,np.zeros_like(current))]+[(past,np.zeros_like(past))]*3
        setting={'max_sensor_rotation_deg':10,'max_sensor_acc_difference_m_s2':1}
        result=select_matches(np.array([0,300,600]),distances,setting,300,5,np.arange(3))
        query={r['policy']:r for r in result if r['query_index']==0}
        self.assertEqual(query['current']['neighbor_index'],1)
        self.assertEqual(query['past_2s']['neighbor_index'],2)
        self.assertEqual(query['current']['pool_size'],query['past_2s']['pool_size'])
        self.assertFalse(any(r['query_index']==r['neighbor_index'] for r in result))

    def test_temporal_exclusion_and_no_threshold_relaxation(self):
        zeros=np.zeros((3,3,5))
        setting={'max_sensor_rotation_deg':10,'max_sensor_acc_difference_m_s2':1}
        args=([ (zeros,zeros) ]*4,setting,300,5,np.arange(3))
        self.assertEqual(select_matches(np.array([0,30,60]),*args),[])
        high=np.ones_like(zeros)*11
        self.assertEqual(select_matches(np.array([0,300,600]),[(high,zeros)]*4,setting,300,5,np.arange(3)),[])


if __name__=='__main__':unittest.main()
