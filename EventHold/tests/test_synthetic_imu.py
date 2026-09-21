import unittest
import numpy as np
from eventhold.synthetic_imu import body_pose,synthesize_at_60hz
from eventhold.freeze_amass_split import assign_groups
from eventhold.stratify_amass_split import stratify


class SyntheticImuTests(unittest.TestCase):
    def test_accad_action_folders_cannot_cross_split(self):
        rows=[{'dataset':'ACCAD','subject_key':'ACCAD/'+s,'group_key':s} for s in ['Female1General','Female1Gestures','Male1Walking']]
        rows += [{'dataset':'Other','subject_key':str(i),'group_key':str(i)} for i in range(10)]
        out=stratify(rows,{'1','2','3'})
        self.assertEqual({r['split'] for r in out if r['dataset']=='ACCAD'},{'train'})
        self.assertEqual(len({r['group_key'] for r in out if r['dataset']=='ACCAD'}),1)
        self.assertTrue(any(r['split']=='development' and r['subject_key'] in {'1','2','3'} for r in out))

    def test_quadratic_acceleration_and_time_alignment(self):
        for fps in [60,120]:
            t=np.arange(600)/fps;r=np.tile(np.eye(3),(600,5,1,1))
            a=np.array([1.,-2.,3.]);v=np.broadcast_to(.5*t[:,None,None]**2*a,(600,5,3)).copy()
            out=synthesize_at_60hz(r,v,fps)
            np.testing.assert_allclose(out['acceleration'],np.broadcast_to(a,out['acceleration'].shape),atol=1e-5)
            np.testing.assert_allclose(np.diff(out['timestamps']),1/60,atol=1e-12)
            self.assertEqual(out['native_center_indices'][0],32+4*(fps//60))

    def test_constant_velocity_has_no_gravity_or_boundary_spikes(self):
        r=np.tile(np.eye(3),(600,5,1,1));v=np.broadcast_to(np.arange(600)[:,None,None]/120,(600,5,3))
        out=synthesize_at_60hz(r,v,120)
        np.testing.assert_allclose(out['acceleration'],0,atol=1e-8)
        with self.assertRaises(ValueError):synthesize_at_60hz(r,v,59.94)

    def test_smplh_hands_are_explicitly_mapped(self):
        p=np.arange(156)[None].astype(float)
        out=body_pose(p)
        np.testing.assert_equal(out[0,22],p.reshape(52,3)[22])
        np.testing.assert_equal(out[0,23],p.reshape(52,3)[37])
        with self.assertRaises(ValueError):body_pose(np.zeros((2,72)))

    def test_duplicate_sources_merge_subjects_and_order_is_irrelevant(self):
        rows=[{'subject_key':'a','sha256':'x'},{'subject_key':'b','sha256':'x'},
              {'subject_key':'b','sha256':'y'},{'subject_key':'c','sha256':'y'}]
        a=assign_groups(rows);b=assign_groups(list(reversed(rows)))
        self.assertEqual(len(set(x['group_key'] for x in a)),1)
        self.assertEqual({x['split'] for x in a},{x['split'] for x in b})


if __name__=='__main__':unittest.main()
