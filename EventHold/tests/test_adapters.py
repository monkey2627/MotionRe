import dataclasses
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from eventhold.adapters import (eventhold_features,dynaip_features,pnp_from_native,
                               pnp_from_world_record,angular_velocity_world_causal)
from eventhold.dip import measurements, DIP_INDEX, split_for_subject
from eventhold.records import FIVE, PNP_SIX, DYNA_SIX, ImuRecord, causal_fill


def fixture(names=PNP_SIX,t=20,hz=60):
    rng=np.random.RandomState(1)
    r=Rotation.from_rotvec(rng.normal(size=(t*len(names),3))*.3).as_matrix().reshape(t,len(names),3,3)
    return ImuRecord(np.arange(t)/hz,tuple(names),r,rng.normal(size=(t,len(names),3)),
                     np.ones((t,len(names)),bool),'synthetic_contract_fixture')


class AdapterTests(unittest.TestCase):
    def test_wrong_locations_rejected(self):
        r=fixture(('pelvis','left_forearm','right_forearm','left_thigh','right_thigh'))
        with self.assertRaises(ValueError):eventhold_features(r)

    def test_hidden_head_does_not_affect_five_features(self):
        r=fixture(); a=eventhold_features(r)
        r.orientation[:,r.names.index('head')]=np.eye(3)
        r.acceleration[:,r.names.index('head')]=1e6
        np.testing.assert_array_equal(a,eventhold_features(r))

    def test_permutation_by_names(self):
        r=fixture(); np.testing.assert_allclose(eventhold_features(r),eventhold_features(r.select(tuple(reversed(r.names)))))

    def test_wrong_rate_fails_native(self):
        with self.assertRaises(ValueError):dynaip_features(fixture(hz=30))
        with self.assertRaises(ValueError):pnp_from_world_record(fixture(hz=30))

    def test_five_not_silently_padded_for_six_method(self):
        with self.assertRaises(ValueError):dynaip_features(fixture(FIVE))
        with self.assertRaises(ValueError):pnp_from_world_record(fixture(FIVE))

    def test_raw_specific_force_rejected(self):
        with self.assertRaises(ValueError):eventhold_features(dataclasses.replace(fixture(),acceleration_kind='specific_force_sensor'))

    def test_so3_validation(self):
        r=fixture();r.orientation[2,1]=0
        with self.assertRaises(ValueError):r.validate()

    def test_causal_filling(self):
        r=fixture();r.orientation[3:6,0]=np.nan;r.acceleration[0,1]=np.nan
        ori,a,v=causal_fill(r.orientation,r.acceleration)
        self.assertFalse(v[0,1]);self.assertFalse(v[4,0])
        np.testing.assert_equal(ori[4,0],r.orientation[2,0])
        changed=r.orientation.copy();changed[6:,0]=np.eye(3)
        np.testing.assert_array_equal(ori[:6],causal_fill(changed,r.acceleration)[0][:6])

    def test_native_methods_reject_missing(self):
        r=fixture();r.valid[2,1]=False
        with self.assertRaises(ValueError):dynaip_features(r)

    def test_future_does_not_change_prefix(self):
        r=fixture();w,_=angular_velocity_world_causal(r.orientation,r.timestamps)
        f=eventhold_features(r);r.orientation[10:]=np.eye(3);r.acceleration[10:]=50
        np.testing.assert_array_equal(w[:10],angular_velocity_world_causal(r.orientation,r.timestamps)[0][:10])
        np.testing.assert_array_equal(f[:10],eventhold_features(r)[:10])

    def test_world_gyro_frame(self):
        # Body local z rotated to world x: rotating about world x must not
        # accidentally return body-coordinate z (the old bridge's failure).
        t=np.arange(30)/60
        r=(Rotation.from_rotvec(np.c_[t,np.zeros((30,2))]).as_matrix()
           @ Rotation.from_euler('y',90,degrees=True).as_matrix())[:,None]
        w,mask=angular_velocity_world_causal(r,t)
        np.testing.assert_allclose(w[1:,0],np.tile([1,0,0],(29,1)),atol=1e-12)
        self.assertFalse(mask[0,0])

    def test_stationary_native_gravity(self):
        r=fixture(t=3);rs=r.orientation
        g=np.array([0.,-9.8,0.]);aS=np.einsum('tnji,j->tni',rs,-g)
        a,w,R=pnp_from_native(aS,np.zeros_like(aS),rs,np.tile(np.eye(3),(6,1,1)),np.tile(np.eye(3),(6,1,1)),g)
        np.testing.assert_allclose(a,0,atol=1e-12)
        np.testing.assert_allclose(w,0,atol=1e-12)
        np.testing.assert_allclose(R,rs)

    def test_world_record_does_not_add_gravity(self):
        r=fixture();np.testing.assert_array_equal(pnp_from_world_record(r)['a'],r.acceleration)

    def test_original_DIP_mapping_and_gt_isolation(self):
        d={'imu_ori':np.tile(np.eye(3),(10,17,1,1)),
           'imu_acc':np.tile(np.arange(17)[None,:,None],(10,1,3)).astype(float),
           'gt':np.zeros((10,72))}
        r=measurements(d,'fixture');np.testing.assert_equal(r.acceleration[0,:,0],[2,7,8,11,12])
        before=eventhold_features(r);d['gt'][:]=1e5
        np.testing.assert_array_equal(before,eventhold_features(measurements(d,'fixture')))

    def test_test_subjects_locked(self):
        self.assertEqual(split_for_subject('s_09'),'test_locked')
        self.assertEqual(split_for_subject('s_07'),'validation')


if __name__=='__main__':unittest.main()
