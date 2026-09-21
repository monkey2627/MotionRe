import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from eventhold.mounting import fit_mounting,apply_mounting,change_basis


class MountingTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.RandomState(2)
        self.bones=Rotation.random(100,random_state=self.rng).as_matrix().reshape(20,5,3,3)
        self.offset=Rotation.random(5,random_state=self.rng).as_matrix()
        self.sensors=self.bones@self.offset.swapaxes(-1,-2)
        self.names=tuple('abcde')

    def test_recovers_right_multiplication_without_refitting_future(self):
        c=fit_mounting(self.sensors[:5],self.bones[:5],self.names,provenance='commanded_pose',source_note='synthetic known-pose fixture')
        np.testing.assert_allclose(c.bone_to_sensor,self.offset,atol=1e-12)
        np.testing.assert_allclose(apply_mounting(self.sensors,c,self.names),self.bones,atol=1e-12)
        changed=self.sensors.copy();changed[10:]=np.eye(3)
        np.testing.assert_allclose(apply_mounting(changed,c,self.names)[:10],self.bones[:10],atol=1e-12)

    def test_reference_profile_refused_by_default(self):
        c=fit_mounting(self.sensors[:5],self.bones[:5],self.names,provenance='reference_assisted_diagnostic',source_note='synthetic reference diagnostic')
        with self.assertRaises(ValueError):apply_mounting(self.sensors,c,self.names)
        np.testing.assert_allclose(apply_mounting(self.sensors,c,self.names,allow_reference_diagnostic=True),self.bones,atol=1e-12)

    def test_sensor_identity_and_reflections_rejected(self):
        c=fit_mounting(self.sensors[:5],self.bones[:5],self.names,provenance='commanded_pose',source_note='fixture')
        with self.assertRaises(ValueError):apply_mounting(self.sensors,c,tuple(reversed(self.names)))
        bad=self.sensors.copy();bad[0,0,:,0]*=-1
        with self.assertRaises(ValueError):apply_mounting(bad,c,self.names)

    def test_basis_change_preserves_orientation_action_and_vector_norm(self):
        c=np.array([[0,1,0],[0,0,1],[1,0,0]])
        vectors=self.rng.randn(20,5,3)
        r,v=change_basis(self.bones,vectors,c)
        np.testing.assert_allclose(np.linalg.norm(v,axis=-1),np.linalg.norm(vectors,axis=-1))
        lhs=np.einsum('...ij,...j->...i',r,v)
        rhs=np.einsum('ij,...jk,...k->...i',c,self.bones,vectors)
        np.testing.assert_allclose(lhs,rhs,atol=1e-12)


if __name__=='__main__':unittest.main()
