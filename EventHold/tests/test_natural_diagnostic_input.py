import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import freqz
from eventhold.natural_diagnostic_input import BASIS, TAPS, lower_reference, causal_decimate


class NaturalDiagnosticInputTests(unittest.TestCase):
    def fixture(self, n=800):
        r=Rotation.from_euler('z',np.arange(n)*.001).as_matrix()[:,None]
        a=np.random.RandomState(3).randn(n,1,3)
        return r,a,np.arange(n),np.floor(np.arange(n)*1000/240)

    def test_future_perturbations_cannot_change_emitted_samples(self):
        r,a,fi,t=self.fixture();out=causal_decimate(r,a,fi,t)
        r[401:]=np.eye(3);a[401:]=999
        changed=causal_decimate(r,a,fi,t);mask=out['source_indices']<=400
        for key in ['orientation','acceleration']:
            np.testing.assert_allclose(out[key][mask],changed[key][mask],atol=1e-12)

    def test_dc_preservation_rotation_delay_and_availability(self):
        r,a,fi,t=self.fixture();a[:]=[1,2,3];out=causal_decimate(r,a,fi,t)
        np.testing.assert_allclose(out['acceleration'],np.broadcast_to([1,2,3],out['acceleration'].shape),atol=1e-12)
        expected=r[out['source_indices']-32]
        np.testing.assert_allclose(out['orientation'],expected,atol=1e-12)
        np.testing.assert_allclose(np.diff(out['nominal_availability_seconds']),1/60,atol=1e-12)
        self.assertAlmostEqual(out['signal_delay_seconds'],32/240)

    def test_stopband_and_gap_rejection(self):
        f,h=freqz(TAPS,worN=8192,fs=240)
        self.assertLess(np.max(np.abs(h[f>=30])),.01)
        r,a,fi,t=self.fixture();fi[200:]+=1
        with self.assertRaises(ValueError):causal_decimate(r,a,fi,t)

    def test_lower_mapping_known_flexion_and_world_heading_invariance(self):
        names=['Pelvis','LeftUpperLeg','RightUpperLeg','LeftLowerLeg','RightLowerLeg']
        g=np.tile(np.eye(3),(1,5,1,1));hip=Rotation.from_euler('y',-70,degrees=True).as_matrix()
        knee=Rotation.from_euler('y',90,degrees=True).as_matrix()
        g[:,1]=hip;g[:,3]=hip@knee
        loc=lower_reference(g,names)
        np.testing.assert_allclose(loc[0,0],BASIS@hip@BASIS.T,atol=1e-12)
        np.testing.assert_allclose(loc[0,2],BASIS@knee@BASIS.T,atol=1e-12)
        heading=Rotation.from_euler('z',1.1).as_matrix()
        np.testing.assert_allclose(lower_reference(heading@g,names),loc,atol=1e-12)


if __name__=='__main__':unittest.main()
