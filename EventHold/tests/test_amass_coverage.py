import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from eventhold.audit_amass_coverage import ALIGN, fk, screen


class AmassCoverageTests(unittest.TestCase):
    def fixture(self, fps=60):
        parents=[None,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
        rest=np.zeros((24,3))
        rest[1]=[.1,0,0];rest[2]=[-.1,0,0]
        rest[4]=[.1,-.4,0];rest[5]=[-.1,-.4,0]
        rest[7]=[.1,-.8,0];rest[8]=[-.1,-.8,0];rest[9]=[0,.5,0]
        pose=np.zeros((14*fps,24,3));pose[:,0]=Rotation.from_matrix(ALIGN.T).as_rotvec()
        pose[2*fps:12*fps,[1,2],0]=-np.pi/2
        pose[2*fps:12*fps,[4,5],0]=np.pi/2
        return pose.reshape(-1,72),np.zeros(10),fps,rest,np.zeros((24,3,10)),parents

    def test_known_stand_sit_stand_detected_at_native_and_screen_rate(self):
        args=self.fixture()
        for hz in [10,60]:
            rows,_=screen(*args,screen_hz=hz)
            seated=[x for x in rows if x['kind']=='seated_like']
            self.assertEqual(len(seated),1)
            self.assertTrue(seated[0]['complete_transition_proxy'])
            self.assertAlmostEqual(seated[0]['duration_seconds'],10)
            self.assertEqual(seated[0]['start_frame'],120)

    def test_standing_is_not_seated_and_59hz_is_not_relabelled(self):
        args=list(self.fixture(59));args[0][:,3:]=0
        rows,hz=screen(*args)
        self.assertEqual(rows,[])
        self.assertAlmostEqual(hz,59/6)


if __name__=='__main__':unittest.main()
