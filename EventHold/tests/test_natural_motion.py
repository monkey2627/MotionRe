import io
import unittest
import numpy as np
from eventhold.natural_motion import TARGET, read_stream, posture_masks, audit_arrays


NAMES=['Pelvis','T8','LeftUpperLeg','LeftLowerLeg','LeftFoot','RightUpperLeg','RightLowerLeg','RightFoot']


def positions(sitting=True,n=1):
    p=np.zeros((n,8,3));p[:,0,2]=1;p[:,1,2]=1.5
    for hip,knee,foot,side in [(2,3,4,-.1),(5,6,7,.1)]:
        p[:,hip]=[0,side,1]
        p[:,knee]=[.4,side,1] if sitting else [0,side,.6]
        p[:,foot]=[.4,side,.6] if sitting else [0,side,.2]
    return p


class NaturalMotionTests(unittest.TestCase):
    def test_geometry_distinguishes_sitting_proxy_from_straight_legs(self):
        sit,stand,_=posture_masks(np.concatenate([positions(True),positions(False)]),NAMES)
        np.testing.assert_array_equal(sit,[True,False])
        np.testing.assert_array_equal(stand,[False,True])
        bad=positions();bad[:,3]=bad[:,2]
        self.assertFalse(posture_masks(bad,NAMES)[0].any())

    def test_reader_maps_sensor_labels_and_skips_calibration(self):
        names=list(reversed(TARGET))+['Head']
        sensors=''.join('<sensor label="'+n+'"/>' for n in names)
        segments=''.join('<segment label="'+n+'"/>' for n in NAMES)
        q=' '.join(['1 0 0 0']*6);gq=' '.join(['1 0 0 0']*8)
        acc=' '.join(str(i) for i in range(18));pos=' '.join(['0']*24)
        frame='<frame type="normal" index="0" time="0"><sensorOrientation>'+q+'</sensorOrientation><sensorFreeAcceleration>'+acc+'</sensorFreeAcceleration><orientation>'+gq+'</orientation><position>'+pos+'</position></frame>'
        xml='<mvnx xmlns="urn:fixture"><subject frameRate="240"><segments>'+segments+'</segments><sensors>'+sensors+'</sensors><frames><frame type="tpose"/>'+frame+'</frames></subject></mvnx>'
        a,m=read_stream(io.BytesIO(xml.encode()))
        self.assertEqual(m['selected_sensor_names'],TARGET)
        self.assertEqual(m['calibration_frame_types'],['tpose'])
        self.assertEqual(a['sensor_orientation_wxyz'].shape,(1,5,4))
        np.testing.assert_array_equal(a['sensor_free_acceleration'][0,:,0],[12,9,6,3,0])

    def test_time_gap_splits_posture_candidate(self):
        n=3000;fps=60
        a={'time_ms':np.arange(n)*1000/fps,'frame_index':np.arange(n),
           'sensor_orientation_wxyz':np.tile([1.,0,0,0],(n,5,1)),
           'sensor_free_acceleration':np.zeros((n,5,3)),
           'segment_orientation_wxyz':np.tile([1.,0,0,0],(n,8,1)),
           'segment_position':positions(n=n)}
        meta={'frame_rate':fps,'segment_names':NAMES}
        first,rows=audit_arrays(a,meta)
        self.assertGreater(first['max_seated_like_seconds'],49)
        a['time_ms'][1500:]+=1000
        second,rows=audit_arrays(a,meta)
        self.assertLess(second['max_seated_like_seconds'],26)
        self.assertFalse(second['sensor_calibration_from_reference_performed'])


if __name__=='__main__':unittest.main()
