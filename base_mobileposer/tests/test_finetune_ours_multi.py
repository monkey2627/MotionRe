import tempfile
import unittest
from pathlib import Path
from mobileposer.finetune_ours_multi import discover, frame_average


class MultiCaptureTests(unittest.TestCase):
    def test_recursive_distinct_capture_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('1', '动作A/1', '动作B/1'):
                folder = root/name
                folder.mkdir(parents=True)
                (folder/'manifest.json').write_text('{}')
            found = discover(root)
            self.assertEqual(len(found), 3)
            self.assertEqual(len({key for key, _ in found}), 3)
            self.assertEqual(found, discover(root))

    def test_selection_uses_all_sequences_weighted_by_frames(self):
        score = frame_average([(100, {'error': 2.}), (300, {'error': 6.})])
        self.assertEqual(score['error'], 5.)

    def test_empty_root_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'No manifest'):
                discover(Path(tmp))


if __name__ == '__main__':
    unittest.main()
