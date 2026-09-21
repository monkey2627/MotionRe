import json
import tempfile
import unittest
from pathlib import Path


class SyntheticLossoProtocolTests(unittest.TestCase):
    def test_existing_extension_split_is_source_grouped(self):
        split = Path("reports/g0/five_imu_synthetic_extension_g1_split/split.json")
        self.assertTrue(split.exists())
        data = json.loads(split.read_text())
        seen = {}
        for row in data["rows"]:
            old = seen.setdefault(row["subject_key"], row["split"])
            self.assertEqual(old, row["split"])
        self.assertTrue(data["no_test_claim"])


if __name__ == "__main__":
    unittest.main()
