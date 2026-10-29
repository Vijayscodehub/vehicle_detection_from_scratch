"""Check reproducibility and leakage boundaries of the actual selected dataset."""

import csv
import hashlib
import unittest
from collections import Counter, defaultdict

from src.prepare_data import ROOT, spaced_indices


class PreparationTests(unittest.TestCase):
    def test_spacing(self):
        self.assertEqual(spaced_indices(203, 20)[0], 0)
        self.assertEqual(spaced_indices(203, 20)[-1], 202)
        self.assertEqual(len(set(spaced_indices(203, 20))), 20)
        with self.assertRaises(ValueError):
            spaced_indices(19, 20)

    def test_frozen_manifest(self):
        with (ROOT / "data/manifest.csv").open() as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(Counter(row["split"] for row in records),
                         {"train": 300, "validation": 100, "test": 100})
        self.assertEqual(len({row["sha256"] for row in records}), 500)
        self.assertEqual(len({row["image_id"] for row in records}), 500)
        prefixes = defaultdict(set)
        for row in records:
            prefixes[row["video_id"].split("-")[0]].add(row["split"])
            path = ROOT / row["image_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
        self.assertEqual(len(prefixes), 25)
        self.assertTrue(all(len(splits) == 1 for splits in prefixes.values()))
        pilot = [row for row in records if row["pilot"] == "1"]
        self.assertEqual(len(pilot), 75)
        self.assertEqual({row["split"] for row in pilot}, {"train"})
        self.assertEqual(len({row["video_id"] for row in pilot}), 6)


if __name__ == "__main__":
    unittest.main()