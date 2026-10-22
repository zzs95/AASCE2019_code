"""Native AASCE layout, normalized CSV, preflight and related-view splitting."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.io import savemat

import train
from training.data import (Sample, audit_samples, case_id, check_split, discover, load_sample,
                           split_training)


class NativeDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "boostnet_labeldata"
        self.images = self.root / "data" / "training"
        self.labels = self.root / "labels" / "training"
        self.images.mkdir(parents=True)
        self.labels.mkdir(parents=True)
        self.points = np.array([[[20, 8 + j * 6], [55, 8 + j * 6],
                                 [20, 12 + j * 6], [55, 12 + j * 6]] for j in range(17)], np.float32)
        records = []
        for case in range(1, 7):
            for view in ("A", "B"):
                name = f"sunhl-1th-25-Jul-2016-{case} {view} AP.png"
                yy, xx = np.mgrid[:128, :80]
                image = (xx + yy / 2 + case * 3 + (view == "B")).astype(np.uint8)
                Image.fromarray(image).save(self.images / name)
                points = self.points + [case, 0]
                savemat(self.labels / (name + ".mat"), {"p2": points.reshape(68, 2)})
                flat = points.reshape(68, 2)
                records.append((name, np.r_[flat[:, 0] / 80, flat[:, 1] / 128]))
        # CSV order intentionally differs from directory order.
        records.reverse()
        with (self.labels / "filenames.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            csv.writer(stream).writerows([[name] for name, _ in records])
        np.savetxt(self.labels / "landmarks.csv", np.array([row for _, row in records]), delimiter=",")

    def tearDown(self):
        self.temp.cleanup()

    def test_native_discovery_csv_name_join_and_mat_preference(self):
        mat = discover(self.root, "training")
        direct = discover(self.images)
        self.assertEqual(mat, direct)
        self.assertTrue(all(s.row is None for s in mat))
        normalized = discover(self.root, "training", "csv")
        for pixel, row in zip(mat, normalized):
            self.assertEqual(pixel.image, row.image)
            _, p = load_sample(pixel)
            _, q = load_sample(row)
            np.testing.assert_allclose(p, q, atol=1e-5)
            # MATLAB pixel indexing flag must not shift normalized CSV targets.
            _, q_one = load_sample(row, coordinate_base=1)
            np.testing.assert_array_equal(q, q_one)
        mat[0].label.unlink()
        mixed = discover(self.root, "training")
        self.assertIsNotNone(mixed[0].row)
        self.assertTrue(all(s.row is None for s in mixed[1:]))

    def test_group_split_is_deterministic_and_keeps_related_views_together(self):
        samples = discover(self.root, "training")
        a, b = split_training(samples, 0.34, seed=2019)
        a2, b2 = split_training(samples, 0.34, seed=2019)
        self.assertEqual((a, b), (a2, b2))
        self.assertEqual(len(a) + len(b), 12)
        self.assertFalse({case_id(s) for s in a} & {case_id(s) for s in b})
        check_split(a, b)
        with self.assertRaisesRegex(ValueError, "case groups overlap"):
            check_split([samples[0], Sample(Path("unknown.png"), Path("dummy.mat"))], [samples[1]])
        base = Sample(Path("sunhl-1th-01-Mar-2017-310 a ap.jpg"), Path("dummy.mat"))
        duplicate = Sample(Path("sunhl-1th-01-Mar-2017-310 A AP (2).jpg"), Path("dummy.mat"))
        self.assertEqual(case_id(base), case_id(duplicate))
        coronal_a = Sample(Path("sunhl-1th-20-Jul-2016-coronal1.jpg"), Path("dummy.mat"))
        coronal_b = Sample(Path("sunhl-1th-20-Jul-2016-coronal17.jpg"), Path("dummy.mat"))
        self.assertEqual(case_id(coronal_a), case_id(coronal_b))

    def test_bad_annotations_are_reported_without_mutating_or_reordering(self):
        samples = discover(self.root, "training")
        bad = self.points.reshape(68, 2).copy()
        bad[0, 1] = 150
        savemat(samples[0].label, {"p2": bad})
        savemat(samples[1].label, {"p2": np.roll(self.points, 1, axis=0).reshape(68, 2)})
        savemat(samples[2].label, {"p2": np.ones((76, 2))})
        before = [hashlib.sha256(s.label.read_bytes()).hexdigest() for s in samples]
        valid, audit = audit_samples(samples)
        self.assertEqual(len(valid), 9)
        self.assertEqual(len(audit["rejected"]), 3)
        messages = " ".join(r["reason"] for r in audit["rejected"])
        self.assertIn("within", messages)
        self.assertIn("superior to inferior", messages)
        self.assertIn("got (76, 2)", messages)
        self.assertEqual(before, [hashlib.sha256(s.label.read_bytes()).hexdigest() for s in samples])

    def test_cli_data_check_skip_and_strict_modes(self):
        sample = discover(self.root, "training")[0]
        bad = self.points.reshape(68, 2).copy()
        bad[0, 1] = 150
        savemat(sample.label, {"p2": bad})
        output = self.root.parent / "checked"
        base = ["--stage", "wsm", "--data-root", str(self.root.parent), "--check-data", "--output-dir", str(output)]
        with patch("train.build_model", side_effect=AssertionError("Data check must not construct a model")):
            with self.assertRaisesRegex(ValueError, "invalid samples"):
                train.main(base)
            audit = train.main(base + ["--invalid-samples", "skip"])
        self.assertEqual(audit["source_training"]["accepted"], 11)
        self.assertEqual(len(audit["source_training"]["rejected"]), 1)
        split = json.loads((output / "split.json").read_text(encoding="utf-8"))
        self.assertEqual(len(split["train"]) + len(split["val"]), 11)
        self.assertFalse((output / "last.pth").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
