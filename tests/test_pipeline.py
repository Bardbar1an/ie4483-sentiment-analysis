"""Checks for leakage boundaries, data validation and submission alignment."""
import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment import build_model, load_data, prepare_training, read_table, save_predictions


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parent / (".test-" + uuid.uuid4().hex)
        self.folder.mkdir()

    def tearDown(self):
        assert self.folder.resolve().parent == Path(__file__).resolve().parent
        shutil.rmtree(self.folder)

    def test_csv_preserves_ids_and_rejects_bad_labels(self):
        path = self.folder / "data.csv"
        path.write_text("id,reviews,sentiments\n001,Good product,1\n002,Bad product,0\n", encoding="utf-8")
        data = load_data(path, True)
        self.assertEqual(data.id.tolist(), ["001", "002"])
        path.write_text("reviews,sentiments\nFine,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Labels must"):
            load_data(path, True)

    def test_test_labels_are_ignored(self):
        path = self.folder / "test.json"
        path.write_text(json.dumps([{"id": "x", "reviews": "A review", "sentiments": "hidden"}]), encoding="utf-8")
        data = load_data(path, False)
        self.assertNotIn("label", data.columns)

    def test_missing_text_is_rejected(self):
        path = self.folder / "data.json"
        path.write_text(json.dumps([{"reviews": None, "sentiments": 0}]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            load_data(path, True)

    def test_json_orientations(self):
        rows = [{"id": "01", "reviews": "Good", "sentiments": 1}]
        values = [rows, {"data": rows}, {"columns": list(rows[0]), "data": [list(rows[0].values())]},
                  {"row0": rows[0]}, {k: [v] for k, v in rows[0].items()},
                  {k: {"0": v} for k, v in rows[0].items()}]
        for i, value in enumerate(values):
            path = self.folder / f"format{i}.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(load_data(path, True).text.tolist(), ["Good"])
        path = self.folder / "rows.jsonl"
        path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        self.assertEqual(len(read_table(path)), 1)

    def test_duplicate_reviews_do_not_cross_split(self):
        data = pd.DataFrame({"text": [f"positive {i}" for i in range(8)] + [f"negative {i}" for i in range(8)],
                             "label": [1] * 8 + [0] * 8})
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        cleaned, train, val, removed = prepare_training(data, .25, 42)
        self.assertEqual(removed, 1)
        self.assertFalse(set(cleaned.iloc[train].duplicate_key) & set(cleaned.iloc[val].duplicate_key))
        _, train2, val2, _ = prepare_training(data, .25, 42)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(val, val2)
        data.loc[len(data) - 1, "label"] = 0
        with self.assertRaisesRegex(ValueError, "conflicting"):
            prepare_training(data, .25, 42)

    def test_validation_words_do_not_enter_vocabulary(self):
        pipeline = build_model("logistic", 42, 100, 2, 1, 8, .001, .001)
        pipeline.fit(["wonderful product", "awful product", "great purchase", "terrible purchase"], [1, 0, 1, 0])
        pipeline.predict(["uniquelyheldouttoken"])
        self.assertNotIn("uniquelyheldouttoken", pipeline.named_steps["tfidf"].vocabulary_)

    def test_submission_aligns_by_id_and_rejects_mismatch(self):
        data = pd.DataFrame({"id": ["001", "002"], "text": ["good", "bad"]})
        template = self.folder / "template.csv"
        template.write_text("id,label\n002,\n001,\n", encoding="utf-8")
        result = save_predictions(data, [1, 0], self.folder / "submission.csv", template)
        self.assertEqual(result.id.tolist(), ["002", "001"])
        self.assertEqual(result.label.tolist(), [0, 1])
        template.write_text("id,label\n003,\n001,\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            save_predictions(data, [1, 0], self.folder / "wrong.csv", template)

    def test_missing_ids_and_existing_outputs_are_not_silently_replaced(self):
        data = pd.DataFrame({"text": ["good", "bad"]})
        output = self.folder / "predictions.csv"
        with self.assertRaisesRegex(ValueError, "no IDs"):
            save_predictions(data, [1, 0], output)
        save_predictions(data, [1, 0], output, allow_generated_ids=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            save_predictions(data, [1, 0], output, allow_generated_ids=True)


if __name__ == "__main__":
    unittest.main()
