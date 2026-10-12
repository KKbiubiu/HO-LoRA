import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summary", ROOT / "tools/summarize_results.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ResultTests(unittest.TestCase):
    def test_recorded_delta_and_ablation_differences(self):
        rows = {r["experiment"]: r for r in summary.summarize(ROOT / "results/raw")}
        self.assertAlmostEqual(rows["v12"]["texture_delta"], 1.1111409589648251)
        self.assertAlmostEqual(rows["v12"]["texture_delta"] - rows["ablation_3step"]["texture_delta"], 1.1622697340945405)
        self.assertAlmostEqual(rows["v12"]["texture_delta"] - rows["ablation_no_ortho"]["texture_delta"], 0.020245436330637023)

    def test_invalid_metric_is_rejected(self):
        original = json.loads((ROOT / "results/raw/eval_v12.json").read_text(encoding="utf-8"))
        for bad in (float("nan"), float("inf"), True, "1.0"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                data = copy.deepcopy(original)
                data["per_variant"]["texture_only"]["gram_dist"]["mean"] = bad
                Path(directory, "eval_baseline_standard.json").write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    summary.summarize(Path(directory))

    def test_committed_summaries_match_raw_evidence(self):
        for name, content in summary.render(summary.summarize(ROOT / "results/raw")).items():
            self.assertEqual((ROOT / "results" / name).read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
