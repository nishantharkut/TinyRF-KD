import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.wsnds.evidence_completion import analyze_fgds_calibration as analysis


class CalibrationMetricUnitTests(unittest.TestCase):
    def test_multiclass_brier_and_nll_match_explicit_formulas(self):
        probabilities = np.array(
            [[0.70, 0.20, 0.10], [0.10, 0.30, 0.60]], dtype=np.float64
        )
        labels = np.array([0, 2], dtype=np.int64)

        self.assertAlmostEqual(analysis.multiclass_brier_score(probabilities, labels), 0.20)
        self.assertAlmostEqual(
            analysis.negative_log_likelihood(probabilities, labels),
            -0.5 * (math.log(0.70) + math.log(0.60)),
        )

    def test_equal_width_ece_uses_top_label_confidence(self):
        probabilities = np.array(
            [
                [0.60, 0.40],
                [0.80, 0.20],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 1], dtype=np.int64)

        result = analysis.top_label_ece(probabilities, labels, n_bins=2)

        self.assertAlmostEqual(result["ece"], 0.20)
        self.assertEqual(result["nonempty_bin_count"], 1)
        self.assertEqual(result["bins"][1]["count"], 2)
        self.assertAlmostEqual(result["bins"][1]["mean_confidence"], 0.70)
        self.assertAlmostEqual(result["bins"][1]["accuracy"], 0.50)

    def test_adaptive_ece_partitions_sorted_confidences_without_dropping_rows(self):
        probabilities = np.array(
            [
                [0.55, 0.45],
                [0.60, 0.40],
                [0.65, 0.35],
                [0.70, 0.30],
                [0.75, 0.25],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)

        result = analysis.adaptive_top_label_ece(probabilities, labels, n_bins=2)

        self.assertEqual(sum(item["count"] for item in result["bins"]), 5)
        self.assertEqual([item["count"] for item in result["bins"]], [3, 2])

    def test_adaptive_ece_breaks_confidence_ties_by_source_row_index(self):
        probabilities = np.array(
            [[0.60, 0.40], [0.60, 0.40], [0.60, 0.40], [0.60, 0.40]],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 1, 0], dtype=np.int64)
        source_row_index = np.array([30, 10, 20, 40], dtype=np.int64)

        result = analysis.adaptive_top_label_ece(
            probabilities, labels, n_bins=2, source_row_index=source_row_index
        )

        self.assertEqual(result["tie_break"], "ascending source_row_index")
        self.assertAlmostEqual(result["bins"][0]["accuracy"], 0.0)
        self.assertAlmostEqual(result["bins"][1]["accuracy"], 1.0)

    def test_classwise_ece_returns_one_vs_rest_metrics_for_every_class(self):
        probabilities = np.array(
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.70, 0.20],
                [0.20, 0.20, 0.60],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 2], dtype=np.int64)

        result = analysis.classwise_ece(probabilities, labels, n_bins=3)

        self.assertEqual(len(result), 3)
        self.assertEqual([item["class_index"] for item in result], [0, 1, 2])
        self.assertTrue(all(0.0 <= item["ece"] <= 1.0 for item in result))

    def test_metrics_reject_probability_rows_that_do_not_sum_to_one(self):
        probabilities = np.array([[0.60, 0.30]], dtype=np.float64)
        labels = np.array([0], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "sum to one"):
            analysis.multiclass_brier_score(probabilities, labels)

    def test_nll_details_record_true_class_zeros_and_declared_floor(self):
        probabilities = np.array([[1.0, 0.0], [0.25, 0.75]], dtype=np.float64)
        labels = np.array([1, 1], dtype=np.int64)

        result = analysis.negative_log_likelihood_details(probabilities, labels)

        self.assertEqual(result["true_probability_zero_count"], 1)
        self.assertEqual(result["probability_floor"], analysis.NLL_FLOOR)
        self.assertIsNone(result["raw_nll"])
        self.assertTrue(math.isfinite(result["floored_nll"]))

    def test_prediction_loader_preserves_declared_class_order_and_row_identity(self):
        frame = pd.DataFrame(
            {
                "source_row_index": [11, 17],
                "true_label": [0, 2],
                "predicted_label": [0, 2],
                "probability_0_Blackhole": [0.70, 0.10],
                "probability_1_Flooding": [0.20, 0.20],
                "probability_2_Grayhole": [0.05, 0.60],
                "probability_3_Normal": [0.03, 0.05],
                "probability_4_TDMA": [0.02, 0.05],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            frame.to_csv(path, index=False)
            loaded = analysis.load_prediction_file(path, expected_rows=2)

        np.testing.assert_array_equal(loaded["source_row_index"], [11, 17])
        np.testing.assert_array_equal(loaded["true_label"], [0, 2])
        self.assertEqual(loaded["class_names"], list(analysis.CLASS_NAMES))
        np.testing.assert_allclose(
            loaded["probabilities"], frame.iloc[:, 3:].to_numpy(dtype=np.float64)
        )

    def test_paired_metric_summary_keeps_seed_as_statistical_unit(self):
        rows = []
        for offset, seed in enumerate(analysis.EXPECTED_SEEDS):
            scratch = 0.20 + 0.001 * offset
            rows.append(
                {
                    "seed": seed,
                    "student": "student_A",
                    "route": "scratch",
                    "brier_multiclass": scratch,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "student": "student_A",
                    "route": "rf_kd",
                    "brier_multiclass": scratch - 0.05,
                }
            )

        result = analysis.paired_metric_summary(
            pd.DataFrame(rows), "student_A", "brier_multiclass"
        )

        self.assertEqual(result["statistical_unit"], "paired model-training seed")
        self.assertEqual(result["paired_seed_count"], 10)
        self.assertAlmostEqual(result["improvement_mean"], 0.05)
        self.assertEqual(result["improved_seed_count"], 10)
        self.assertAlmostEqual(
            result["exact_paired_wilcoxon"]["p_value_two_sided"], 2.0 / 1024.0
        )

    def test_analysis_script_supports_direct_help_invocation(self):
        proc = subprocess.run(
            [sys.executable, str(analysis.SCRIPT_PATH), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--verify-existing", proc.stdout)


if __name__ == "__main__":
    unittest.main()
