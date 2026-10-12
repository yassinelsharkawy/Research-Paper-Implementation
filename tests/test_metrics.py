import unittest
import numpy as np
from src.evaluate import compute_metrics
from src.triage_analysis import compute_ndcg_at_k, compute_precision_recall_at_k


class TestMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0, 52.0])
        
        m = compute_metrics(y_true, y_pred, k_values=[2, 4])
        self.assertAlmostEqual(m["MAE"], 2.0)
        self.assertGreater(m["Spearman_rho"], 0.9)
        self.assertIn("NDCG@2", m)
        self.assertIn("NDCG@4", m)

    def test_triage_precision_recall(self):
        y_binary = np.array([1, 1, 0, 0, 0])
        y_score = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        prec, rec = compute_precision_recall_at_k(y_binary, y_score, k=2)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        
        prec3, rec3 = compute_precision_recall_at_k(y_binary, y_score, k=3)
        self.assertAlmostEqual(prec3, 2/3)
        self.assertEqual(rec3, 1.0)


if __name__ == "__main__":
    unittest.main()
