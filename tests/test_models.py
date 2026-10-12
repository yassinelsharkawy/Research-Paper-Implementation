import unittest
import numpy as np
import pandas as pd
from src.models import EBMModelWrapper


class TestModels(unittest.TestCase):
    def test_ebm_fit_predict_exact_additivity(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "feat1": np.random.randn(50),
            "feat2": np.random.randint(0, 10, 50),
            "feat3": np.random.randn(50)
        })
        y = 2.0 * X["feat1"] + 0.5 * X["feat2"] + 10.0
        
        wrapper = EBMModelWrapper(feature_names=["feat1", "feat2", "feat3"], random_state=42)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        
        self.assertEqual(len(preds), 50)
        self.assertFalse(np.isnan(preds).any())
        
        
        terms = wrapper.model.eval_terms(X)
        reconstructed = np.sum(terms, axis=1) + wrapper.model.intercept_
        np.testing.assert_allclose(preds, reconstructed, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
