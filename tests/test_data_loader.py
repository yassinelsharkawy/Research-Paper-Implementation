import os
import unittest
import pandas as pd
from src.data_loader import load_dataset
from src.schema import validate_dataset_invariants


class TestDataLoader(unittest.TestCase):
    def test_load_and_invariants(self):
        df, path = load_dataset()
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 534)
        
        invariants = validate_dataset_invariants(df)
        self.assertEqual(invariants["total_samples"], 534)
        self.assertEqual(invariants["n_type_a_pairs"], 208)
        self.assertEqual(invariants["n_type_b_pairs"], 39)
        self.assertEqual(invariants["n_unpaired"], 40)
        self.assertEqual(invariants["n_scenario_families"], 13)


if __name__ == "__main__":
    unittest.main()
