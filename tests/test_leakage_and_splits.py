import unittest
import pandas as pd
from src.data_loader import load_dataset
from src.split_strategy import get_stratified_group_folds
from src.feature_groups import get_feature_groups


class TestLeakageAndSplits(unittest.TestCase):
    def setUp(self):
        self.df, _ = load_dataset()

    def test_group_cv_no_lineage_leakage(self):
        folds = get_stratified_group_folds(self.df, n_splits=5, seed=42)
        self.assertEqual(len(folds), 5)
        
        seen_lineages = set()
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_lineages = set(self.df.iloc[train_idx]["policy_lineage_id"])
            val_lineages = set(self.df.iloc[val_idx]["policy_lineage_id"])
            
            
            intersection = train_lineages.intersection(val_lineages)
            self.assertEqual(len(intersection), 0, f"Lineage leakage in fold {fold_idx}: {intersection}")

    def test_pair_members_together(self):
        folds = get_stratified_group_folds(self.df, n_splits=5, seed=42)
        paired_df = self.df[self.df["pair_id"].notna()]
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            val_set = set(val_idx)
            
            for pid, grp in paired_df.groupby("pair_id"):
                indices = set(grp.index)
                val_members = indices.intersection(val_set)
                self.assertTrue(
                    len(val_members) == 0 or len(val_members) == len(indices),
                    f"Pair {pid} was split across train/val in fold {fold_idx}"
                )

    def test_no_forbidden_metadata_features(self):
        m0, m1, m2, forbidden, _ = get_feature_groups()
        all_features = set(m0 + m1 + m2)
        
        forbidden_set = set(forbidden)
        leaked = all_features.intersection(forbidden_set)
        self.assertEqual(len(leaked), 0, f"Forbidden features found: {leaked}")
        
        
        gt_leaked = [f for f in all_features if f.startswith("gt_")]
        self.assertEqual(len(gt_leaked), 0, f"Ground truth leakage: {gt_leaked}")


if __name__ == "__main__":
    unittest.main()
