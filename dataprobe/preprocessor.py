import math
from typing import Any, Dict, List, Optional, Tuple
from .profiler import ColumnProfile, ColumnType


class AutoPreprocessor:
    """Automated tabular data cleaning, imputation, and feature transformation pipeline."""

    def __init__(self):
        self.medians: Dict[str, float] = {}
        self.modes: Dict[str, str] = {}
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.category_maps: Dict[str, Dict[str, int]] = {}
        self.selected_features: List[str] = []

    def fit_transform(
        self,
        headers: List[str],
        rows: List[List[Any]],
        target_col: str,
        profiles: List[ColumnProfile],
    ) -> Tuple[List[str], List[List[float]], List[float]]:
        """Fit preprocessor parameters on dataset and transform into numeric matrix."""
        col_map = {p.name: p for p in profiles}
        target_idx = headers.index(target_col)

        # 1. Feature selection (exclude target, constants, unique IDs)
        self.selected_features = []
        for p in profiles:
            if p.name == target_col:
                continue
            if p.inferred_type in (ColumnType.CONSTANT, ColumnType.UNIQUE_ID):
                continue
            if p.missing_ratio > 0.85:
                continue  # discard columns with >85% missing
            self.selected_features.append(p.name)

        # 2. Compute statistics for imputation & normalization
        for name in self.selected_features:
            p = col_map[name]
            idx = headers.index(name)
            col_vals = [r[idx] for r in rows if str(r[idx]).strip() not in ("", "nan", "null", "none")]

            if p.inferred_type == ColumnType.NUMERICAL:
                num_vals = [float(v) for v in col_vals]
                if num_vals:
                    num_vals.sort()
                    med = num_vals[len(num_vals) // 2]
                    mean = sum(num_vals) / len(num_vals)
                    variance = sum((x - mean) ** 2 for x in num_vals) / max(len(num_vals) - 1, 1)
                    std = math.sqrt(variance) or 1.0
                else:
                    med, mean, std = 0.0, 0.0, 1.0
                self.medians[name] = med
                self.means[name] = mean
                self.stds[name] = std
            else:
                # Mode for categorical
                freq: Dict[str, int] = {}
                for v in col_vals:
                    freq[str(v)] = freq.get(str(v), 0) + 1
                mode_val = max(freq.items(), key=lambda x: x[1])[0] if freq else "missing"
                self.modes[name] = mode_val
                # Discrete label encoding map
                unique_cats = sorted(list(set(str(v) for v in col_vals)))
                self.category_maps[name] = {cat: i for i, cat in enumerate(unique_cats)}

        # 3. Transform rows
        transformed_X: List[List[float]] = []
        transformed_y: List[float] = []

        for r in rows:
            feature_vector: List[float] = []
            for name in self.selected_features:
                p = col_map[name]
                idx = headers.index(name)
                val = r[idx] if idx < len(r) else None

                if str(val).strip() in ("", "nan", "null", "none"):
                    if p.inferred_type == ColumnType.NUMERICAL:
                        norm_val = (self.medians[name] - self.means[name]) / self.stds[name]
                    else:
                        norm_val = float(self.category_maps[name].get(self.modes[name], 0))
                else:
                    if p.inferred_type == ColumnType.NUMERICAL:
                        norm_val = (float(val) - self.means[name]) / self.stds[name]
                    else:
                        norm_val = float(self.category_maps[name].get(str(val), -1))

                feature_vector.append(round(norm_val, 4))

            # Target value
            raw_target = r[target_idx]
            try:
                target_val = float(raw_target)
            except ValueError:
                target_val = 1.0 if str(raw_target).lower() in ("yes", "true", "1", "positive", "phish") else 0.0

            transformed_X.append(feature_vector)
            transformed_y.append(target_val)

        return (self.selected_features, transformed_X, transformed_y)
