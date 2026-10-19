import csv
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ColumnType(str, Enum):
    NUMERICAL = "NUMERICAL"
    CATEGORICAL = "CATEGORICAL"
    DATETIME = "DATETIME"
    CONSTANT = "CONSTANT"
    UNIQUE_ID = "UNIQUE_ID"


@dataclass
class ColumnProfile:
    name: str
    inferred_type: ColumnType
    total_count: int
    missing_count: int
    missing_ratio: float
    distinct_count: int
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.inferred_type.value,
            "total_count": self.total_count,
            "missing_count": self.missing_count,
            "missing_ratio": round(self.missing_ratio, 4),
            "distinct_count": self.distinct_count,
            "stats": self.stats,
        }


@dataclass
class DatasetSummary:
    row_count: int
    column_count: int
    total_cells: int
    total_missing_cells: int
    missing_cell_ratio: float
    duplicate_rows: int
    columns: List[ColumnProfile] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "total_cells": self.total_cells,
            "total_missing_cells": self.total_missing_cells,
            "missing_cell_ratio": round(self.missing_cell_ratio, 4),
            "duplicate_rows": self.duplicate_rows,
            "columns": [c.to_dict() for c in self.columns],
        }


class DataProfiler:
    """Automated Exploratory Data Analysis & statistical profiling engine."""

    @staticmethod
    def profile_csv(file_path: str) -> DatasetSummary:
        rows: List[List[str]] = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            for row in reader:
                if row:
                    rows.append(row)

        return DataProfiler.profile_matrix(headers, rows)

    @staticmethod
    def profile_matrix(headers: List[str], rows: List[List[Any]]) -> DatasetSummary:
        row_count = len(rows)
        col_count = len(headers)
        total_cells = row_count * col_count
        total_missing = 0

        # Check duplicates
        seen_rows = set()
        duplicate_count = 0
        for r in rows:
            t = tuple(r)
            if t in seen_rows:
                duplicate_count += 1
            else:
                seen_rows.add(t)

        column_profiles: List[ColumnProfile] = []

        for col_idx, col_name in enumerate(headers):
            raw_values = [r[col_idx] if col_idx < len(r) else None for r in rows]
            non_null = []
            missing_count = 0

            for v in raw_values:
                if v is None or str(v).strip() in ("", "nan", "null", "none", "na", "?"):
                    missing_count += 1
                else:
                    non_null.append(v)

            total_missing += missing_count
            missing_ratio = missing_count / max(row_count, 1)
            distinct_count = len(set(non_null))

            # Attempt numeric conversion
            numeric_vals: List[float] = []
            for v in non_null:
                try:
                    numeric_vals.append(float(v))
                except (ValueError, TypeError):
                    break

            is_numeric = len(numeric_vals) == len(non_null) and len(non_null) > 0

            stats: Dict[str, Any] = {}
            is_id_name = any(term in col_name.lower().split("_") or term == col_name.lower() for term in ["id", "uuid", "guid", "index", "pk"])
            if distinct_count == 1:
                col_type = ColumnType.CONSTANT
                stats["constant_value"] = non_null[0] if non_null else None
            elif distinct_count == row_count and (not is_numeric or is_id_name):
                col_type = ColumnType.UNIQUE_ID
            elif is_numeric:
                col_type = ColumnType.NUMERICAL

                numeric_vals.sort()
                n = len(numeric_vals)
                mean = sum(numeric_vals) / n
                variance = sum((x - mean) ** 2 for x in numeric_vals) / max(n - 1, 1)
                std = math.sqrt(variance)

                q25 = numeric_vals[int(n * 0.25)]
                median = numeric_vals[int(n * 0.50)]
                q75 = numeric_vals[int(n * 0.75)]
                iqr = q75 - q25
                outliers = sum(1 for x in numeric_vals if x < q25 - 1.5 * iqr or x > q75 + 1.5 * iqr)

                stats = {
                    "min": numeric_vals[0],
                    "q25": q25,
                    "median": median,
                    "q75": q75,
                    "max": numeric_vals[-1],
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "iqr": round(iqr, 4),
                    "outlier_count": outliers,
                }
            else:
                col_type = ColumnType.CATEGORICAL
                freq: Dict[str, int] = {}
                for v in non_null:
                    s = str(v)
                    freq[s] = freq.get(s, 0) + 1
                top_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
                stats = {
                    "top_categories": [
                        {"value": k, "count": v, "percentage": round(v / len(non_null), 4)}
                        for k, v in top_items
                    ]
                }

            column_profiles.append(
                ColumnProfile(
                    name=col_name,
                    inferred_type=col_type,
                    total_count=row_count,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                    distinct_count=distinct_count,
                    stats=stats,
                )
            )

        return DatasetSummary(
            row_count=row_count,
            column_count=col_count,
            total_cells=total_cells,
            total_missing_cells=total_missing,
            missing_cell_ratio=total_missing / max(total_cells, 1),
            duplicate_rows=duplicate_count,
            columns=column_profiles,
        )
