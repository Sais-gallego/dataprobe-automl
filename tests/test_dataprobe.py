import unittest
from dataprobe.benchmark import AutoMLBenchmark
from dataprobe.preprocessor import AutoPreprocessor
from dataprobe.profiler import ColumnType, DataProfiler


class TestDataProbeAutoML(unittest.TestCase):
    def setUp(self):
        self.headers = ["id", "age", "salary", "department", "is_promoted"]
        self.rows = [
            ["1", "25", "50000", "IT", "0"],
            ["2", "30", "65000", "Sales", "0"],
            ["3", "45", "120000", "Executive", "1"],
            ["4", "38", "95000", "IT", "1"],
            ["5", "29", "58000", "Marketing", "0"],
            ["6", "52", "140000", "Executive", "1"],
            ["7", "23", "45000", "Sales", "0"],
            ["8", "41", "105000", "IT", "1"],
        ]

    def test_profiling_inference(self):
        summary = DataProfiler.profile_matrix(self.headers, self.rows)
        self.assertEqual(summary.row_count, 8)
        self.assertEqual(summary.column_count, 5)

        col_dict = {c.name: c for c in summary.columns}
        self.assertEqual(col_dict["id"].inferred_type, ColumnType.UNIQUE_ID)
        self.assertEqual(col_dict["age"].inferred_type, ColumnType.NUMERICAL)
        self.assertEqual(col_dict["salary"].inferred_type, ColumnType.NUMERICAL)
        self.assertEqual(col_dict["department"].inferred_type, ColumnType.CATEGORICAL)
        self.assertAlmostEqual(col_dict["age"].stats["mean"], 35.375, places=2)

    def test_preprocessor_and_automl_pipeline(self):
        summary = DataProfiler.profile_matrix(self.headers, self.rows)
        preprocessor = AutoPreprocessor()
        feats, X, y = preprocessor.fit_transform(
            self.headers, self.rows, target_col="is_promoted", profiles=summary.columns
        )

        self.assertNotIn("id", feats)
        self.assertIn("age", feats)
        self.assertIn("salary", feats)
        self.assertEqual(len(X), 8)
        self.assertEqual(len(y), 8)

        leaderboard = AutoMLBenchmark.evaluate(X, y, test_ratio=0.25)
        self.assertGreaterEqual(len(leaderboard), 1)
        top_model = leaderboard[0]
        self.assertGreaterEqual(top_model.accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
