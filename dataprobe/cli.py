import argparse
import csv
import json
import sys
from .benchmark import AutoMLBenchmark
from .preprocessor import AutoPreprocessor
from .profiler import ColumnType, DataProfiler


def print_eda_summary(summary):
    print("=" * 65)
    print("  DataProbe-AutoML: Automated Dataset Profile")
    print("=" * 65)
    print(f" Rows : {summary.row_count} | Columns : {summary.column_count}")
    print(f" Total Cells : {summary.total_cells} | Missing Cells : {summary.total_missing_cells} ({summary.missing_cell_ratio * 100:.2f}%)")
    print(f" Duplicate Rows : {summary.duplicate_rows}")
    print("-" * 65)
    print(f"{'Column Name':<20} {'Type':<14} {'Missing %':<10} {'Unique':<8} {'Key Stat'}")
    print("-" * 65)
    for col in summary.columns:
        stat_preview = ""
        if col.inferred_type == ColumnType.NUMERICAL:
            stat_preview = f"mean={col.stats.get('mean')}, iqr={col.stats.get('iqr')}"
        elif col.inferred_type == ColumnType.CATEGORICAL:
            top = col.stats.get("top_categories", [])
            stat_preview = f"top={top[0]['value']}" if top else ""
        print(f"{col.name:<20} {col.inferred_type.value:<14} {col.missing_ratio * 100:>6.1f}%    {col.distinct_count:<8} {stat_preview}")
    print("=" * 65)


def print_leaderboard(results):
    print("\n" + "=" * 65)
    print("  AutoML Baseline Benchmark: Model Leaderboard")
    print("=" * 65)
    print(f"{'Rank':<5} {'Model Name':<26} {'Accuracy':<10} {'F1 Score':<10} {'Latency'}")
    print("-" * 65)
    for i, res in enumerate(results):
        print(f"#{i+1:<4} {res.model_name:<26} {res.accuracy * 100:>6.2f}%   {res.f1_score * 100:>6.2f}%   {res.train_time_ms:>6.2f} ms")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="DataProbe-AutoML CLI")
    parser.add_argument("csv_file", help="Path to input CSV dataset")
    parser.add_argument("--target", "-t", help="Target column for AutoML modeling")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    # 1. Profile CSV
    try:
        summary = DataProfiler.profile_csv(args.csv_file)
    except Exception as e:
        print(f"Error profiling CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.target:
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print_eda_summary(summary)
        sys.exit(0)

    # 2. Train AutoML
    with open(args.csv_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        rows = [r for r in reader if r]

    if args.target not in headers:
        print(f"Error: Target column '{args.target}' not found in headers {headers}", file=sys.stderr)
        sys.exit(1)

    preprocessor = AutoPreprocessor()
    selected_features, X, y = preprocessor.fit_transform(headers, rows, args.target, summary.columns)
    results = AutoMLBenchmark.evaluate(X, y)

    if args.json:
        payload = {
            "dataset_summary": summary.to_dict(),
            "selected_features": selected_features,
            "leaderboard": [r.to_dict() for r in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print_eda_summary(summary)
        print(f"\nTarget Variable : '{args.target}'")
        print(f"Selected Model Features ({len(selected_features)}) : {', '.join(selected_features)}")
        print_leaderboard(results)


if __name__ == "__main__":
    main()
