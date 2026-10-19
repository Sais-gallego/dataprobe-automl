# 📊 DataProbe-AutoML

[![CI Pipeline](https://github.com/Sais-gallego/dataprobe-automl/actions/workflows/ci.yml/badge.svg)](https://github.com/Sais-gallego/dataprobe-automl/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/AI%2FML-AutoML%20Baseline-purple.svg)](https://scikit-learn.org/)

> **High-Performance Automated Exploratory Data Analysis (EDA) & Baseline AutoML Pipeline Generator.**  
> Automatically infers schema types, diagnoses data quality issues (missingness, skewness, outliers, constant/ID features), applies adaptive preprocessing, and trains an evaluation leaderboard of baseline models.

---

## ⚡ Key Capabilities

- **Automatic Type & Semantic Inference**: Accurately classifies columns as `NUMERICAL`, `CATEGORICAL`, `DATETIME`, `CONSTANT`, or `UNIQUE_ID`.
- **Comprehensive Quality Profiling**: Computes mean, standard deviation, interquartile range (IQR), median, and outlier counts using statistical fence rules.
- **Adaptive Preprocessing Pipeline**: Automatically imputes numerical values via median, categorical values via mode, scales features, and drops high-cardinality ID noise.
- **AutoML Baseline Benchmark**: Trains and cross-evaluates multiple baseline machine learning algorithms (SGD Logistic Regression, Optimal Decision Stumps, and scikit-learn models if present).
- **Executive Terminal & JSON Reports**: Generates clean, human-readable terminal summaries or structured JSON artifacts for integration into CI/CD data gates.

---

## 📐 Architecture Overview

```
                      ┌────────────────────────┐
                      │    Input Raw CSV       │
                      └───────────┬────────────┘
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │       DataProbe Profiler      │
                   │  - Schema & Type Inference    │
                   │  - Missing Rates & Outliers   │
                   │  - Statistical Moments & IQR  │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │       Auto Preprocessor       │
                   │  - Median / Mode Imputation   │
                   │  - Constant / ID Elimination  │
                   │  - Standard Feature Scaling   │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌───────────────────────────────┐
                   │    AutoML Model Benchmark     │
                   │  - Train / Validation Split   │
                   │  - Baseline Model Training    │
                   │  - F1, Recall, Latency Ranks  │
                   └──────────────┬────────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
         Interactive Terminal           Executive JSON
              Leaderboard                 Data Artifact
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Sais-gallego/dataprobe-automl.git
cd dataprobe-automl
pip install -r requirements.txt
pip install .
```

### CLI Usage

**1. Profile Dataset (Exploratory Data Analysis):**
```bash
dataprobe dataset.csv
```

**Output:**
```
=================================================================
  DataProbe-AutoML: Automated Dataset Profile
=================================================================
 Rows : 1000 | Columns : 6
 Total Cells : 6000 | Missing Cells : 14 (0.23%)
 Duplicate Rows : 0
-----------------------------------------------------------------
Column Name          Type           Missing %  Unique   Key Stat
-----------------------------------------------------------------
customer_id          UNIQUE_ID          0.0%    1000     
age                  NUMERICAL          0.2%    62       mean=38.4, iqr=14.0
annual_income        NUMERICAL          0.5%    412      mean=68500, iqr=24000
channel              CATEGORICAL        0.0%    4        top=Web
status               CONSTANT           0.0%    1        
churned              NUMERICAL          0.0%    2        mean=0.22, iqr=0.0
=================================================================
```

**2. Run Automated Preprocessing & Baseline Benchmark:**
```bash
dataprobe dataset.csv --target churned
```

**Leaderboard Output:**
```
=================================================================
  AutoML Baseline Benchmark: Model Leaderboard
=================================================================
Rank  Model Name                 Accuracy   F1 Score   Latency
-----------------------------------------------------------------
#1    SGD-LogisticRegression      84.20%     81.50%      1.42 ms
#2    OptimalDecisionStump        78.00%     74.30%      0.35 ms
=================================================================
```

---

## 🐍 Python SDK Integration

```python
from dataprobe import DataProfiler, AutoPreprocessor, AutoMLBenchmark

# Profile CSV
summary = DataProfiler.profile_csv("dataset.csv")

# Clean & Prepare
headers = [c.name for c in summary.columns]
preprocessor = AutoPreprocessor()
features, X, y = preprocessor.fit_transform(headers, raw_rows, target_col="churned", profiles=summary.columns)

# Run AutoML
leaderboard = AutoMLBenchmark.evaluate(X, y)
print(f"Top Model: {leaderboard[0].model_name} (F1: {leaderboard[0].f1_score:.2f})")
```

---

## 🧪 Testing

```bash
python -m unittest discover tests -v
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
