"""
DataProbe-AutoML: Automated Exploratory Data Analysis & Baseline AutoML Pipeline.
"""

__version__ = "1.0.0"
__author__ = "Sais-gallego"

from .profiler import DataProfiler, ColumnProfile, DatasetSummary
from .preprocessor import AutoPreprocessor
from .benchmark import AutoMLBenchmark, ModelEvaluationResult

__all__ = [
    "DataProfiler",
    "ColumnProfile",
    "DatasetSummary",
    "AutoPreprocessor",
    "AutoMLBenchmark",
    "ModelEvaluationResult",
]
