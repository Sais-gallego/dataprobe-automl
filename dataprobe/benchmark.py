import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelEvaluationResult:
    model_name: str
    task_type: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    train_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "train_time_ms": round(self.train_time_ms, 2),
        }


class SimpleLogisticModel:
    """Lightweight pure-python baseline logistic regression with SGD."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 40):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0

    def _sigmoid(self, z: float) -> float:
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X: List[List[float]], y: List[float]):
        if not X:
            return
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n_samples = len(X)

        for _ in range(self.epochs):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                linear = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                y_hat = self._sigmoid(linear)
                err = y_hat - yi

                for j in range(n_features):
                    self.weights[j] -= self.lr * err * xi[j]
                self.bias -= self.lr * err

    def predict(self, X: List[List[float]]) -> List[int]:
        preds = []
        for xi in X:
            linear = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
            prob = self._sigmoid(linear)
            preds.append(1 if prob >= 0.5 else 0)
        return preds


class SimpleDecisionStump:
    """Fast baseline single-feature decision stump."""

    def __init__(self):
        self.best_feat = 0
        self.threshold = 0.0
        self.polarity = 1

    def fit(self, X: List[List[float]], y: List[float]):
        if not X:
            return
        n_features = len(X[0])
        best_acc = -1.0

        for f in range(n_features):
            vals = [row[f] for row in X]
            thresh = sum(vals) / len(vals)
            for pol in [1, -1]:
                correct = 0
                for xi, yi in zip(X, y):
                    pred = 1 if (xi[f] >= thresh if pol == 1 else xi[f] < thresh) else 0
                    if pred == int(yi):
                        correct += 1
                acc = correct / len(y)
                if acc > best_acc:
                    best_acc = acc
                    self.best_feat = f
                    self.threshold = thresh
                    self.polarity = pol

    def predict(self, X: List[List[float]]) -> List[int]:
        preds = []
        for xi in X:
            val = xi[self.best_feat]
            pred = 1 if (val >= self.threshold if self.polarity == 1 else val < self.threshold) else 0
            preds.append(pred)
        return preds


class AutoMLBenchmark:
    """Automated baseline model benchmark and leaderboard generator."""

    @staticmethod
    def evaluate(
        X: List[List[float]],
        y: List[float],
        test_ratio: float = 0.25,
        seed: int = 42,
    ) -> List[ModelEvaluationResult]:
        if not X or len(X) < 4:
            return []

        # Train / test split
        random.seed(seed)
        combined = list(zip(X, y))
        random.shuffle(combined)
        split_idx = int(len(combined) * (1.0 - test_ratio))
        train_data = combined[:split_idx]
        test_data = combined[split_idx:]

        X_train, y_train = [c[0] for c in train_data], [c[1] for c in train_data]
        X_test, y_test = [c[0] for c in test_data], [c[1] for c in test_data]

        models = [
            ("SGD-LogisticRegression", SimpleLogisticModel(learning_rate=0.08, epochs=50)),
            ("OptimalDecisionStump", SimpleDecisionStump()),
        ]

        leaderboard: List[ModelEvaluationResult] = []

        for name, model in models:
            t0 = time.time()
            model.fit(X_train, y_train)
            train_ms = (time.time() - t0) * 1000.0

            preds = model.predict(X_test)
            metrics = AutoMLBenchmark._calc_classification_metrics(y_test, preds)

            leaderboard.append(
                ModelEvaluationResult(
                    model_name=name,
                    task_type="binary_classification",
                    accuracy=metrics["accuracy"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1_score=metrics["f1"],
                    train_time_ms=train_ms,
                )
            )

        leaderboard.sort(key=lambda r: r.f1_score, reverse=True)
        return leaderboard

    @staticmethod
    def _calc_classification_metrics(y_true: List[float], y_pred: List[int]) -> Dict[str, float]:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if int(yt) == 1 and yp == 0)

        total = len(y_true)
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
