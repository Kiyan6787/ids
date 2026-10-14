import numpy as np


class MetricsCalculator:

    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have identical shape")
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    @staticmethod
    def compute_metrics(cm: dict) -> dict:
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = ((tp + tn) / (tp + fp + tn + fn)
                    if (tp + fp + tn + fn) > 0 else 0.0)
        return {"precision": precision, "recall": recall, "f1": f1,
                "fpr": fpr, "accuracy": accuracy}

    @staticmethod
    def _sort_by_score(y_true: np.ndarray, scores: np.ndarray) -> tuple[list, list]:
        """Return labels and scores as plain lists, highest score first."""
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        order = np.argsort(scores)[::-1]
        return y_true[order].tolist(), scores[order].tolist()

    @staticmethod
    def compute_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> dict:
        labels, sorted_scores = MetricsCalculator._sort_by_score(y_true, scores)
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos

        tp, fp = 0, 0
        fpr, tpr = [0.0], [0.0] 
        for i in range(len(labels)):
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
            is_last = i == len(labels) - 1
            if is_last or sorted_scores[i + 1] != sorted_scores[i]:
                fpr.append(fp / n_neg if n_neg > 0 else 0.0)
                tpr.append(tp / n_pos if n_pos > 0 else 0.0)
        return {"fpr": fpr, "tpr": tpr}

    @staticmethod
    def compute_pr_curve(y_true: np.ndarray, scores: np.ndarray) -> dict:
        labels, sorted_scores = MetricsCalculator._sort_by_score(y_true, scores)
        n_pos = sum(labels)

        tp, fp = 0, 0
        precision, recall = [1.0], [0.0]
        for i in range(len(labels)):
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
            is_last = i == len(labels) - 1
            if is_last or sorted_scores[i + 1] != sorted_scores[i]:
                precision.append(tp / (tp + fp))
                recall.append(tp / n_pos if n_pos > 0 else 0.0)
        return {"precision": precision, "recall": recall}

    @staticmethod
    def compute_auc(x: list, y: list) -> float:
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("compute_auc needs two equal-length arrays, n >= 2")
        area = 0.0
        for i in range(1, len(x)):
            width = x[i] - x[i - 1]
            average_height = (y[i] + y[i - 1]) / 2
            area += width * average_height
        return float(area)
