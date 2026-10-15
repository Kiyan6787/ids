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
