from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, brier_score_loss
def classification_metrics(y_true, proba, threshold=0.5):
    import numpy as np
    y_pred = (proba >= threshold).astype(int)
    return {
      "auprc": float(average_precision_score(y_true, proba)),
      "auroc": float(roc_auc_score(y_true, proba)),
      "f1": float(f1_score(y_true, y_pred)),
      "brier": float(brier_score_loss(y_true, proba))
    }
