from sklearn.metrics import roc_auc_score, roc_curve

def evaluate_attack(scores, labels):
    """
    Evaluate attack performance.
    :param scores: Membership scores.
    :param labels: True labels (1 for member, 0 for non-member).
    :return: AUC-ROC and TPR@1%FPR.
    """
    auc_roc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_at_1_fpr = tpr[next(i for i, x in enumerate(fpr) if x >= 0.01)]

    return {"AUC-ROC": auc_roc, "TPR@1%FPR": tpr_at_1_fpr}