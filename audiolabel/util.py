import sklearn.metrics


def f1_score(y_true, y_pred, average=None):
    return sklearn.metrics.f1_score(
        y_true,
        y_pred,
        average=average,
    )

