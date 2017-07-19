import sklearn.metrics


def batches(*datasets, **kwargs):
    batch_size = kwargs.pop('batch_size', 256)

    assert datasets
    assert not kwargs

    length = len(datasets[0])
    assert all(len(dataset) == length for dataset in datasets)

    for begin_idx in range(0, length, batch_size):

        end_idx = min(begin_idx + batch_size, length)

        yield tuple(
            dataset[begin_idx:end_idx]
            for dataset in datasets
        )


def f1_score(y_true, y_pred, average=None):
    return sklearn.metrics.f1_score(
        y_true,
        y_pred,
        average=average,
    )

