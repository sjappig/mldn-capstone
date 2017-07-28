import collections

import numpy as np
import pandas as pd

import sklearn.metrics


Dataset = collections.namedtuple('Dataset', ('name', 'x', 'y', 'nonpadded_lengths'))


def f1_score(y_true, y_pred, average=None):
    return sklearn.metrics.f1_score(
        y_true,
        y_pred,
        average=average,
    )


def precision_score(y_true, y_pred, average=None):
    return sklearn.metrics.precision_score(
        y_true,
        y_pred,
        average=average,
    )

def read_data(filepath):
    return pd.read_hdf(filepath, 'data')


def read_dataset(filepath, dataset_size):
    data = read_data(filepath)

    return dataframe_to_dataset(data, dataset_size)


def dataframe_to_dataset(data, dataset_size=None):
    samples = np.array(data['samples'].tolist())

    labels = np.array(data['labels_ohe'].tolist())

    nonpadded_lengths = np.array(data['nonpadded_length'].tolist())

    if dataset_size is not None:
        dataset_size = min(len(samples), dataset_size)

    else:
        dataset_size = len(samples)

    print 'Using {} samples'.format(dataset_size)

    shuffled_indices = np.random.permutation(dataset_size)

    samples = samples[shuffled_indices]
    labels = labels[shuffled_indices]
    nonpadded_lengths = nonpadded_lengths[shuffled_indices]

    return Dataset('full', samples, labels, nonpadded_lengths)

def batches(*datasets, **kwargs):
    batch_size = kwargs.pop('batch_size', 1024)

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

