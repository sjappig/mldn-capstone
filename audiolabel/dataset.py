import numpy as np
import pandas as pd
import sklearn.model_selection

import audiolabel.preprocess


def read(filepath):
    data =  {
        key: pd.read_hdf(filepath, key)
        for key in ('samples', 'min', 'max')
    }
    return Dataset(**data)


class Dataset(object):
    def __init__(self, **data):
        self._min_features = np.array(data['min']).T
        self._max_features = np.array(data['max']).T

        self._features = audiolabel.preprocess.min_max_normalize(
            np.array(data['samples']['features'].tolist()),
            self._min_features,
            self._max_features,
        )

        self._labels = np.array(data['samples']['labels_ohe'].tolist())

    def train_test_split(self, test_size=0.2):
        return sklearn.model_selection.train_test_split(
            self._features,
            self._labels,
            test_size=test_size,
        )

    @property
    def min_features(self):
        return self._min_features

    @property
    def max_features(self):
        return self._max_features
