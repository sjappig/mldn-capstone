import argparse
import sys

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.multiclass

import audiolabel.dataset
import audiolabel.preprocess
import audiolabel.util


_ZERO_HYPOTHESIS_PREDICTION = [
    audiolabel.preprocess.k_hot_encode(['/m/04rlf'])
]


def zero_hypothesis_scores(*y_args):
    return [
        audiolabel.util.f1_score(y, np.array(_ZERO_HYPOTHESIS_PREDICTION*len(y)))
        for y in y_args
    ]


def baseline_scores(x_train, y_train, x_validation, y_validation):
    def to_mean_features(samples):
        return np.array([
            sample.mean(axis=0)
            for sample in samples
        ])

    estimator = sklearn.linear_model.LogisticRegression()

    multilabel_estimator = sklearn.multiclass.OneVsRestClassifier(
        estimator,
        n_jobs=-1,
    )

    x_train = to_mean_features(x_train)
    x_validation = to_mean_features(x_validation)

    multilabel_estimator.fit(x_train, y_train)

    return [
        audiolabel.util.f1_score(y_train, multilabel_estimator.predict(x_train)),
        audiolabel.util.f1_score(y_validation, multilabel_estimator.predict(x_validation)),
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate baseline F1 scores for audioset samples')
    parser.add_argument(
        'hdf_store',
        help='Filepath to HDF store with preprocessed samples.',
    )
    args = parser.parse_args()

    train = {
        key: pd.read_hdf(args.hdf_store, key)
        for key in ('label_distribution',)
    }

    print 'Labels distribution: {}'.format(train['label_distribution'])

    dataset = audiolabel.dataset.read(args.hdf_store)

    x_train, x_validation, y_train, y_validation = dataset.train_test_split()

    print 'Calculating F1-scores for zero-hypothesis...'

    train_score, validation_score = zero_hypothesis_scores(y_train, y_validation)

    print 'Zero-hypothesis: {} (training set), {} (validation set)'.format(train_score, validation_score)

    print 'Calculating F1-scores for baseline estimator...'

    train_score, validation_score = baseline_scores(x_train, y_train, x_validation, y_validation)

    print 'Baseline estimator: {} (training set), {} (validation set)'.format(train_score, validation_score)
