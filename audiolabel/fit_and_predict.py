import argparse
import collections

import numpy as np
import pandas as pd
import sklearn.model_selection

import audiolabel.baseline
import audiolabel.util
import audiolabel.zero_hypothesis


ClassifierType = collections.namedtuple('ClassifierType', ('name', 'create'))

Dataset = collections.namedtuple('Dataset', ('name', 'x', 'y'))


def read_and_split_datasets(filepath, test_size=0.2):
    data = pd.read_hdf(filepath, 'data')

    samples = np.array(data['samples'].tolist())

    labels = np.array(data['labels_ohe'].tolist())

    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(
        samples,
        labels,
        test_size=test_size,
    )

    return (
        Dataset('train', x_train, y_train),
        Dataset('validation', x_validation, y_validation),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate F1 scores for audioset samples')
    parser.add_argument(
        'hdf_store',
        help='Filepath to HDF store with preprocessed samples',
    )
    parser.add_argument(
        '--skip',
        help='Classifier to skip'
    )
    args = parser.parse_args()

    classifier_types = (
        ClassifierType('zero-hypothesis', audiolabel.zero_hypothesis.create),
        ClassifierType('baseline', audiolabel.baseline.create),
    )

    if args.skip is not None:
        if not any(args.skip == clf_type.name for clf_type in classifier_types):
            print 'Unkown classifier to skip: {}'.format(args.skip)

    datasets = read_and_split_datasets(args.hdf_store)

    for classifier_type in classifier_types:

        if classifier_type.name == args.skip:
            print 'Skipping {}'.format(args.skip)
            continue

        print 'Creating {} classifier...'.format(classifier_type.name)

        # First dataset is used as the training set
        classifier = classifier_type.create(datasets[0].x, datasets[0].y)

        for dataset in datasets:
            y_pred = classifier.predict(dataset.x)

            score = audiolabel.util.f1_score(dataset.y, y_pred)

            print '{}: F1-score for {}: {}'.format(
                classifier_type.name,
                dataset.name,
                score,
            )
