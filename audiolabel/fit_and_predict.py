import argparse
import collections

import sklearn.model_selection
import numpy as np

import audiolabel.baseline
import audiolabel.rnn
import audiolabel.util
import audiolabel.zero_hypothesis


ClassifierType = collections.namedtuple('ClassifierType', ('name', 'create'))


def read_and_split_datasets(filepath, dataset_size=None, validation_size=0.2):
    '''Read data from given filepath. If *dataset_size* is given, use only at most
    that many samples. If *validation_size* is zero, do not create validation dataset.
    '''

    dataset = audiolabel.util.read_dataset(filepath, dataset_size)

    if validation_size > 0:
        x_train, x_validation, y_train, y_validation, lengths_train, lengths_validation = (
            sklearn.model_selection.train_test_split(
                dataset.x,
                dataset.y,
                dataset.nonpadded_lengths,
                test_size=validation_size,
                stratify=dataset.y,
            )
        )

        return (
            audiolabel.util.Dataset('train', x_train, y_train, lengths_train),
            audiolabel.util.Dataset('validation', x_validation, y_validation, lengths_validation),
        )

    else:
        return (
            audiolabel.util.Dataset('train', dataset.x, dataset.y, dataset.nonpadded_lengths),
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
    parser.add_argument(
        '--N',
        help='# of samples to use',
        type=int,
    )
    parser.add_argument(
        '--epochs',
        help='# of epochs to train RNN',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--validation-size',
        type=float,
        help='Size of validation set drawn from the training dataset [0,1]',
        required=True,
    )
    parser.add_argument(
        '--test',
        metavar='test_hdf_store',
        help='Filepath to HDF store with preprocessed test data',
    )
    args = parser.parse_args()

    classifier_types = (
        ClassifierType('zero-hypothesis', audiolabel.zero_hypothesis.create),
        ClassifierType('baseline', audiolabel.baseline.create),
        ClassifierType('RNN', audiolabel.rnn.create),
    )

    if args.skip is not None:
        if not any(args.skip == clf_type.name for clf_type in classifier_types):
            print 'Unkown classifier to skip: {}'.format(args.skip)

    # *datasets* may contain up to three datasets: train, validation and test
    datasets = read_and_split_datasets(
        args.hdf_store,
        dataset_size=args.N,
        validation_size=args.validation_size,
    )

    if args.test is not None:
        test_dataset = audiolabel.util.read_dataset(args.test)
        datasets += (
            audiolabel.util.Dataset('test', test_dataset.x, test_dataset.y, test_dataset.nonpadded_lengths),
        )

    scorers = (
        ('F1', audiolabel.util.f1_score),
        ('Precision', audiolabel.util.precision_score),
    )

    for classifier_type in classifier_types:

        if classifier_type.name == args.skip:
            print 'Skipping {}'.format(args.skip)
            continue

        print 'Creating {} classifier...'.format(classifier_type.name)

        extra_args = {}

        # This is a bit hackish way to detect if we have validation set in use
        if datasets[1].name == 'validation':
            extra_args['x_validation'] = datasets[1].x
            extra_args['y_validation'] = datasets[1].y
            extra_args['validation_lengths'] = datasets[1].nonpadded_lengths

        # First dataset is used as the training set
        classifier = classifier_type.create(
            datasets[0].x,
            datasets[0].y,
            # Below keyword arguments affect only RNN
            train_lengths=datasets[0].nonpadded_lengths,
            num_epochs=args.epochs,
            **extra_args
        )

        for dataset in datasets:
            y_pred = classifier.predict(
                dataset.x,
                nonpadded_lengths=dataset.nonpadded_lengths,
            )

            for name, scorer in scorers:
                score = scorer(dataset.y, y_pred)
                weighted_score = scorer(dataset.y, y_pred, 'weighted')

                print '{}: {} score for {}: {} => {}'.format(
                    classifier_type.name,
                    name,
                    dataset.name,
                    score,
                    weighted_score,
                )

