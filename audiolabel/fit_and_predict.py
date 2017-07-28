import argparse
import collections

import sklearn.model_selection

import audiolabel.baseline
import audiolabel.rnn
import audiolabel.util
import audiolabel.zero_hypothesis


ClassifierType = collections.namedtuple('ClassifierType', ('name', 'create'))


def read_and_split_datasets(filepath, dataset_size, test_size=0.2):
    dataset = audiolabel.util.read_dataset(filepath, dataset_size)

    x_train, x_validation, y_train, y_validation, lengths_train, lengths_validation = (
        sklearn.model_selection.train_test_split(
            dataset.x,
            dataset.y,
            dataset.nonpadded_lengths,
            test_size=test_size,
            stratify=dataset.y,
        )
    )

    return (
        audiolabel.util.Dataset('train', x_train, y_train, lengths_train),
        audiolabel.util.Dataset('validation', x_validation, y_validation, lengths_validation),
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
    args = parser.parse_args()

    classifier_types = (
        ClassifierType('zero-hypothesis', audiolabel.zero_hypothesis.create),
        ClassifierType('baseline', audiolabel.baseline.create),
    )

    if args.skip is not None:
        if not any(args.skip == clf_type.name for clf_type in classifier_types):
            print 'Unkown classifier to skip: {}'.format(args.skip)

    datasets = read_and_split_datasets(args.hdf_store, dataset_size=args.N)

    scorers = (
        ('F1', audiolabel.util.f1_score),
        ('Precision', audiolabel.util.precision_score),
    )

    for classifier_type in classifier_types:

        if classifier_type.name == args.skip:
            print 'Skipping {}'.format(args.skip)
            continue

        print 'Creating {} classifier...'.format(classifier_type.name)

        # First dataset is used as the training set
        classifier = classifier_type.create(datasets[0].x, datasets[0].y)

        for dataset in datasets:
            y_pred = classifier.predict(dataset.x)

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

    x = datasets[0].x
    y = datasets[0].y
    lengths = datasets[0].nonpadded_lengths

    audiolabel.rnn.train_graph(x, y, lengths, datasets[1].x, datasets[1].y, datasets[1].nonpadded_lengths)
