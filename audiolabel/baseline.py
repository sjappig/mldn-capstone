import argparse

import pandas as pd
import sklearn.metrics
import sklearn.model_selection

import audiolabel.preprocess


_ZERO_HYPOTHESIS_PREDICTION = [audiolabel.preprocess.k_hot_encode(['/m/04rlf'])]


def zero_hypothesis_scores(*y_args):
    def calculate():
        for y in y_args:
            yield sklearn.metrics.f1_score(
                y.values,
                _ZERO_HYPOTHESIS_PREDICTION*len(y),
                average='macro',
            )

    return tuple(calculate())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MFCCs and stats for audioset samples')
    parser.add_argument(
        'hdf_store',
        help='Filepath to HDF store. If store already exists, it will be overwritten.',
    )
    args = parser.parse_args()

    data = pd.read_hdf(args.hdf_store, 'data')
    count = pd.read_hdf(args.hdf_store, 'count')
    dist = pd.read_hdf(args.hdf_store, 'label_distribution')

    print '{} samples, labels distribution: {}'.format(count, dist)

    print 'Splitting training data to train and validation sets...'


    train_data, validation_data = sklearn.model_selection.train_test_split(
        data,
        test_size=0.2,
    )

    print train_data['labels_ohe'][0:9]
    print _ZERO_HYPOTHESIS_PREDICTION*10
    print 'Calculating F1-scores for zero-hypothesis...'

    train_score, validation_score = zero_hypothesis_scores(train_data['labels_ohe'], validation_data['labels_ohe'])

    print 'Zero-hypothesis: {} (training set), {} (validation set)'.format(train_score, validation_score)





