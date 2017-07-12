import pandas as pd
import sklearn.metrics
import sklearn.model_selection

import audiolabel.preprocess


_ZERO_HYPOTHESIS_PREDICTION = [audiolabel.preprocess.one_hot_encode('/m/04rlf')]


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
    data = pd.read_hdf(audiolabel.preprocess.TRAIN_DATA_HDF, 'data')

    print 'Splitting training data to train and validation sets...'

    train_data, validation_data = sklearn.model_selection.train_test_split(
        data,
        test_size=0.2,
    )

    print train_data['label_ohe'][0:9]
    print _ZERO_HYPOTHESIS_PREDICTION*10
    print 'Calculating F1-scores for zero-hypothesis...'

    train_score, validation_score = zero_hypothesis_scores(train_data['label_ohe'], validation_data['label_ohe'])

    print 'Zero-hypothesis: {} (training set), {} (validation set)'.format(train_score, validation_score)





