import numpy as np

import sklearn.dummy
import sklearn.pipeline
import sklearn.preprocessing

import audiolabel.preprocess


def create(x_train, y_train):

    transform = sklearn.preprocessing.FunctionTransformer(
        lambda X: np.zeros((len(X), 1)),
        validate=False,
    )

    classifier = sklearn.dummy.DummyClassifier(
        'constant',
        constant=audiolabel.preprocess.k_hot_encode(['/m/04rlf']),
    )

    pipeline = sklearn.pipeline.Pipeline([
        ('transform', transform),
        ('classifier', classifier),
    ])

    return pipeline.fit(x_train, y_train)
