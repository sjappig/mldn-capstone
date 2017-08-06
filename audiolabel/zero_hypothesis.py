import numpy as np

import sklearn.dummy
import sklearn.pipeline
import sklearn.preprocessing

import audiolabel.preprocess
import audiolabel.util

def create(x_train, y_train, **_):
    '''Create zero-hypothesis predictor.
    '''

    # Even DummyClassifier does not use the incoming data,
    # sklearn will explode if the data provided to classifier
    # is not flat.
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

    predictor = pipeline.fit(x_train, y_train)

    return audiolabel.util.Predictor(predictor.predict)
