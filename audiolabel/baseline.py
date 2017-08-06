import numpy as np

import sklearn.linear_model
import sklearn.multiclass
import sklearn.preprocessing
import sklearn.pipeline

import audiolabel.util


def create(x_train, y_train, **_):
    '''Create logistic regression multi-label classifier.
    '''

    def flatten(samples):
        return np.array([
            sample.flatten('F')
            for sample in samples
        ])

    transformer = sklearn.preprocessing.FunctionTransformer(
        flatten,
        validate=False,
    )

    classifier = sklearn.multiclass.OneVsRestClassifier(
        sklearn.linear_model.LogisticRegression(),
        n_jobs=-1,
    )

    pipeline = sklearn.pipeline.Pipeline([
        ('transformer', transformer),
        ('classifier', classifier),
    ])

    predictor = pipeline.fit(x_train, y_train)

    return audiolabel.util.Predictor(predictor.predict)
