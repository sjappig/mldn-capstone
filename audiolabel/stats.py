import numpy as np
import pandas as pd


class StatisticsCollector(object):

    def __init__(self, ontology):
        self._sample_count = 0
        self._label_distribution = {
            concept['id']: 0
            for concept in ontology.main_concepts
        }
        self._maximum_features = None
        self._minimum_features = None

    def update(self, sample):
        assert all(label in self._label_distribution for label in sample.labels)

        self._sample_count += 1

        for label in sample.labels:
            self._label_distribution[label] += 1

        sample_max = sample.data.max(axis=0)
        sample_min = sample.data.min(axis=0)

        if self._maximum_features is None:
            self._maximum_features = sample_max 
            self._minimum_features = sample_min
            return

        self._maximum_features = np.maximum(
            sample_max,
            self._maximum_features,
        )

        self._minimum_features = np.minimum(
            sample_min,
            self._minimum_features,
        )

    @property
    def count(self):
        return self._sample_count

    @property
    def label_distribution(self):
        return self._label_distribution

    @property
    def maximum(self):
        return self._maximum_features

    @property
    def minimum(self):
        return self._minimum_features

    def to_hdf(self, store):
        store.put('max', pd.DataFrame(self.maximum))
        store.put('min', pd.DataFrame(self.minimum))
        store.put('count', pd.Series(self.count))
        store.put('label_distribution', pd.Series(self.label_distribution))
