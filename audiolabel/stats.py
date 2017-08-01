import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StatisticsCollector(object):

    def __init__(self, label_distribution, minimum=None, maximum=None, sample_count=0):
        self._label_distribution = label_distribution
        self._maximum_features = maximum
        self._minimum_features = minimum
        self._sample_count = sample_count

    def update(self, features, labels):
        labels = set(labels)
        assert all(label in self._label_distribution for label in labels)

        self._sample_count += 1

        for label in labels:
            self._label_distribution[label] += 1

        sample_max = features.max(axis=0)
        sample_min = features.min(axis=0)

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
        store.put('label_distribution', pd.Series(self.label_distribution))

    @classmethod
    def from_hdf(cls, store):
        maximum = np.array(store.get('max')[0].tolist())
        minimum = np.array(store.get('min')[0].tolist())
        label_distribution = store.get('label_distribution').to_dict()

        return cls(label_distribution, maximum=maximum, minimum=minimum)

    @classmethod
    def from_ontology(cls, ontology):
        label_distribution = {
            label: 0
            for label in ontology.topmost_labels
        }
        return cls(label_distribution)


_MAX_SAMPLE_LEN = 160000


def collect_samples(sample, sum_sample, min_sample, max_sample):

    if sum_sample is None:
        sum_sample = np.zeros(_MAX_SAMPLE_LEN)

    if min_sample is None:
        min_sample = np.array([99999]*_MAX_SAMPLE_LEN)

    if max_sample is None:
        max_sample = np.array([None]*_MAX_SAMPLE_LEN)

    sample_len = len(sample)

    sum_sample[0:sample_len] += sample
    min_sample[0:sample_len] = np.minimum(min_sample[0:sample_len], sample)
    max_sample[0:sample_len] = np.maximum(max_sample[0:sample_len], sample)

    return sum_sample, min_sample, max_sample


def plot(collected, sample):
    fig = plt.figure()
    ax = plt.subplot(111)
    x = np.array(range(0,160000)) * 1.0/16000

    ax.plot(x, sample, 'b', label='Example')
    ax.plot(x, collected[2], 'y', label='Max')
    ax.plot(x, collected[0] / 21778, 'r', label='Mean')
    ax.plot(x, collected[1], 'g', label='Min')

    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    ax.autoscale(enable=True, axis='x', tight=True)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
