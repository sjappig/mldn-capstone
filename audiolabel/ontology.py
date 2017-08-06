import itertools
import json


def read(filepath):
    with open(filepath, 'rt') as file_obj:
        return Ontology(json.load(file_obj))


class Ontology(object):
    '''Encapsulate AudioSet ontology.
    '''

    def __init__(self, data):
        self._data = {
            concept['id']: concept
            for concept in data
        }
        all_child_labels = set(itertools.chain.from_iterable(
            concept['child_ids']
            for concept in data
        ))
        self._topmost_labels = set(self._data) - all_child_labels

    @property
    def topmost_labels(self):
        return self._topmost_labels

    def get_topmost_labels(self, labels):
        return list(self._generate_topmost_labels(labels))

    def _generate_topmost_labels(self, labels):
        for label in labels:
            if label in self._topmost_labels:
                yield label

                continue

            parent_labels = [
                concept['id']
                for concept in self._data.values()
                if label in concept['child_ids']
            ]

            for topmost_label in self._generate_topmost_labels(parent_labels):
                yield topmost_label
