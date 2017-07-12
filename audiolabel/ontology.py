'''
AudioSet ontology wrapper
'''

import itertools
import json


def read(filepath):
    with open(filepath, 'rt') as file_obj:
        return Ontology(json.load(file_obj))


class Ontology(object):
    def __init__(self, data):
        self._data = {
            concept['id']: concept
            for concept in data
        }

    @property
    def main_concepts(self):
        return [
            concept
            for identifier, concept in self._data.iteritems()
            if identifier not in self._child_ids
        ]

    @property
    def _child_ids(self):
        child_ids = [
            concept['child_ids']
            for concept in self._data.values()
        ]
        return set(itertools.chain(*child_ids))

    def get_concept(self, identifier):
        return self._data[identifier]

    def get_main_concepts(self, identifier):
        if identifier not in self._child_ids:
            return [self.get_concept(identifier)]

        main_ids = [
            concept['id']
            for concept in self._data.values()
            if identifier in concept['child_ids']
        ]

        return itertools.chain.from_iterable(
            self.get_main_concepts(main_id)
            for main_id in main_ids
        )
