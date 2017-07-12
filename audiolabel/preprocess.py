import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd

import audiolabel.sample
import audiolabel.ontology
import audiolabel.stats


ONTOLOGY = audiolabel.ontology.read(
    'dataset/ontology/ontology.json',
)


_MAIN_CONCEPT_LABELS = sorted([
    concept['id']
    for concept in ONTOLOGY.main_concepts
])


def k_hot_encode(labels):
    label_indices = [
        _MAIN_CONCEPT_LABELS.index(label)
        for label in labels
    ]

    return [
        int(idx in label_indices)
        for idx in range(0, len(_MAIN_CONCEPT_LABELS))
    ]


def calculate_and_store_features(filepath, max_samples=None):
    hdf_dir = os.path.dirname(filepath)

    if not os.path.isdir(hdf_dir):
        os.makedirs(hdf_dir)

    print 'Creating HDF store, statistics collector and training data generator...'

    store = pd.HDFStore(filepath)

    stats_collector = audiolabel.stats.StatisticsCollector(ONTOLOGY)

    generator = _generate_training_data(stats_collector, max_samples)

    print 'Creating and populating dataframe...'
    dataframe = pd.DataFrame(generator, columns=('features', 'labels', 'labels_ohe'))

    print 'Storing dataframe to HDF store...'

    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    store.put('data', dataframe)

    stats_collector.to_hdf(store)

    store.close()


def _generate_training_data(stats_collector, max_samples=None):
    audio_generator = audiolabel.sample.generate_all_audios(
        'dataset/audioset/balanced_train_segments.csv',
        'dataset/audioset/train',
    )

    for sample in audio_generator:
        pp_sample = audiolabel.sample.PreprocessedSample(sample, ONTOLOGY)

        if max_samples is not None and stats_collector.count >= max_samples:
            return

        if stats_collector.count  % 1000 == 0:
            print 'Sample #{}'.format(stats_collector.count)

        stats_collector.update(pp_sample)

        labels = pp_sample.labels

        yield pp_sample.data, labels, k_hot_encode(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MFCCs and stats for audioset samples')
    parser.add_argument(
        'hdf_store',
        help='Filepath to HDF store. If store already exists, it will be overwritten.',
    )
    parser.add_argument(
        '--max_samples',
        metavar='N',
        type=int,
        help='Maximum number of samples to process',
    )
    args = parser.parse_args()
    calculate_and_store_features(args.hdf_store, args.max_samples)

