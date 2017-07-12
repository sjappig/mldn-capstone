import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd

import audiolabel.sample
import audiolabel.ontology
import audiolabel.stats


PP_DATA_DIR = 'pp_data'
TRAIN_DATA_HDF = os.path.join(PP_DATA_DIR, 'train_data.h5')

ONTOLOGY = audiolabel.ontology.read(
    'dataset/ontology/ontology.json',
)


def _create_label_one_hot_encode():
    main_concept_ids = sorted([
        concept['id']
        for concept in ONTOLOGY.main_concepts
    ])

    identity = np.identity(len(main_concept_ids), dtype=int)

    return {
        main_concept_ids[idx]: identity[idx]
        for idx in range(0, len(main_concept_ids))
    }


_LABEL_ONE_HOT_ENCODE = _create_label_one_hot_encode()


def one_hot_encode(label):
    return _LABEL_ONE_HOT_ENCODE[label]


def calculate_and_store_features(max_samples=None):
    if not os.path.isdir(PP_DATA_DIR):
        os.makedirs(PP_DATA_DIR)

    print 'Creating HDF store, statistics collector and training data generator...'

    store = pd.HDFStore(TRAIN_DATA_HDF)

    stats_collector = audiolabel.stats.StatisticsCollector(ONTOLOGY)

    generator = _generate_training_data(stats_collector, max_samples)

    print 'Creating and populating dataframe...'
    dataframe = pd.DataFrame(generator, columns=('features', 'label', 'label_ohe'))

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

        for splitted_sample in pp_sample.split():

            if max_samples is not None and stats_collector.count >= max_samples:
                return

            if stats_collector.count  % 1000 == 0:
                print 'Sample #{}'.format(stats_collector.count)

            stats_collector.update(splitted_sample)

            label = splitted_sample.labels[0]

            yield splitted_sample.data, label, one_hot_encode(label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MFCC for audioset samples')
    parser.add_argument(
        '--max_samples',
        metavar='N',
        type=int,
        help='Maximum number of samples to process',
    )
    args = parser.parse_args()
    calculate_and_store_features(args.max_samples)

