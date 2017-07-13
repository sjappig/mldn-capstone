import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import python_speech_features as psf

import audiolabel.audio
import audiolabel.ontology
import audiolabel.stats


ONTOLOGY = audiolabel.ontology.read(
    'dataset/ontology/ontology.json',
)


_MAIN_CONCEPT_LABELS = sorted(ONTOLOGY.topmost_labels)


def k_hot_encode(labels):
    label_indices = [
        _MAIN_CONCEPT_LABELS.index(label)
        for label in labels
    ]

    return [
        int(idx in label_indices)
        for idx in range(0, len(_MAIN_CONCEPT_LABELS))
    ]


def min_max_normalize(features, min_features, max_features):
    return np.array([
        (feature - min_features) / (max_features - min_features)
        for feature in features
    ])


def extract_features(data, samplerate):
    return psf.mfcc(
        data,
        samplerate=samplerate,
        winlen=0.1,
        winstep=0.05,
    )


def pad_samples(dataframe):
    max_length = np.max(dataframe.nonpadded_length)

    def pad_with_zeros(df_element):
        pad_size = (
            max_length - df_element.nonpadded_length,
            df_element.samples.shape[1],
        )
        df_element.samples = np.concatenate((
            df_element.samples,
            np.zeros(pad_size),
        ))
        return df_element

    padded_samples = dataframe[dataframe.nonpadded_length != max_length].apply(pad_with_zeros, axis=1)

    dataframe.update(padded_samples)

    return dataframe


def calculate_and_store_features(filepath, max_samples=None):
    hdf_dir = os.path.dirname(filepath)

    if not os.path.isdir(hdf_dir):
        os.makedirs(hdf_dir)

    print 'Creating HDF store, statistics collector and training data generator...'

    store = pd.HDFStore(filepath)

    stats_collector = audiolabel.stats.StatisticsCollector(ONTOLOGY)

    generator = _generate_data(stats_collector, max_samples)

    print 'Creating and populating dataframe...'
    dataframe = pd.DataFrame(generator, columns=('samples', 'nonpadded_length', 'labels_ohe'))

    dataframe.samples = min_max_normalize(
        dataframe.samples,
        stats_collector.minimum,
        stats_collector.maximum,
    )

    dataframe = pad_samples(dataframe)

#    import pdb; pdb.set_trace()

    print 'Storing dataframe to HDF store...'

    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    store.put('data', dataframe)

    stats_collector.to_hdf(store)

    store.close()


def _generate_data(stats_collector, max_samples=None):
    audio_generator = audiolabel.audio.generate_all(
        'dataset/audioset/balanced_train_segments.csv',
        'dataset/audioset/train',
    )

    for labels, data, samplerate in audio_generator:
        if max_samples is not None and stats_collector.count >= max_samples:
            return

        if stats_collector.count  % 1000 == 0:
            print 'Sample #{}'.format(stats_collector.count)


        sample = extract_features(data, samplerate)
        topmost_labels = ONTOLOGY.get_topmost_labels(labels)

        stats_collector.update(sample, topmost_labels)

        yield sample, len(sample), k_hot_encode(topmost_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate MFCCs and stats for audioset samples')
    parser.add_argument(
        'hdf_store',
        help='Filepath to HDF store. If store already exists, it will be overwritten.',
    )
    parser.add_argument(
        '--max-samples',
        metavar='N',
        type=int,
        help='Maximum number of samples to process',
    )
    args = parser.parse_args()
    calculate_and_store_features(args.hdf_store, args.max_samples)

