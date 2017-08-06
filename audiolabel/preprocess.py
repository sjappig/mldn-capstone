import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd

import audiolabel.audio
import audiolabel.ontology
import audiolabel.stats


ONTOLOGY = audiolabel.ontology.read(
    'ontology/ontology.json',
)


_MAIN_CONCEPT_LABELS = sorted(ONTOLOGY.topmost_labels)


def k_hot_encode(labels):
    '''Encode labels. Example output [0, 1, 0, 1, 0, 0, 1].
    '''

    label_indices = [
        _MAIN_CONCEPT_LABELS.index(label)
        for label in labels
    ]

    return [
        int(idx in label_indices)
        for idx in range(0, len(_MAIN_CONCEPT_LABELS))
    ]


def min_max_normalize(features, min_features, max_features):
    '''Normalize *features** between 0 and 1 (feature-wise).
    '''

    return np.array([
        (feature - min_features) / (max_features - min_features)
        for feature in features
    ])


def extract_features(data, samplerate):
    '''Calculate MFCC features from audio.
    '''

    import python_speech_features

    return python_speech_features.mfcc(
        data,
        samplerate=samplerate,
        winlen=0.1,
        winstep=0.05,
    )


def pad_samples(dataframe):
    '''Pad samples with zeros, so the
    sample sequences have equal lengths.
    '''

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


def calculate_and_store_features(filepath, max_samples, csv_filepath, audio_directory, norm_stats=None):
    '''Calculate (at most *max_samples*; if None, all samples are used) features from *csv_filepath*
    and *audio_directory*, and store them to *filepath*. If *norm_stats* is given, use that for
    min-max-normalization instead of statistics calculated from the features.
    '''

    hdf_dir = os.path.dirname(filepath)

    if not os.path.isdir(hdf_dir):
        os.makedirs(hdf_dir)

    print 'Creating HDF store, statistics collector and training data generator...'

    store = pd.HDFStore(filepath)

    stats_collector = audiolabel.stats.StatisticsCollector.from_ontology(ONTOLOGY)

    generator = _generate_data(stats_collector, max_samples, csv_filepath, audio_directory)

    print 'Creating and populating dataframe...'
    dataframe = pd.DataFrame(generator, columns=('samples', 'nonpadded_length', 'labels_ohe'))

    minimum = norm_stats.minimum if norm_stats is not None else stats_collector.minimum
    maximum = norm_stats.maximum if norm_stats is not None else stats_collector.maximum

    dataframe.samples = min_max_normalize(
        dataframe.samples,
        minimum,
        maximum,
    )

    dataframe = pad_samples(dataframe)

    print 'Storing dataframe to HDF store...'

    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    store.put('data', dataframe)

    stats_collector.to_hdf(store)

    store.close()


def _generate_data(stats_collector, max_samples, csv_filepath, audio_directory):
    '''Generate (at most *max_samples*; if None, all samples are used) tuples
    (sequence of features, length of sequence, k-hot-encoded samples).
    '''

    audio_generator = audiolabel.audio.generate_all(
        csv_filepath,
        audio_directory,
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
        'csv_file',
        help='Filepath to CSV file with sample data.',
    )
    parser.add_argument(
        'audio_dir',
        help='Path to directory with wav-files',
    )
    parser.add_argument(
        '--max-samples',
        metavar='N',
        type=int,
        help='Maximum number of samples to process',
    )
    parser.add_argument(
        '--normalize-using',
        metavar='some_hdf_store',
        help='Use statistics from this HDF store when normalizing features.',
    )
    args = parser.parse_args()

    norm_stats = (
        audiolabel.stats.StatisticsCollector.from_hdf(pd.HDFStore(args.normalize_using, mode='r'))
        if args.normalize_using is not None
        else None
    )

    calculate_and_store_features(args.hdf_store, args.max_samples, args.csv_file, args.audio_dir, norm_stats)

