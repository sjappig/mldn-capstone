import csv
import itertools
import os.path

import python_speech_features as psf
import scipy.io.wavfile as wavfile


def generate_all_audios(csv_filename, wav_dir, missing_file_action=None):

    if missing_file_action is None:
        missing_file_action = lambda *_: None

    with open(csv_filename, 'rt') as file_obj:

        reader = csv.DictReader(
            file_obj,
            fieldnames=('id', 'start', 'end'),
            restkey='labels',
        )

        for line in reader:

            line['id'] = line['id'].strip()

            if line['id'].startswith('#'):
                continue

            line['start'] = line['start'].strip()

            filename = 'sample_{}_{}.wav'.format(line['id'], line['start'])
            labels = [
                label.strip().strip('"')
                for label in line['labels']
            ]

            filepath = os.path.join(wav_dir, filename)

            if not os.path.isfile(filepath):

                missing_file_action(filepath)
                
                continue

            yield AudioSample(
                labels,
                filepath,
            )   


class Sample(object):

    def __init__(self, labels, data=None):
        self._labels = labels
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels


class AudioSample(Sample):

    def __init__(self, labels, filepath):
        samplerate, data = wavfile.read(filepath)
        
        super(AudioSample, self).__init__(labels, data)

        self.samplerate = samplerate


class PreprocessedSample(Sample):

    def __init__(self, audio_sample, ontology):
        data = psf.mfcc(
            audio_sample.data,
            samplerate=audio_sample.samplerate,
            winlen=0.1,
            winstep=0.05,
        )

        main_concepts = itertools.chain.from_iterable(
            ontology.get_main_concepts(label)
            for label in audio_sample.labels
        )
 
        labels = [
            concept['id']
            for concept in main_concepts
        ]
         
        super(PreprocessedSample, self).__init__(
            labels,
            data,
        )

    def split(self):
        return [
            Sample([label], self.data)
            for label in self.labels
        ]

