import csv
import os.path

import scipy.io.wavfile as wavfile


def generate_all(csv_filename, wav_dir, missing_file_action=None):
    '''Generate tuples (labels, data, samplerate) for all samples
    from *csv_filename* that exist in *wav_dir*.
    '''

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

            samplerate, data = wavfile.read(filepath)

            yield (
                labels,
                data,
                samplerate,
            )
