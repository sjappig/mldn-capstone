Capstone Project for Machine Learning Nanodegree
------------------------------------------------

### Acquiring AudioSet samples

When in directory `dataset/audioset/`:

1. Check following dependencies: `bash`, `ffmpeg` and `youtube-dl`
1. Training dataset: `mkdir train_data; cd train_data; cat ../balanced_train_segments.csv | ../download/download.sh`
1. Test dataset: `mkdir test_data; cd test_data; cat ../eval_segments.csv | ../download/download.sh`

Downloading can be canceled by pressing `Ctrl+C` (few times if one does not seem to work). When started the next time, it will skip the already downloaded files
(you might need to remove .part-files manually). Note that the below scripts won't work if there is very small number of samples (there seems to be some bug in
pandas), so download at least ~100 samples for both training and testing before proceeding.

### Preprocessing the samples and training models

When in repository root:

1. Create virtual environment: `virtualenv --python=python2.7 sandbox`
1. Activate the environment: `source sandbox/bin/activate`
1. Install required packages: `pip install -r requirements.txt`
1. Preprocess training data: `python -m audiolabel.preprocess pp/train.h5 dataset/audioset/balanced_train_segments.csv dataset/audioset/train_data`
1. Preprocess test data: `python -m audiolabel.preprocess --normalize-using pp/train.h5 pp/test.h5 dataset/audioset/eval_segments.csv dataset/audioset/test_data`
1. Train and test model with small number of samples and small epoch count: `python -m audiolabel.fit_and_predict  pp/train.h5 --N 100 --epochs 100 --validation-size 0 --test pp/test.h5` (this might take several minutes)

Check available command line parameters for both scripts with `--help`.

