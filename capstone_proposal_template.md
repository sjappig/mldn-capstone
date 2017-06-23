# Machine Learning Engineer Nanodegree
## Capstone Proposal
Jari-Pekka Ryynänen
June 23, 2017

## Proposal

### Domain Background

Audio event classification is a task of giving appropriate label to audio.
Having automatic system for labeling has several easy-to-understand
applications, e.g:

 * categorising audios/videos and making semantic search possible

 * displaying extra information with captions (very helpful for deaf people)

 * triggering some actions based on the audio event (like saving only speech)

There are several papers tackling this problem (see [1],[2],[3]). As the amount
of data available has grown, the deep learning methods have become more
suitable also for this problem.

### Problem Statement

Problem to solve: Create a system that is capable of giving label to a short
audio segment (possibly from a video). More precisily, given audio sample,
predict a label that describes that sample, where label is from predefined set.
The approach will be supervised, i.e. system is trained to do this prediction
by showing sample-label pairs.

### Datasets and Inputs

**AudioSet**: Large-scale collection (over 2M samples) of human-labeled 10-second
sound clips drawn from YouTube videos [4]. This dataset is perhaps the largest
freely available one, and has very good documentation and support. There is
also starter code which is helpful when starting to work with the dataset [5].

AudioSet includes precalculated features and balanced subsets for training and
evaluation.

Dataset is described in more detail in [4] and [6].

### Solution Statement

To ease the computational burden, precalculated features provided with AudioSet
are used (128-dimensional audio features extracted at 1 Hz). Also instead of
using the whole dataset, which is huge (over 2M samples), balanced training and
balanced evaluation sets are used, both containing more than 20k samples.
Target labels will be the top-most labels in AudioSet ontology ("Human Sounds",
"Source-ambiguous sounds", "Animal" etc).

Using those as inputs, recurrent neural network is trained to do the
classification. Recurrent neural networks are suitable for working with audio,
as they contain memory about past samples. Evaluation set is kept aside until
the classifier is in its final form (before that, model generalization is
approximated with splits done within the training set).

### Benchmark Model

In [6] benchmark model is reported with "balanced mean Average Precision across
the 485 categories of 0.314". However, this result is not directly comparable
with our target, as we will use more coarse labels.

In [2], there are results of audio scene classification expressed with
F1-score. The audio scene classification has subtle differences to our task at
hand, but eventually it is about classifying audio segments, so it makes sense
to compare our results with it. The dataset used in [2] is LITIS-Rouen, which
has 19 categories.

In this work, simple logistic regression model will be trained without
parameter tuning to give baseline. Also zero-hypothesis (model that
predicts first category always) is used to gain understanding how well
our model is performing.

### Evaluation Metrics

Balanced error measure which is easy to understand, and is comparable with
values from [2] is **F1-score**: Harmonic mean of precision and recall.
Mathematical equation for F1 is *2*precision*recall/(precision+recall)*.

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?


[1] Kons, Z., Toledo-Ronen, O.: Audio Event Classification Using Deep Neural Networks, 2013 https://pdfs.semanticscholar.org/1881/acfe27135da6c717fb770dfa3793b8b225d5.pdf
[2] Phan, H. et al.: Audio Scene Classification with Deep Recurrent Neural Networks, 2017 https://arxiv.org/pdf/1703.04770.pdf
[3] Bae, H. et al: Acoustic Scene Classification Using Parallel Combination of LSTM and CNN, 2016 https://www.cs.tut.fi/sgn/arg/dcase2016/documents/challenge_technical_reports/Task1/Bae_2016_task1.pdf
[4] AudioSet: A large-scale dataset of manually annotated audio events https://research.google.com/audioset/dataset/index.html
[5] YouTube-8M Tensorflow Starter Code: https://github.com/google/youtube-8m
[6] Gemmeke, J. et al: Audio Set: An ontology and human-labeled dataset for audio events, 2017 https://research.google.com/pubs/pub45857.html
