# Machine Learning Engineer Nanodegree
## Capstone Project
Jari-Pekka Ryynänen
December 31st, 2050

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

This capstone project is about categorizing audio samples. Categorizing audio samples is a process
of giving an appropriate label to audio. In this project we are using Google AudioSet [1], which
contains huge amount of prelabeled samples drawn from YouTube-videos. More precisly, we are using
balanced subsets of the AudioSet, which contain approximately 22k samples for training and Xk
samples for testing.

As the used dataset is rather big, the resulting model has good chance to generalize well also to
audio from other sources than YouTube. There are several application ideas for this kind of model:

 * categorising audios/videos and making semantic search possible

 * displaying extra information with captions (very helpful for deaf people)

 * triggering some actions based on the audio event (like saving only speech)


In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

### Problem Statement

Problem to solve: Create a system that is capable of giving (possible multiple) label(s) to a short
audio segment. More precisily, given audio sample, predict a label that describes that sample,
where label is from predefined set. The approach will be supervised, i.e. system is trained to do
this prediction by showing simple-label pairs. Precise predictions with coarse labels are preferred
over imprecise with detailed labels.

First task is to download and store samples. The samples are segments of YouTube-videos, but as we
are interested in the audio of those videos, the audio must be extracted from the videos.

After the samples are stored locally, preprocessing can be done. Our preprocess-part will consist of
extraction of features, min-max-normalization, label hot-encoding and padding audio samples to equal
lengths.

Using output from the preprocess, zero-hypothesis, baseline and RNN model are fitted.
Model fit is evaluated against validation set, which is randomly drawn subset of training set. Depending
on the evaluation result, hyperparameters (and possibly features) are tuned and models are re-evaluated
until RNN model does not seem to improve with reasonable amount of work. For this part we might use
smaller subset of the training set to make the iterations faster.

When final model hyperparameters and features are found, full training set is used to train models,
and full testing set is used to get final scores.

In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

### Metrics

Balanced error measure which is easy to understand, and is comparable with
values from [2] is **F1-score**: Harmonic mean of precision and recall.
Mathematical equation for F1 is *2\*precision\*recall/(precision+recall)*.

F1-scores are inspected both class-wise and weighted average over classes.

In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration

Training data set has below class distribution:

IMAGE

As we can see, there is slight imbalance between classes. To compensate this,
we will use weighted cost function, which emphasizes the classes with less
samples.

Samples of the data have variable lengths; however, as seen from above table,
the full lengths samples dominate, and the short values seem to be more like outliers.
However, out RNN model is able to work with variable lengths (even though the actual
tensors provided as input must have fixed lengths, the model will discard the padding).

TABLE


The features seem to also have some values that are outlier according to Tukey's test, but removing them
does not seem doable, as they are spread across different samples. However, this gives indication that
we probably want to use some regularization when training model, as we might not be able to trust
the data completely.

In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques

#### Benchmark models

In this project will be using two simple models which performance will be compared against
our final model:

 * Zero-hypothesis: Model that predicts always the same (first) label.

 * Logistic regression: Linear model that "compresses" its output using logistic function.
   For multilabel classification, multiple models are trained, each giving probability of one class.

These models will use the same features as the final model.

#### Mel-Frequency Cepstral Coefficients

Mel-Frequency Cepstral Coefficients (MFCC) are a way to present spectral information
of audio. They are usually calculated using short overlapping frames of audio.

Normal cepstral coefficients present the change in the spectrum; by picking the first few
cepstral coefficient we are able to capture the envelope of the spectrum. With MFCC, we
are also taking account how human auditory system works by dividing the spectrum using
Mel-scale, which approximates human hearing.

Using MFCCs as features reduces the dimensionality of the problem compared to raw audio.
As they model the data similar manner as human hearing system, they should be able to capture 
relevant information that allows humans (and in this case, hopefully machines) to recognize
the audio samples.

#### Recurrent Neural Network

Recurrent Neural Networks (RNN) are neural networks that use their output from the previous time step
t-1 when calculating output for the timestep t. Also the errors are propagated through timesteps backwards.
The chain rule that is used in all neural networks applies also here.
 
Long Short-Term Memory (LSTM) units are building blocks for RNN that are able to learn over many time steps.
LSTM responds to vanishing/exploding gradient problems which may occur in long sequences by provding the
previous "internal state" of the unit as is to next timestep, and gating that state, input and output by 
"fuzzy" gates (i.e. if traditional logic gates can be seen as multiplication of 0 or 1, these gates multiply
using value which is [0, 1]).

By using RNN, we are able to capture temporal information of audio samples. With LSTM, our model can learn
dependencies which span over whole sequence (which are mostly 199 feature vectors long).

#### Dropout, batch normalization and weight initialization

Dropout is a regularization technique to neural networks. It sets randomly outputs of the 
previous layer to zero with some given probability. This counters directly against overfitting, as
the model discards data randomly and is this way pushed to learn latent variables.

Batch normalization is method which helps relaxing tuning required for weights and learning rate
(it is also reported to have other many other benefits). It tries to normalize its input to have mean of 0
and variance of 1.

For weight initializing orthogonal initializer is used. It has been reported to have
positive effect on controlling vanishing/exploding gradients and in some cases also
making the models to converge faster.

#### Optimizer and learning rate

Adam optimization algorithm, which is used in this project, is an optimizer that adapts the learning rate by itself.
However, the optimizer is provided with maximum learning rate. This maximum value will be decayed, to make the
model stabilize as iterations proceed.

#### Multi-label output and label weights

Since the output for our models can have multiple labels, we will be using sigmoid activation in each output
node independently, and interpret the result as probability of corresponding class being present.
If the probability is over 1/2, our final prediction is that the class is present, and vice versa.

Label weights that are used to counter imbalanced distribution of classes are calcuated using class frequenciesi
in full training set.

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

 * Audio is extraced from videos and stored with sample rate 16 kHz.

 * 13 MFCC features are calculated with windown length 0.1 seconds and overlap of 0.05 seconds.

 * Top-level labels for audio samples are gathered from the ontology tree and k-hot-encoded.

 * Each feature is min-max-normalized. This is done even batch normalization is used at parts of
   the model, as adding the batch normalization inside TensorFlow RNN did not seems easily doable.
 
 * Sequences are zero-padded to have equal lengths. Even though RNN is able to handle variable length
   sequences, the sequences used should have equal shapes and then the results are picked according to
   original lenghts.

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation

#### Getting the data

Downloading the balanced subsets of the AudioSet videos and extracting the audio from them was done using
modified version of download.sh. Modifications to original tool were:

Downloading all the samples tookseveral days even with good quality Internet-connection. However for testing
purposes, download-script can be canceled e.g. after few hundred samples are ready.

#### Preprocessing

For preprocessing, numpy, pandas and library python_speech_features were used. Preprocess step is runnable as
separate Python-script, and creates storage single file with the preprocessed data. This file is fed to
script doing the model fitting and predictions.

#### Baseline models

Zero-hypothesis and logistic regression were implemented using scikit-learn. Logistic regression was wrapped
with one vs. many -ensemble classifier to create classifiers able to do multilabel classification.

#### RNN

RNN implementation was done with TensorFlow. Below is the network graph.


#### Environment

Both personal computer and FloydHub were used.
 
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement

Since the model training time was rather long with the whole training data (depending on the model complexity, from one day to several days)
parameter estimation was done with smaller subset of 1000 randomly drawn samples from training set, which was
furthermore divided to training and validation sets of 800 and 200 samples, respectively.

With this split, number of LSTM cells used was tuned and the affect of batch normalization was tested.
There would be number of other tunable parameters as well, but the scope of this project seemed start to grow too big.

One problem that was encountered, had to do with stability of Adam optimization. The loss seemingly randomly started to increase
a lot after tens of thousands iterations were run. Fortunately, this was known feature of Adam, and the TensorFlow documentation
mentioned epsilon parameter and suggested adjustment for that.


In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
