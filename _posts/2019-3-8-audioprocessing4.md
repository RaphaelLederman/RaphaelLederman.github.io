---
published: true
title: Speech Emotion Recognition - Classification with Time Distributed CNN and LSTM (4)
collection: articles
layout: single
author_profile: false
read_time: true
categories: [articles]
excerpt : "Speech Emotion Recognition"
header :
    overlay_image: "https://raphaellederman.github.io/assets/images/night.jpg"
    teaser_image: "https://raphaellederman.github.io/assets/images/night.jpg"
toc: true
toc_sticky: true
---

For this last short article on speech emotion recognition, we will present a methodology to classify emotions from audio features using Time Distributed CNN and LSTM.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


## Why using deep learning for speech emotion recognition ?

Convolutional Neural Networks (CNNs) show remarkable recognition performance for computer vision tasks as they allow constructing features using only raw data. Characteristics such as local connectivity, parameters sharing and shift/translation invariance play a major role in the efficiency of this type of network. Recurrent Neural Networks (RNNs) show impressive achievement in many sequential data processing tasks : unlike regular neural networks where inputs are assumed to be independent of each other, these architectures progressively accumulate and capture information through the sequences. In particular, Long Short Term Memory architectures have the property of selectively remembering patterns for long durations of time (for a detailed description of the mechanism of LSTM cells, have a look at my short [article](https://raphaellederman.github.io/articles/rnn/#)). 

The concept of time distributed convolutional neural network is to combine a deep hierarchical CNN feature extraction architecture with a recurrent neural network model that can learn to recognize  sequential dynamics in a speech signal. Unlike the SVM approach, we will no longer work on global statistics generated on features from time and frequency domain. This network only takes the log-mel-spectrogram (presented in previous section) as input. 

## The methodology

The main idea of time distributed convolutional neural network is to apply a rolling window (fixed size and time-step) all along the log-mel-spectrogram. Each of these windows will be the entry of a convolutional neural network, composed by four Local Feature Learning Blocks (LFLBs) and the output of each of these convolutional networks will be fed into a recurrent neural network composed by 2 LSTM cells in order to learn the long-term contextual dependencies. Finally, a fully connected layer with softmax activation is used to predict the emotion detected in the voice. 

![image](https://raphaellederman.github.io/assets/images/CNNLSTM.png){:height="100%" width="100%"}

## Model parameters

In the following table, you can see the parameters of the model's layers.

![image](https://raphaellederman.github.io/assets/images/params.png){:height="100%" width="100%"}

To limit overfitting during training phase, we split our data set into train (80\%), validation (15\%) and test set (5\%). We also added early stopping to stop the training when the validation accuracy starts to decrease while the training accuracy steadily increases. We chose Stochastic Gradient Descent with decay and momentum as optimizer and a batch size of 64. 

## Model performance

Following graphics present loss (categorical cross-entropy) and accuracy for both train and validation set:

![image](https://raphaellederman.github.io/assets/images/loss_ac.png){:height="100%" width="100%"}

This deep learning model yielded a maximum score of \textbf{74\%} on the validation set and \textbf{72\%} on the test set. The use of deep learning and time distributed convolutional neural network allows us to achieve a 10\% higher performance compared to the traditional approach using SVM.  

> **Conclusion** : Convolutional Neural Networks (CNNs) and Long-Short Term Memory (LSTM) cells allow extracting features from the log-mel-spectrogram and capturing the temporal dynamic incorporated in speech. Thanks to this deep learning method, it is possible to obtain far higher accuracy than with conventional SVM approach.


