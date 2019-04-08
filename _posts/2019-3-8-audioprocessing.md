---
published: true
title: Speech Emotion Recognition - Signal Preprocessing (1)
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

## Introduction

The purpose of this series of 3 article is to present how to identify the emotional state of a human being from his voice. In order to stay in line with the academic litterature, we will focus only on the 6 emotional states introduced by Ekman:
* happiness
* sadness
* angriness
* disgust
* fear
* surprise

The usual process for speech emotion recognition consists of three parts: signal processing, feature extraction and finally classification. Signal processing consists in applying acoustic filters on original audio signals and splitting it into units. The second step involves the  extraction of features that are both efficiently characterizing the emotional content of a speech and not depending on the lexical content or identity of the speaker. Finally, the classification will map features matrices to emotion labels.

![image](https://raphaellederman.github.io/assets/images/Process_Speech_Classification.png){:height="100%" width="100%"}

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Database

Here we will use the $$\textbf{RAVDESS}$$ database in order to test our methodology. It contains acted emotions speech of male and female actors (gender balanced) that were asked to play six different emotions (happy, sad, angry, disgust, fear, surprise and neutral) at two levels of emotional intensity. Based on the research paper written by Thurid Vogt and Elisabeth André, "Improving Automatic Emotion Recognition from Speech via Gender Differentiation" (2006), we separate out the male and female emotions using the identifiers provided in order to implement gender-dependent emotion classifiers. In their paper, separating male and female voices improved the overall recognition rate of their classifier by 2-4%.   

## Signal preprocessing

The following part describes the audio signal preprocessing that has to be done before extracting audio features.

### Pre-emphasis filter

Before starting the features extraction, it’s common to apply a pre-emphasis filter on the audio signal to amplify high frequencies. This pre-emphasis filter has several advantages: it allows both balancing the frequency spectrum (since high frequencies usually have smaller magnitudes compared to lower ones) and avoiding numerical problems on the Fourier Transform computation.

$$y_t =  x_t - \alpha x_{t-1}$$

Typical values for the pre-emphasis filter coefficient $$\alpha$$ are 0.95 or 0.97.

### Framing

After applying the pre-emphasis filter, we split the audio signal into short-term windows called $$\textit{frames}$$. For speech processing, the window size is usually ranging from 20ms to 50ms with 40% to 50% overlap between two consecutive windows. One of the most popular settings is 25ms for the frame size with a 15ms overlap (10ms window step). 

The main motivation behind this step is to avoid the loss of the frequency contours of an audio signal over time due to it's non-stationary nature. As frequency properties in a signal change over time, it does not really make sense to apply the Discrete Fourier Transform across the entire sample. Supposing that frequencies in a signal are constant over a very short period of time, we can apply the Discrete Fourier Transform over these short time windows and obtain a good approximation of the frequency contours of the entire signal.

### Hamming

After splitting the signal into multiple frames, we multiply each frame by a Hamming window function on order to reduce spectral leakage or any signal discontinuities and improve the signal's clarity. By applying a Hamming function, we make sure that the beginning and the end of a frame match up while smoothing the signal. The following equation describes the Hamming window function.

$$H_n =  \alpha - \beta cos(\frac{2\pi n}{N-1})$$

where $$\alpha=0.54$$, $$\beta=0.46$$ and $$0\leq n\leq N-1$$ with $$N$$ the window length.

### Discrete Fourier Transform

The Discrete Fourier Transform is the most widely used transform in all areas of digital signal processing because it allows converting a sequence from the time domain to the frequency domain. DCT provides a convenient representation of the distribution of the frequency content of an audio signal. The use of this transform is crucial because the majority of audio features extracted to analyze speech emotion are defined in the frequency domain.

Given a discrete-time signal $$x_n$$, $$n=0$$,$$ … $$, $$N-1$$, the Discrete Fourier Transform can be defined as :
$$ X_k =  \sum\limits_{n=0}^{N-1} x_ke^{\frac{-i2\pi}{N}kn} \hspace{1cm} k=0, ..., N-1 $$

The Discrete Fourier Transform outputs sequence of N coefficient $$X_k$$ constituting the frequency domain representation of a signal. The inverse Discrete Fourier Transform takes Discrete Fourier coefficient and returns the original signal in the time-domain:
$$x_n = \frac{1}{N} \sum\limits_{k=0}^{N-1} x_ne^{\frac{i2\pi}{N}kn} \hspace{1cm} n=0, ..., N-1$$

> **Conclusion** : we went through the first basic steps of audio signal preprocessing. In the next short article we will use this preprocessing in order to extract relevant audio features.