---
published: true
title: Speech Emotion Recognition - Feature Extraction (2)
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

In this second part, we will present in more details the audio features extraction methodology and different speech features typically used in the context of speech emotion recognition.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Short-term audio features

Once partitioning done, we can extract 34 features from time (3) and frequency (31) domains for each frame. Features from time domain are directly extracted from the raw signal sample while frequency features are based on the magnitude of the Discrete Fourier Transform. 

**Time-domain features**

In following formulas, $$x_i(n)$$, $$n=0$$, $$...$$, $$N-1$$ is the $$n$$th discrete time signal of the $$i$$th frame and $$N$$ the number of samples per frame (window size).

* Energy: sum of squares of the signal values, normalized by the respective frame length.

$$E_i = \frac{1}{N} \sum\limits_{n=0}^{N-1} \mid x_i(n) \mid^2$$

* Entropy of Energy: entropy of sub-frames normalized energies. It allows measuring abrupt changes in the energy amplitude of an audio signal. To compute the Entropy of Energy of the $$i$$th frame, we divide each frame into K sub-frames of fixed duration, compute the Energy of each sub-frame and then divide it by the total Energy of the frame $$E_i$$.

$$e_j = \frac{E_{subFrame_j}}{E_i}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where $$E_i = \sum\limits_{j=1}^{K} E_{subFrame_j}$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Finally, the entropy $$H_i$$ is computed according to the equation:

$$H_i = -\sum\limits_{j=1}^{K} e_j.log_2(e_j)$$

* Zero Crossing rate: rate of sign-changes of an audio signal.

$$ZCR_i = \frac{1}{2N} \sum\limits_{n=0}^{N-1} \mid sgn[x_i(n)]-sgn[x_i(n-1)] \mid$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where $$sgn(.)$$ is the sign function.

**Frequency-domain features**

In the following formulas, $$X_i(k)$$, $$k=0$$,$$...$$, $$N-1$$ is the $$k$$th Discrete Fourier Transform coefficient of the $$i$$th frame and $$N$$ is the number of samples per frame (window size).

* Spectral centroid: center of gravity of the sound spectrum.

$$C_i = \frac{\sum\limits_{k=0}^{N-1} kX_i(k)} {\sum\limits_{k=0}^{N-1}X_i(k)}$$

* Spectral spread: second central moment of the sound spectrum.

$$S_i = \sqrt{\frac{\sum\limits_{k=0}^{N-1}(k - C_i)^2X_i(k)}{\sum\limits_{k=0}^{N-1}X_i(k)}}$$

* Spectral entropy: entropy of sub-frames normalized spectral energies. To compute the spectral entropy of the $$i$$th frame, we first divide each frame into K sub-frames of fixed size, compute spectral Energy (similar formula as time-domain energy) of each sub-frame and divide it by the total Energy of the frame. The spectral entropy $$H_i$$ is then computed according to the equation:

$$H_i = -\sum\limits_{k=1}^{K} n_k.log_2(n_n)$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;with $$n_k = \frac{E_{subFrame_k}}{\sum\limits_{j=1}^{K} E_{subFrame_j}}$$

* Spectral flux: squared difference between the normalized magnitudes of the spectra of two successive frames. It permits to mesure the spectral changes between two frames.

$$F_i = \sum\limits_{k=0}^{N-1} [EN_i(k) - EN_{i-1}(k) ]^2$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;with $$EN_i(k)=\frac{X_i(k)}{\sum\limits_{l=0}^{N-1}X_i(l)}$$

* Spectral rolloff: frequency below which 90\% of the magnitude distribution of the spectrum is concentrated. The $$l$$th DFT coefficient corresponds to the spectral rolloff if it satisfies the following conditions:

$$\sum\limits_{k=0}^{l-1}X_i(k)=0.90\sum\limits_{k=0}^{N-1}X_i(k)$$

* MFCCs: Mel Frequency Cepstral Coefficients model the spectral energy distribution in a perceptually meaningful way. These features are the most widely used audio features for speech emotion recognition. The following process allows computing the MFCCs of the $$i$$th frame. Calculate the periodogram of the power spectrum of the $$i$$th frame:

$$P_i(k) = \frac{1}{N}\mid X_i(k)\mid^2$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Apply the Mel-spaced filterbank (set of L triangular filters) to the periodogram and calculate the energy in each filter. Finally, we take the Discrete Cosinus Transform (DCT) of the logarithm of all filterbank energies and only keep the first 12 DCT coefficients 
$$C^l_{l=1,...,12}$$:

$$C_{i}^l = \sum\limits_{k=1}^{L} (log \tilde{E_{i}^k}) cos[l(k-\frac{1}{2})\frac{\pi}{L}] \hspace{1cm} l=1, ..., L$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where $$\tilde{E_k}$$ is the energy at the ouptut of the $$kth$$ filter on the $$ith$$ frame.

## Features engineering

Once having extracted the speech features from the preprocessed audio signal, we obtain a matrix of featurese. We then compute the first and second derivatives of each of those features in order to capture frame to frame changes in the signal. Finally, we calculate the following global statistics on these features (original, first and second derivatives): mean, standard deviation, kurtosis, skewness, 1st and 99th percentile. Thereby a vector of 360 candidate features is obtained for each audio signal.

![image](https://raphaellederman.github.io/assets/images/Audio_feature_extraction.png){:height="250%" width="250%"}

Some post-processing also may be necessary before training and testing a classifier. First, normalization could be meaningful as extracted feature values have different orders of magnitude and may be expressed in different units. Secondly, it is also common to use dimensionality reduction techniques to reduce memory usage and improve computational efficiency. There are two options for dimensionality reduction: features selection using statistical tests and features transformation (Principal Component Analysis for instance).

> **Conclusion** : we went through a features extraction methodology and presented some of the most common audio features in the context of speech emotion recognition. In the next part we will provide some details about building a classifier in order to map our features matrices to emotion labels.