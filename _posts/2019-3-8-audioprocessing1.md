---
published: true
title: Signal Processing
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

The purpose of this article is to present how to identify the emotional state of a human being from his voice. In order to stay in line with the academic litterature, we will focus only on the 6 emotional states introduced by Ekman:
* happiness
* sadness
* angriness
* disgust
* fear
* surprise.
The usual process for speech emotion recognition consists of three parts: signal processing, feature extraction and finally classification. Signal processing consists in applying acoustic filters on original audio signals and splitting it into units. The second step involves the  extraction of features that are both efficiently characterizing the emotional content of a speech and not depending on the lexical content or identity of the speaker. Finally, the classification will map features matrices to emotion labels.

## Database

Here we will use the $$\textbf{RAVDESS}$$ database in order to test our methodology. It contains acted emotions speech of male and female actors (gender balanced) that were asked to play six different emotions (happy, sad, angry, disgust, fear, surprise and neutral) at two levels of emotional intensity. Based on $$\textbf{Thurid Vogt and Elisabeth André}$$ research: $$\textit{Improving Automatic Emotion Recognition from Speech via Gender Differentiation (2006)}$$, we separate out the male and female emotions using the identifiers provided in order to implement gender-dependent emotion classifiers. In their paper, separating male and female voices improved the overall recognition rate of their classifier by 2-4\%.   

## Signal preprocessing

The following part describes the audio signal preprocessing that has to be done before extracting audio features.

### Pre-emphasis filter

Before starting the features extraction, it’s common to apply a pre-emphasis filter on the audio signal to amplify high frequencies. This pre-emphasis filter has several advantages: it allows both balancing the frequency spectrum (since high frequencies usually have smaller magnitudes compared to lower ones) and avoiding numerical problems on the Fourier Transform computation.

$$y_t =  x_t - \alpha x_{t-1}$$


Typical values for the pre-emphasis filter coefficient $\alpha$ are 0.95 or 0.97.

### Framing

After applying the pre-emphasis filter, we split the audio signal into short-term windows called $$\textit{frames}$$. For speech processing, the window size is usually ranging from 20ms to 50ms with 40\% to 50\% overlap between two consecutive windows. One of the most popular settings is 25ms for the frame size with a 15ms overlap (10ms window step). 

The main motivation behind this step is to avoid the loss of the frequency contours of an audio signal over time due to it's non-stationary nature. As frequency properties in a signal change over time, it does not really make sense to apply the Discrete Fourier Transform across the entire sample. Supposing that frequencies in a signal are constant over a very short period of time, we can apply the Discrete Fourier Transform over these short time windows and obtain a good approximation of the frequency contours of the entire signal.

### Hamming

After splitting the signal into multiple frames, we multiply each frame by a Hamming window function on order to reduce spectral leakage or any signal discontinuities and improve the signal's clarity. By applying a Hamming function, we make sure that the beginning and the end of a frame match up while smoothing the signal. The following equation describes the Hamming window function.

$$H_n =  \alpha - \beta cos(\frac{2\pi n}{N-1})$$

where $$\alpha=0.54$$, $$\beta=0.46$$ and $$0\leq n\leq N-1$$ with $$N$$ the window length.


### Discrete Fourier Transform

The Discrete Fourier Transform is the most widely used transform in all areas of digital signal processing because it allows converting a sequence from the time domain to the frequency domain. DCT provides a convenient representation of the distribution of the frequency content of an audio signal. The use of this transform is crucial because the majority of audio features extracted to analyze speech emotion are defined in the frequency domain.

Given a discrete-time signal $$x_n$ $n=0, … N-1$$ (N samples long) the Discrete Fourier Transform can be defined as
$$ X_k =  \sum\limits_{n=0}^{N-1} x_ke^{\frac{-i2\pi}{N}kn} \hspace{1cm} k=0, ..., N-1 $$

The Discrete Fourier Transform outputs sequence of N coefficient $X_k$ constituting the frequency domain representation of a signal. The inverse Discrete Fourier Transform takes Discrete Fourier coefficient and returns the original signal in the time-domain:
$$x_n = \frac{1}{N} \sum\limits_{k=0}^{N-1} x_ne^{\frac{i2\pi}{N}kn} \hspace{1cm} n=0, ..., N-1$$

## Features extraction

### Short-term audio features

Once partitioning done, we can extract 34 features from time (3) and frequency (31) domains for each frame. Features from time domain are directly extracted from the raw signal sample while frequency features are based on the magnitude of the Discrete Fourier Transform. 

**Time-domain features**

In following formulas, $$x_i(n)$$, $$n=0$$, $$...$$, $$N-1$$ is the $$n$$th discrete time signal of the $$i$$th frame and $$N$$ the number of samples per frame (window size).

>Energy: sum of squares of the signal values, normalized by the respective frame length
$$E_i = \frac{1}{N} \sum\limits_{n=0}^{N-1} \mid x_i(n) \mid^2$$

>Entropy of Energy: entropy of sub-frames normalized energies. It allows measuring abrupt changes in the energy amplitude of an audio signal. To compute the Entropy of Energy of the $$i$$th frame, we divide each frame into K sub-frames of fixed duration, compute the Energy of each sub-frame and then divide it by the total Energy of the frame $$E_i$$:
$$e_j = \frac{E_{subFrame_j}}{E_i}$$
where $$E_i = \sum\limits_{j=1}^{K} E_{subFrame_j}$$
Finally, the entropy $H_i$ is computed according to the equation:
$$H_i = -\sum\limits_{j=1}^{K} e_j.log_2(e_j)$$

>Zero Crossing rate: rate of sign-changes of an audio signal
$$ZCR_i = \frac{1}{2N} \sum\limits_{n=0}^{N-1} \mid sgn[x_i(n)]-sgn[x_i(n-1)] \mid$$
Where $$sgn(.)$$ is the sign function.

**Frequency-domain features**

In the following formulas, $$X_i(k)$$, $$k=0$$,$$...$$, $$N-1$$ is the $$k$$th Discrete Fourier Transform coefficient of the $$i$$th frame and $$N$$ is the number of samples per frame (window size).

>Spectral centroid: center of gravity of the sound spectrum.
$$C_i = \frac{\sum\limits_{k=0}^{N-1} kX_i(k)} {\sum\limits_{k=0}^{N-1}X_i(k)}$$

>Spectral spread: second central moment of the sound spectrum.
$$S_i = \sqrt{\frac{\sum\limits_{k=0}^{N-1}(k - C_i)^2X_i(k)}{\sum\limits_{k=0}^{N-1}X_i(k)}}$$

>Spectral entropy: entropy of sub-frames normalized spectral energies. To compute the spectral entropy of the $$i$$th frame, we first divide each frame into K sub-frames of fixed size, compute spectral Energy (similar formula as time-domain energy) of each sub-frame and divide it by the total Energy of the frame. The spectral entropy $$H_i$$ is then computed according to the equation:
$$H_i = -\sum\limits_{k=1}^{K} n_k.log_2(n_n)$$
with 
$$n_k = \frac{E_{subFrame_k}}{\sum\limits_{j=1}^{K} E_{subFrame_j}}$$

>Spectral flux: squared difference between the normalized magnitudes of the spectra of two successive frames. It permits to mesure the spectral changes between two frames.
$$F_i = \sum\limits_{k=0}^{N-1} [EN_i(k) - EN_{i-1}(k) ]^2$$
with
$$EN_i(k)=\frac{X_i(k)}{\sum\limits_{l=0}^{N-1}X_i(l)}$$

>Spectral rolloff: frequency below which 90\% of the magnitude distribution of the spectrum is concentrated. The $$l$$th DFT coefficient corresponds to the spectral rolloff if it satisfies the following conditions:
$$\sum\limits_{k=0}^{l-1}X_i(k)=0.90\sum\limits_{k=0}^{N-1}X_i(k)$$

>MFCCs: Mel Frequency Cepstral Coefficients model the spectral energy distribution in a perceptually meaningful way. These features are the most widely used audio features for speech emotion recognition. The following process allows computing the MFCCs of the $$i$$th frame. Calculate the periodogram of the power spectrum of the $$i$$th frame:
$$P_i(k) = \frac{1}{N}\mid X_i(k)\mid^2$$
Apply the Mel-spaced filterbank (set of L triangular filters) to the periodogram and calculate the energy in each filter. Finally, we take the Discrete Cosinus Transform (DCT) of the logarithm of all filterbank energies and only keep the first 12 DCT coefficients $$C^l_{l=1,...,12}$$:
$$C_{i}^l = \sum\limits_{k=1}^{L} (log \tilde{E_{i}^k}) cos[l(k-\frac{1}{2})\frac{\pi}{L}] \hspace{1cm} l=1, ..., L$$
where $$\tilde{E_k}$$ is the energy at the ouptut of the $$k$$th filter on the $i$th frame.

### Features engineering

Once having extracted the speech features from the preprocessed audio signal, we obtain a matrix of featurese. We then compute the first and second derivatives of each of those features in order to capture frame to frame changes in the signal. Finally, we calculate the following global statistics on these features (original, first and second derivatives): mean, standard deviation, kurtosis, skewness, 1st and 99th percentile. Thereby a vector of 360 candidate features is obtained for each audio signal.

![image](https://raphaellederman.github.io/assets/images/Audio_feature_extraction.png){:height="150%" width="150%"}

Some post-processing also may be necessary before training and testing a classifier. First, normalization could be meaningful as extracted feature values have different orders of magnitude and may be expressed in different units. Secondly, it is also common to use dimensionality reduction techniques to reduce memory usage and improve computational efficiency. There are two options for dimensionality reduction: features selection using statistical tests and features transformation (Principal Component Analysis for instance).

## Classifier

In literature, various machine learning algorithms based on acoustic features (presented in previous section) are utilized to construct satisfying classifiers for hidden emotion detection in a human speech. Support Vector Machines (SVM) is the most popular and the most often successfully applied algorithm for speech emotion recognition. 
SVM is a non-linear classifier transforming the input feature vectors into a higher dimensional feature space using a kernel mapping function. By choosing appropriate non-linear kernels functions, classifiers that are non-linear in the original space can therefore become linear in the feature space. Most common kernel function are described below: 
\begin{itemize}[label=\textbullet]
\setlength\itemsep{0.5em}
\item \textbf{linear kernel}: $ K(x_i,x_j)= x_i * x_j$
\item \textbf{radial basis function (rbf) kernel}: $ K(x_i,x_j) = exp(- \frac{\mid\mid x_i - x_j\mid\mid^2}{2\sigma^2})$
\item \textbf{d-degree polynomial kernel}: $ K(x_i,x_j)= (x_i * x_j + \gamma)^d$
\end{itemize}

\smallskip
State of the art paper \textit{"Speech emotion recognition: Features and classification models"} by \textbf{L. Chen}, \textbf{X. Mao}, \textbf{Y. Xue}, and \textbf{L. L. Cheng} achieved an accuracy of \textbf{86.5\%} by combining principal component analysis and SVM respectively for dimensionality reduction and classification. Some sophisticated classifiers do achieve higher recognition rates than simple SVM but not much.
In the next section we will try to get as close as possible to the state of the art performances



\subsection{Empirical results}~\\
We first implemented SVM classifiers based on different kernel functions (linear, polynomial and RBF), without dimensionality reduction and gender differentiation. Speech emotion recognition accuracies shown in next table were relatively low. However, the SVM with RBF kernel functions seems to be the best performer with an accuracy rate of 56.51\%. Then we applied both feature selection (1\%-Chi-squared test removed 75 features) and feature transformation (PCA) to reduce the dimension of the features. For PCA, three levels of explained variance were tested (90\%, 95\% and 98\%) respectively leading to the following features dimensions : 100, 120 and 140. Our performances were still very low but the accuracy of polynomial and RBF increased respectively by 6\% and 3\% with the 140 feature dimension corresponding to the 98\% contribution. RBF kernel still remains the best classifier. 

\begin{table}
\centering
\begin{tabular}{ |C{3cm}||C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|  }
 \toprule
 \textbf{PCA dimension}&\textbf{linear}&\textbf{poly (2)}&\textbf{poly (3)}&\textbf{rbf}\\
 \midrule
 None&\textit{51.67\%}&\textit{54.28\%}&\textit{52.79\%}&\textit{56.51\%}\\
 \hline
 140&\textit{53.53\%}&\textit{50.19\%}&\textit{52.79\%}&\textbf{\textit{59.48\%}}\\
 \hline
 120&\textit{55.02\%}&\textit{50.56\%}&\textit{52.79\%}&\textit{58.74\%}\\
 \hline
 100&\textit{52.79\%}&\textit{48.33\%}&\textit{52.79\%}&\textit{58.36\%}\\
 \bottomrule
\end{tabular}
\caption{Different dimension and different kernel cross-validation accuracy rate}
\vspace{-1cm}
\end{table}

The first major improvement was observed with the implementation of gender differentiation as suggested in previous section. As shown in the following table, accuracy scores of almost all classifier (except for 3-degree polynomial) increased by almost 5\%. The next figure illustrates accuracy rates obtained by cross-validation and the confusion matrix of the classifier with the highest accuracy score: RBF Kernel and PCA 180 features dimension (corresponding to 98\% contribution). 

\begin{table}
\centering
\begin{tabular}{ |C{3cm}||C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}| }
 \toprule
 \textbf{PCA dimension}&\textbf{linear}&\textbf{poly (2)}&\textbf{poly (3)}&\textbf{rbf}\\
 \midrule
 None&\textit{53.23\%}&\textit{55.02\%}&\textit{54.28\%}&\textit{60.59\%}\\
 \hline
 180&\textit{59.85\%}&\textit{55.39\%}&\textit{55.76\%}&\textbf{\textit{64.20\%}}\\
 \bottomrule
\end{tabular}
\caption{Gender differentiation - Cross-validation accuracy rate for different dimension and different kernel.}
\vspace{-1cm}
\end{table}

\FloatBarrier
\begin{table}[h]
\begin{center}
\addtolength{\leftskip} {-0.6cm} % increase (absolute) value if needed
\addtolength{\rightskip}{0.5cm}
\begin{tabular}{cC{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|}
\cline{3-9} & & \multicolumn{7}{ c| }{\textbf{Predicted labels}}\\
\cline{3-9} & &\textbf{Happy}&\textbf{Sad}&\textbf{Angry}&\textbf{Scared}&\textbf{Neutral}&\textbf{Disgusted}&\textbf{Surprised}\\
\cline{1-9} 
\multicolumn{1}{ |c| }{\multirow{7}{*}{\textbf{\rot{Actual labels}}}} &
\multicolumn{1}{ |c| }{\textbf{Happy}} & \textbf{\textit{65.9\%}} & \textit{4.9\%} & \textit{7.3\%} & \textit{0.0\%} & \textit{7.3\%} & \textit{14.6\%} & \textit{0.0\%}\\
\cline{2-9}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Sad}}& \textit{17.9\%} & \textbf{\textit{61.5\%}} & \textit{7.7\%} & \textit{7.7\%} & \textit{0.0\%} & \textit{0.0\%} & \textit{5.1\%}\\
\cline{2-9}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Angry}}& \textit{7.9\%} & \textit{5.3\%} & \textbf{\textit{63.2\%}} & \textit{2.6\%} & \textit{0.0\%} & \textit{5.3\%} &  \textit{15.8\%}\\
\cline{2-9}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Scared}}& \textit{5.3\%} & \textit{5.3\%} & \textit{0.0\%} & \textbf{\textit{76.3\%}} & \textit{7.9\%} & \textit{2.6\%} & \textit{2.6\%}\\
\cline{2-9}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Neutral}}& \textit{10.3\%} & \textit{5.1\%} & \textit{7.7\%} & \textit{5.1\%} & \textbf{\textit{53.8\%}} & \textit{10.3\%} & \textit{7.7\%}\\
\cline{2-9}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Disgusted}}& \textit{4.5\%} & \textit{0.0\%} & \textit{4.5\%} & \textit{4.5\%} & \textit{6.8\%} & \textbf{\textit{72.7\%}} & \textit{6.8\%}\\
\cline{2-9}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Surprised}}& \textit{3.3\%} & \textit{20.0\%} & \textit{3.3\%} & \textit{6.7\%} & \textit{6.7\%} & \textit{3.3\%} & \textbf{\textit{56.7\%}}\\
\cline{1-9}
\end{tabular}
\caption{Confusion Matrix of best classifier}
\end{center}
\vspace{-0.5cm}
\end{table}
\FloatBarrier

As can be seen above, \textit{Surprise} and \textit{Neutral} emotions were classified with the poorest accuracy compared to other emotions such as \textit{Scared} and \textit{Disgust} who achieved the highest results (respectively 76\% and 73\%). \textbf{RAVDESS} database contains speeches for 7 different emotions but we decided to remove \textit{Surprise}, as our classifier had trouble differentiating it from other emotions. Final results have been quite satisfying. We have succeeded to obtain an accuracy score of almost 75\% as shown in following table.

\begin{table}
\centering
\begin{tabular}{ |C{3cm}||C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}| }
 \toprule
 \textbf{PCA dimension}&\textbf{linear}&\textbf{poly (2)}&\textbf{poly (3)}&\textbf{rbf}\\
 \midrule
 None&\textit{63.34\%}&\textit{59.74\%}&\textit{65.80\%}&\textit{70.26\%}\\
 \hline
 120&\textit{54.50\%}&\textit{63.20\%}&\textit{64.94\%}&\textbf{\textit{74.46\%}}\\
 \bottomrule
\end{tabular}
\caption{6-way emotions - Cross-validation accuracy rate for different dimension and different kernel.}
\vspace{-1cm}
\end{table}

\FloatBarrier
\begin{table}[h]
\begin{center}
\addtolength{\leftskip} {-0.6cm} % increase (absolute) value if needed
\addtolength{\rightskip}{0.5cm}
\begin{tabular}{ cC{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|C{1.5cm}|}
\cline{3-8} & & \multicolumn{6}{ c| }{\textbf{Predicted labels}}\\
\cline{3-8} & &\textbf{Happy}&\textbf{Sad}&\textbf{Angry}&\textbf{Scared}&\textbf{Neutral}&\textbf{Disgusted}\\
\cline{1-8} 
\multicolumn{1}{ |c| }{\multirow{6}{*}{\textbf{\rot{Actual labels}}}} &
\multicolumn{1}{ |c| }{\textbf{Happy}} & \textbf{\textit{80.0\%}} & \textit{0.0\%} & \textit{5.7\%} & \textit{5.7\%} & \textit{5.7\%} & \textit{2.9\%}\\
\cline{2-8}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Sad}}& \textit{8.1\%} & \textbf{\textit{81.1\%}} & \textit{0.0\%} & \textit{0.0\%} & \textit{2.7\%} & \textit{8.1\%}\\
\cline{2-8}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Angry}}& \textit{6.3\%} & \textit{6.3\%} & \textbf{\textit{75\%}} & \textit{0.0\%} & \textit{6.3\%} & \textit{6.3\%}\\
\cline{2-8}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Scared}}& \textit{6.7\%} & \textit{0.0\%} & \textit{4.4\%} & \textbf{\textit{71.1\%}} & \textit{8.9\%} & \textit{8.9\%}\\
\cline{2-8}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Neutral}}& \textit{11.1\%} & \textit{5.6\%} & \textit{2.8\%} & \textit{8.3\%} & \textbf{\textit{66.7\%}} & \textit{5.6\%}\\
\cline{2-8}
\multicolumn{1}{ |c  }{} &
\multicolumn{1}{ |c| }{\textbf{Disgusted}}& \textit{0.0\%} & \textit{8.7\%} & \textit{0.0\%} & \textit{4.3\%} & \textit{2.2\%} & \textbf{\textit{84.8\%}}\\
\cline{1-8}
\end{tabular}
\caption{Confusion Matrix of best classifier - 6-way emotions}
\end{center}
\vspace{-1.5cm}
\end{table}
\FloatBarrier

\subsection{Potential improvements}~\\
Our model presents reasonably satisfying results. Our prediction recognition rate is around 65\% for 7-way (happy, sad, angry, scared, disgust, surprised, neutral) emotions and 75\% for  6-way emotions (surprised removed).

In order to improve our results and to try to get closer to the state of the art, we will try to implement more sophisticated classifiers in second period of this project. For example, Hidden Markov Model (HMM) and Convolutional Neural Networks (CNN) seem to be potential good candidates for speech emotion recognition. Unlike SVM classifiers, those classifiers are train on short-term features and not on global statistics features. HMM and CNN are considered to be advantageous for better capturing the temporal dynamic incorporated in speech. 

To implement a multimodal model for emotion recognition we will also need to set up the removal of silence and probably build a speaker identifier to not bias our emotion predictions in the speech domain.




##########################################
##########################################
##########################################
##########################################
##########################################
##########################################



In this article, we will present some of the tools available in order to preprocess video data for personality traits detection and propose a deep learning classification model adapted to this task. The personality traits we are going to detect are the following : openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism (see [here](https://en.wikipedia.org/wiki/Big_Five_personality_traits) for more information on the Big Five model in psychology). Our original dataset is the First Impression V2 [dataset](http://chalearnlap.cvc.uab.es/dataset/24/description/), comprising 10,000 clips (with an average duration of 15 seconds) extracted from more than 3,000 different YouTube high-definition videos of people facing and speaking in English to a camera. Theses videos were labeled with the personality traits from the Big Five model using [Amazon Mechanical Turk](https://www.mturk.com/) (AMT).

We will first focus on single images preprocessing (detecting faces, extracting facial features...). Then we will show how to transform video inputs into sequences of preprocessed images, and feed these sequences to a deep learning model using CNN and LSTM in order to perform personality traits detection.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Preprocessing : facial detection using the Viola–Jones object detection framework.

Let's start using OpenCV (Open Source Computer Vision Library), an open source library for computer vision written in C/C++ with interfaces in C++, Python and Java. These are a few imports that are going to be useful for our classification.

```python
import cv2
import matplotlib.pyplot as plt
from imutils import face_utils
```

First we will define a function that is able to detect a face on a given image. In order to do so efficiently, we will use a Casacade classifier available with the OpenCV library. These types of classifiers are based on boosting, a family of ensemble learning algorithms which converts weak learner to strong learners by training each weak learner sequentially (each one trying to correct its predecessor). These weak learners can typically be decision trees with a single split (called decision stumps), as in the case of AdaBoost (Adaptive Boosting). 

The Viola–Jones object detection framework is an effective method proposed by Paul Viola and Michael Jones in their [paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It relies on such classifiers : it is the methodology that we will use in order to detect faces on our sequences of images. This algorithm proceeds in four different steps in order to perform accurate detection : Haar feature selection, integral image creation, Adaboost training and finally classifiers cascading.

The Haar features are first extracted from positively and negatively labelled images (with faces and witout faces in it) : these features are just like convolutional kernels, each one being a single value obtained by subtracting the sum of pixels under the white areas from the sum of pixels under the black areas. There are several rectangle structures that can be applied in this fashion, for instance two-rectangle features are mainly used for detecting edges and three-rectangle features mainly used for detecting lines.

![image](https://raphaellederman.github.io/assets/images/haar_features.png){:height="100%" width="100%"}

All possible sizes and locations of each kernel are then used to calculate hundreds of thousands features, and the extraction process roughly corresponds to the following image. It is important to note that these Haar features basically correspond to common human face image features, for instance a dark eye region compared to upper-cheeks.

![image](https://raphaellederman.github.io/assets/images/haar_features_2.png){:height="100%" width="100%"}

In order to improve the efficiency of the rectangle features computation, the authors proposed an intermediate representation for the image : the Integral Image (reducing the calculations for a given pixel to an operation involving just four pixels). This method can also be used for calculating the average intensity within a given image. For a more detailed explanation on the integral image, go visit this [website](https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/).

Then, in order to avoid calculating too many features, and make the extraction process more time efficient, the authors use the AdaBoost algorithm to select the best set of features (single rectangle features which split best negative and positive examples). This is really important because most of the features primarily calculated are irrelevant : if a window relies on the property that the eyes region is generally darker than the bridge of the nose (line feature 2.b for instance in the first image above), it should be applied essentially to this particular region (eyes/nose) and not to other regions (for instance the mouth region). In this way, they combined a set of weak classifiers in order to create an accurate ensemble model with fewer features (the authors achieved an accuracy of 95% while retaining only 200 of the original features, and finally chose to select 6,000 features).

The last crucial step in the Viola–Jones object detection framework is the Cascade of Classifiers. This concept helps discarding non-face regions in an image, so that more time is spent evaluating possible face regions. Instead of applying all 6000 features on a window, the features are grouped into different stages of classifiers and applied one-by-one : if a window fails the first stage, it is automatically discarded and if it passes, the second stage of features is applied and the process goes on. A particular window is considered to be a face region only if it passes all stages (on average only 10 features out of more than 6000 are evaluated per sub-window thanks to this filtering). The authors finally organized the features in 38 stages with 1, 10, 25, 25 and 50 features in the first five stages. The Cascade of Classifiers were trained using Adaboost and adjusting the threshold to minimize the false rate, using the following hyperparameters : the number of classifier stages, the number of features in each stage, the threshold of each stage. In the following image, we can see the two features obtained as the best two features from Adaboost). 

![image](https://raphaellederman.github.io/assets/images/haar_features_3.png){:height="100%" width="100%"}

In the following function, we first locate the pre-trained weights and load them in order to use the model. We then convert our image to grayscale, and apply our cascade classifier in order to detect multiple faces (the scaleFactor and minNeighbors correspond respectively to how much the image size is reduced at each image scale and how many neighbors each candidate rectangle should have to retain it). We then add a rectangle around the detected face on our image, and get back different elements (grayscale picture and detected faces coordinates).

```python
def detect_face(frame):
    
    #Cascade classifier pre-trained model
    cascPath = "/anaconda3/envs/py35/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    #BGR -> Gray conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Cascade MultiScale classifier
    detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,
                                                  minSize=(shape_x, shape_y),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    coord = []
    
    for x, y, w, h in detected_faces :
        if w>500:
            sub_img=frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
            coord.append([x,y,w,h])
        
    return gray, detected_faces, coord
```

Here is an example : we first take a photo of Jimi Hendrix.

![image](https://raphaellederman.github.io/assets/images/hendrix.jpg){:height="100%" width="100%"}

We then apply our function in order to obtain a grayscale image, detect the face and add a rectangle around it.

![image](https://raphaellederman.github.io/assets/images/hendrix_frame.png){:height="100%" width="100%"}

This object extraction method can also be applied to eyes or mouth detection.

## Extracting facial features

We will now extract from our original image a new one, scaled and centered around the detected face, with a predefined resolution (here we chose to produce a 48x48 image).

```python
def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
    gray = faces[0]
    detected_face = faces[1]
    new_face = []
    for det in detected_face :
        if det != ():
            x, y, w, h = det
        	
        	#Offset coefficient, np.floor takes the lowest integer (delete border of the image)
            horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
            vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
            extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]

            #Zoom on the extracted face
            new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
            
            #cast type float
            new_extracted_face = new_extracted_face.astype(np.float32)
            
            #scale
            new_extracted_face /= float(new_extracted_face.max())
            new_face.append(new_extracted_face)
            
        else:
            pass
    
    return new_face,
```

Here is what we obtain when we apply the extract_face_features function to the output of our detect_face function based on our original Jimi Hendrix picture :

![image](https://raphaellederman.github.io/assets/images/zoom_hendrix.png){:height="100%" width="100%"}

## Converting videos to sequences of preprocessed images

The next step is to build a function that transforms any video into a sequence of 48x48 images centered around a detected face. In order to do so, we use VideoCapture from OpenCV in order to split a video input into a sequence of successive images, and then apply our preprocessing in order to obtain standardized images.

```python
def FrameCapture(path):  
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
    count = 0
    success = 1   
    images =[]  
    while success: 
        success, image = vidObj.read()
        if image is not None: 
            image = extract_face_features(detect_face(image))[0] 
            if image !=[]:
                images.append(image)
                count += 1        
        else:
            success = False
    result = np.array([images[i][0] for i in range(len(images))])
    return result
```

We now have a sequence of preprocessed images. In order to build a consistent dataset for video classification, we could reformat each video so that it has a predetermined shape, let's say (420, 48, 48) (each video being therefore composed of 420 different images). In order to do so, we could apply some kind of padding :
* if the video is longer than 420 images, we truncate it and only take the first 420 images.
* if the video is shorter than 420 images, we can sample a sequence from itself and add it at the end in order to fill the gap (for instance if a video is composed of 400 images, we can "repeat" the last 20 images). 
Such a method is not optimal as it affects the temporal consistency of the sequence. Hopefully some deep learning frameworks (like PyTorch) do not need sequences to be padded in order to perform classification or other tasks, but Keras requires sequences to have a constant shape. As we will use Keras in this short article, I will provide a very basic example of such padding for video inputs.

```python
def pad_video(video):
    if video.shape[0] < 420:
        while video.shape[0] < 420:
            video_padded = np.concatenate((video, video[-(420-len(video)):]), axis = 0)
            if video_padded.shape[0] == 420:
                break
    else:
        video_padded = video[:420] 
    return video_padded
```

This preprocessing pipeline can then be applied to the whole First Impression V2 dataset in order to start training a classification model. In order to do so, we create a dictionary with the video names as keys and the corresponding sequences of images as values (it is better to use ordered dictionnaries).

```python
shape_x = 48
shape_y = 48
videonames = []
video_di = {}
local ='XXX/data/train_data/'
directory_names = [f for f in listdir(local) if not(isfile(join(local, f)))]
for di in directory_names:
    mypath = local + di + "/"
    subdirectory_names = [f for f in listdir(mypath) if not(isfile(join(mypath, f)))]
    for subdi in subdirectory_names:
        finalpath = mypath + subdi + "/"
        video_names = [f for f in listdir(finalpath) if isfile(join(finalpath, f))]
        for video in video_names:
            video_di[video] = pad_video(FrameCapture(finalpath+video))
            videonames.append(video)
```

## Building an appropriate classification model

Now that we know how to properly preprocess our video data, we can start building a neural network architecture in order to perform classification. Here, our objective is to choose an architecture that is consistent with the temporal nature of our data : we will use a convolution layer along with an LSTM cell. For more information on Long-Short Term Memory cells, you can have a look at this short [article](https://raphaellederman.github.io/articles/rnn/#) I wrote on Recurrent Neural Networks.

```python
dim = (420,48,48,1)
inputShape = (dim)
Input_words = Input(shape=inputShape, name='input_vid')
x = TimeDistributed(Conv2D(filters=50, kernel_size=(8,8), padding='same', activation='relu'))(Input_words)
x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
x = TimeDistributed(SpatialDropout2D(0.2))(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(x)
out = Dense(5,activation='softmax')(x)
model = Model(inputs=Input_words, outputs=[out])
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss = 'categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])
```

Now we can retrieve our training data and fit the model. We add one dimension to our training sequences as all images are in grayscale (therefore the last dimension is 1, compared to 3 in the case of RGB images). Finally, we retrieve the labels from ordered dictionnaries.

```python
X = np.expand_dims(np.stack(X), axis = 4)
y = np.asarray(list(zip(list(labels['extraversion'].values()), list(labels['neuroticism'].values()), list(labels['agreeablenes'].values()),list(labels['conscientiousness'].values()),list(labels['openness'].values()))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = model.fit(X_train, y_train)
pred = model.predict(X_test)
score = multilabel_accuracy(pred,y_test)
print(score)
```
We could perform some tuning in order to provide better predictions, trying different architectures and using Bayesian optimization to find the best set of hyperparameters, but this is not the aim of this short articles.

> **Conclusion** : in this short article, we went through the most important steps in building a video classification pipeline in the context of personality traits detection with inputs consisting of videos of people facing and speaking in English to a camera. We divided each video into a padded sequence of preprocessed images (scladed and centered around a detected face), and fed these sequences to a deep learning architecture in order to perform classification.