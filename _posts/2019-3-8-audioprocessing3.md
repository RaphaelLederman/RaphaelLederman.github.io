---
published: true
title: Speech Emotion Recognition - Classification (3)
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

For this last short article on speech emotion recognition, we will briefly present some common approaches to classifying emotions from audio features.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Classifier

In the literature, various machine learning algorithms based on acoustic features are used to construct classifiers. Support Vector Machines (SVM) with non-linear kernels are the most popular and often the most successfully applied algorithms for speech emotion recognition. 
A SVM using non-linear kernel transforms the input feature vectors into a higher dimensional feature space using a kernel mapping function. By choosing appropriate non-linear kernels functions, classifiers that are non-linear in the original space can become linear in the feature space. Most common kernel function are described below: 
* Linear kernel: $$K(x_i,x_j)= x_i * x_j$$
* Radial Basis Function (rbf) kernel: $$K(x_i,x_j) = exp(- \frac{\mid\mid x_i - x_j\mid\mid^2}{2\sigma^2})$$
* d-degree polynomial kernel: $$ K(x_i,x_j)= (x_i * x_j + \gamma)^d$$

For a presentation of the SVM classifier, you can check my [article](https://raphaellederman.github.io/articles/svm/#).

State of the art paper "Speech emotion recognition: Features and classification models" by L. Chen, X. Mao, Y. Xue and L. L. Cheng achieved an accuracy of 86.5% by combining principal component analysis and SVM respectively for dimensionality reduction and classification. Some sophisticated classifiers based on deep learning do achieve higher rates but we will go through such architectures in another article.

## Implementation

We will first implement SVM classifiers based on different kernel functions (linear, polynomial and RBF), without dimensionality reduction and gender differentiation. The accuracies obtained and shown in next table are relatively low, with the SVM with RBF kernel function being the best performer with an accuracy rate of 56.51%. 
Then we will have a look at both feature selection (1%-Chi-squared test removed 75 features) and features transformation (PCA) in order to perform dimensionality reduction. For PCA, three levels of explained variance were tested (90%, 95% and 98%) respectively leading to the following features dimensions : 100, 120 and 140. The performances are still relatively low but the accuracy of polynomial and RBF increased respectively by 6% and 3% with the 140 feature dimension corresponding to the 98% contribution. SVM with RBF kernel remains the most accurate classifier. 

![image](https://raphaellederman.github.io/assets/images/audiotable1.png){:height="100%" width="100%"}

The first major improvement can be observed with the implementation of gender differentiation as suggested above. As shown in the following table, accuracy scores of almost all classifiers (except for 3-degree polynomial) increased by around 5%. The next figure illustrates accuracy rates obtained by cross-validation (1) and the confusion matrix of the classifier with the highest accuracy score: RBF Kernel and PCA 180 features dimension (corresponding to 98% contribution) (2). 

![image](https://raphaellederman.github.io/assets/images/audiotable2.png){:height="100%" width="100%"}

![image](https://raphaellederman.github.io/assets/images/audiotable3.png){:height="100%" width="100%"}

As shwon above, $$\textit{surprise}$$ and $$\textit{neutral}$$ emotions obtained the poorest classification accuracies compared to other emotions such as $$\textit{scared}$$ and $$\textit{disgust}$$. The initial $$\textbf{RAVDESS}$$ database contains speeches for 7 different emotions, but our classifier has trouble differentiating $$\textit{surprise}$$ from other emotions : we can choose to delete it from our dataset. Final results look satisfyinge : we obtained an accuracy score of almost 75%, as shown in following table.

![image](https://raphaellederman.github.io/assets/images/audiotable4.png){:height="100%" width="100%"}

![image](https://raphaellederman.github.io/assets/images/audiotable5.png){:height="100%" width="100%"}

Our model presents reasonably satisfying results : our accuracy rate is around 65% for 7-way (happy, sad, angry, scared, disgust, surprised, neutral) emotions and 75% for  6-way emotions (surprised removed).

> **Conclusion** : in order to improve our results, more complex models such as Hidden Markov Models (HMM), Convolutional Neural Networks (CNN) and Long-Short Term Memory (LSTM) could be interesting as they better capture the temporal dynamic incorporated in speech. Unlike SVM classifiers, these classifiers can be trained on all short-term features and not on descriptive statistics.


