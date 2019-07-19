---
published: true
title: Performance Analysis and Conclusion
collection: articles
layout: single
author_profile: false
read_time: true
categories: [articles]
excerpt : "Stock Market Prediction"
header :
    overlay_image: "https://raphaellederman.github.io/assets/images/night.jpg"
    teaser_image: "https://raphaellederman.github.io/assets/images/night.jpg"
toc: true
toc_sticky: true
---

In this last article on stock market prediction, we will present the results of our study, and conclude on the limits and potential improvements of our model. It is important to stay aware that there is no 'magic formula' for predicting the stock market. If generating strong and persistent absolute returns over a long-term horizon was only a matter of a few lines of code, I wouldn't be studying financial markets and data science so conscientiously. Most of the authors of articles about stock market prediction using deep learning (and especially LSTM cells) published on the internet lack intellectual honesty : results seem astonishing at first sight, but are disappointing after a more thorough analysis.

As an example, we can have a look at this S&P500 prediction using a simple LSTM neural network with 1 hidden layer and 20 neurons (see [here](https://www.blueskycapitalmanagement.com/machine-learning-in-finance-why-you-should-not-use-lstms-to-predict-the-stock-market/) for the article).

![image](https://raphaellederman.github.io/assets/images/lstmpred1.png){:height="80%" width="160%"}

The low Root Mean Square Error, reasonnable $$R^2$$, and the proximity of the predicted price to the real price could lead us to believe that the predictive model performs very well. Nevertheless, if you look more closely you can see that the prediction made for the next day is very close to the actual value of the previous day. Lagging the actual prices by 1 day compared to the predicted prices, we obtain the following chart.

![image](https://raphaellederman.github.io/assets/images/lstmpred2.png){:height="80%" width="160%"}

As Andrea Leccese, President and Portfolio Manager at Bluesky Capita, says : "The best guess the model can make is a value almost identical to the current day’s price. [...] This is what would be expected by a model that has no predictive ability."

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Classification Results

There are different ways to present classificatipn results. We could for instance present the Area Under the Receiving Operating Characteristics curve (or AUC-ROC curve) of our predictions. The AUC-ROC curve is a performance measurement for classification problems: ROC curves are used in clinical biochemistry to choose the most appropriate cut-off for a test (the best cut-off has the highest true positive rate together with the lowest false positive rate). For a more detailed description of the AUC-ROC curve, have a look at this [article](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5).

Here, for the sake of brevity, we chose to simply display the the confusion matrix. After training our model on 96 different Nasdaq stocks, with sequences from 2013 to 2016, we were obtained the following confusion matrix obtained on our test set (365 trading days from December 2016 to March 2018).

![image](https://raphaellederman.github.io/assets/images/confusion.png){:height="80%" width="160%"}

These results show that delivering alpha using advanced deep learning algorithms is not that simple. Stock markets may not be as efficient as Eugene Fama has stated, but the day we can easily predict stock movements with statistical learning has not yet arrived.

## Potential Improvements

In order to improve our model, we could add different types of data, especially indicators based on Natural Language Processing. Investors largely rely on news flows in order to direct their investment decisions, and text mining (sentiment analysis, topic modelling etc.) could help capturing some of the information that is not perfectly reflected by prices. This could be done both at a micro and at a macro scale : we could build global economic sentiment indicators as well as tactical allocation signals based on corporate-level information (company-related news, 10K reports etc.).

Moreover, a larger panel of securities prices could be included in our model, from sector indices to derivatives or any type of correlated asset (bonds, FX, alternative risk factors etc.). For instance, the massive hedging flows arising from the banks' structured product business can massively impact options prices : we could exploit some patterns linked to supply and demand mechanisms on the equity derivatives market to improve our model. 

Another way of improving our predictive model would be to fine tune the hyperparameters with Bayesian optimization (with a Python library like Hyperopt). For more information on this method, have a look at this short [article](https://towardsdatascience.com/shallow-understanding-on-bayesian-optimization-324b6c1f7083).

Concerning the model itself, other architectures could be explored, including for instance LSTM neural networks with embedded layer (see [here](https://www.scitepress.org/papers/2018/67499/67499.pdf)) or optimized Artificial Neural Networks using genetic algorithm (see [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155133)). We have also tested other types of classifiers, from tree based models (Random Forests and Extra Trees) to boosting algorithms (XGBoost), as their mechanism is comparable to a certain extent to the rules-based investment process of most technical traders. Overall, neural networks seem to be the best at capturing patterns and dependencies in time series in the context of our study.