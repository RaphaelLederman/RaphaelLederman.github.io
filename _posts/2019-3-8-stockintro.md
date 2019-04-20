---
published: true
title: Introduction - Data Retrieving
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

In this series of articles, we will describe a stock market prediction procedure using advanced machine learning models. Instead of defining this problem as a traditional forecasting-style problem, we try to predict whether prices will increase or decrease : the problem is posed as a classification problem, where the class labels indicate an
increase or a decrease in the price of a stock.

I am aware that, theoretically speaking, trying to predict stock price movements using statistical learning methods can be seen as absurd : it simply goes against Eugene Fama's efficient market hypothesis. In $$\textit{A Random Walk Down Wall Street}$$ (1973), Burton Malkiel, famous American economist and writer, claimed that stock prices could not be accurately predicted by using historical data. He argued that prices are best described by a random walk, each day's deviations from the central value being random and unpredictable. Nevertheless, some markets are manifestly characterized by a form of inefficiency, and this is partly related to investors' cognitive biases.

Financial investment yields being uncertain, individuals affect occurrence probabilities to future events in order to compute expectations, making their anticipations biased. For this reason, investors often make sub-optimal and irrational decisions : they are greatly influenced by instincts and emotions and do not seem to be able to maximize their own expected utility function. Individuals consistently commit cognitive errors and use almost always mental simplifications in the collection and processing of information. These biases can have a strong impact on the price formation mechanism, and can potentially be exploited in order to generate profitable investment strategies. For instance, certain stock prices exhibit a high correlation with the volume of Google search requests for related terms (see [here](https://editorialexpress.com/cgi-bin/conference/download.cgi?db_name=SNDE2018&paper_id=100)).

We can therefore state that stock price movements are determined in non-negligible proportion by highly irrational behaviors, including the following:
* Anchoring: attachment of thoughts to an unfounded reference point
* Confirmation bias: overweighting of information confirming the investor's original idea
* Hindsight bias: past events considered obvious and predictable
* Gambler's fallacy: occurrence of a particular random event considered as less likely to happen after a certain event
* Herding: tendency of individuals to mimic actions of a broader group
* Overconfidence: overestimation of one's capacity to successfully perform a task
* Overreaction: excessively emotional reactions to newly available information
* Availability bias: decisions oriented towards interpretation of more recent information
* Prospect asymmetry: gains and losses are valued differently

Behavioural finance is an investment approach that includes in its models a psychological perspective, showing how emotional factors and cognitive biases can influence the investment decisions of individual. The emergence of this field and its progressive gain in popularity is a good evidence of the inefficiency of some of the most mature markets like the stock market. This is one of the reason why I tried to build a predictive model based on historical data : if individuals are characterized by strong cognitive biases, then some patterns should reappear over time.

In order predict stock price movements, we will construct a multi-input deep learning classifier that takes as input different types of data, from pure historical price time series, to technical indicators and Fourier transforms. Additionally, we will learn how to generate a rich representation of this information (historical data and additional indicators) through a custom Bidirectional Generative Adversarial Network. This unsupervised feature learning methodology allows the extraction of meaningful information on the stock price and manual features while being able to generalize on a relatively small amount of data. This alternative representation of our historical data and indicators will also be used as input to the deep learning classifier. Our model architecture combines both Convolutional Neural Networks and Recurrent Neural Networks : it includes 2D convolutions and Long Short Term Memory (LSTM) cells (a particular type of Recurrent Neural Network cell capable of efficiently capturing long term dependencies in sequences). Finally, we will describe the implementation of a long/short trading strategy based on this model, and conclude on the potential profitability but also limits of such an approach.

In this first article, we will firt describe our data retrieving methodology based on the Quandl API (see [here](https://www.quandl.com/tools/api) for their website).

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Retrieving stock market data through the Quandl API

The Quandl API is free and consistent : it allows retrieving and manipulating stock market data in various formats (.csv, .xml, .json). The API calls are very simple, and the quality of the data is sufficient to build our model.

We used this API to retrieve adjusted open, low, high and close prices as well as volumes for a universe of 96 Nasdaq stocks. We used data from January 1st 2013 to December 31st 2018, holding out the first 365 days of data for testing our final classifier. 

Here is the code I wrote in order to gather the data into a single dataframe.

```python
import quandl

quandl.ApiConfig.api_key = 'XXX'

class nasdaq():
    def __init__(self):
        self.company_list = './companylist.csv'

    def build_url(self, symbol):
        url = 'https://www.quandl.com/api/v3/datasets/WIKI/{}.csv?api_key={}'.format(symbol, quandl_api_key)
        return url

    def symbols(self):
        symbols = []
        with open(self.company_list, 'r') as f:
            next(f)
            for line in f:
                symbols.append(line.split(',')[0].strip())
        return symbols
    
def download(symbols):
    print('Downloading {}'.format(symbols))
    try:
        data = quandl.get_table('WIKI/PRICES', ticker = symbols, qopts = { 'columns': ['ticker', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'] }, date = { 'gte': '2013-01-01', 'lte': '2018-12-31' }, paginate=True)
        return data
    except Exception as e:
        print('Failed to download {}'.format(symbol))
        print(e)
        
def download_all():
    nas = nasdaq()
    tickers = nas.symbols()
    return download(tickers)

if __name__ == '__main__':
    df_stocks = download_all()
    df_stocks = df_stocks.set_index('date')
```

> **Conclusion** : in this first brief article about stock market prediction, we have presented our data retrieving methodology using the Quandl API. In the following article, we will describe some of the technical indicators and features that we have chosen to include in our study.