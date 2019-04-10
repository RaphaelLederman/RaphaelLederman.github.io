---
published: true
title: Data Retrieving
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

In this series of articles on stock market prediction, we will describe a complete procedure for classifying whether a stock will gain or lose value over the next trading days using state-of-the-art machine learning models. For this purpose, we will construct a multi-input deep learning classifier that takes as input different types of data, from pure historical price time series, to technical indicators and Fourier transforms. Additionally, we will learn how to generate a rich representation of this information (historical data and additional indicators) through a custom Bidirectional Generative Adversarial Network. This unsupervised feature learning methodology allows the extraction of meaningful information on the stock price and manual features while being able to generalize on a relatively small amount of data. This alternative representation of our historical data and indicators will also be used as input to the deep learning classifier. Our model architecture combines both Convolutional Neural Networks and Recurrent Neural Networks : it includes 2D convolutions and Long Short Term Memory (LSTM) cells (a particular type of Recurrent Neural Network cell capable of efficiently capturing long term dependencies in sequences). Finally, we will describe the implementation of a long/short trading strategy based on this model, and conclude on the potential profitability of such an approach.

In this first article, we will firt describe our data retrieving methodology based on the Quandl API (see [here](https://www.quandl.com/tools/api) for their website).

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Retrieving stock market data through the Quandl API

The Quandl API is free and consistent : it allows retrieving and manipulating stock market data in various formats (.csv, .xml, .json). The API calls are very simple, and the quality of the data is sufficient to build our model.

We used this API to retrieve adjusted open, low, high and close prices as well as volumes for a universe of 96 Nasdaq stocks. We used data from January 1st 2010 to December 31st 2018, holding out 2 years of data for testing our final classifier. 

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
        data = quandl.get_table('WIKI/PRICES', ticker = symbols, qopts = { 'columns': ['ticker', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'] }, date = { 'gte': '2010-01-01', 'lte': '2018-12-31' }, paginate=True)
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