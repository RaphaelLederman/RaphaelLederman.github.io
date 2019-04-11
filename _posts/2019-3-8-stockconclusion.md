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

In this last article on stock market prediction, we will present the results of our study, and conclude on the limits and potential improvements of our model. Although our model seems to reach a high accuracy, it is important to stay aware that there is no 'magic formula' for predicting the stock market. If generating strong and persistent absolute returns over a long-term horizon was only a matter of a few lines of code, I wouldn't be studying financial markets and data science so conscientiously.
Moreover, we chose not to include potential transaction costs (commissions, bid-ask spreads, market impact) in our study : such costs could strongly erode the profitability of an investment strategy based on our model.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Classification Results

After training on 96 different Nasdaq stocks, with sequences from 2014 to 2018, we were able to obtain relatively satisfying results.

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