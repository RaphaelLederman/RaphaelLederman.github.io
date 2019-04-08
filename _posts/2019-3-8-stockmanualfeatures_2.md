---
published: true
title: Manual Feature Extraction (2/2)
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

In this third article, we will describe some other technical indicators and features that we chose to include in our analysis.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Price Volume Trend

It is a momentum based indicator used to measure money flow. It is comparable to the On-Balance-Volume in the way that it is an accumulation of volume. While the OBV adds or subtracts total daily volume depending on if it was an up day or a down day, PVT only adds or subtracts a portion of the daily volume. The amount of volume added or subtracted to/from the PVT total is dependent on the amount of the current day's price rising or falling compared to the previous day's close. Price Volume Trend can primarily be used to confirm trends, as well as spot possible trading signals due to divergences.

$$PVT_{t} = PVT_{t-1} + [((P_{t}^{Close} - P_{t-1}^{Close}) / P_{t-1}^{Close}) \cdot Volume_{t}]$$

![image](https://raphaellederman.github.io/assets/images/pvt.png){:height="50%" width="100%"}

```python
def price_volume_trend(data, trend_periods=21, close_col='adj_close', vol_col='adj_volume'):
    for index, row in data.iterrows():
        if index > 0:
            last_val = data.at[index - 1, 'pvt']
            last_close = data.at[index - 1, close_col]
            today_close = row[close_col]
            today_vol = row[vol_col]
            current_val = last_val + (today_vol * (today_close - last_close) / last_close)
        else:
            current_val = row[vol_col]

        data.at[index, 'pvt']= current_val

    data['pvt_ema' + str(trend_periods)] = data['pvt'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
        
    return data
```

