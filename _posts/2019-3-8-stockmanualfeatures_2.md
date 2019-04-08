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


## Average True Range

It is a technical analysis indicator that measures market volatility by decomposing the entire range of an asset price for that period. The True Range indicator is taken as the greatest of the following: 
* Current high less the current low
* Absolute value of the current high less the previous close
* Absolute value of the current low less the previous close.
The average true range is then a moving average of the true ranges.

$$TR_{t} = max[(P_{t}^{High} - P_{t}^{Low}), abs(P_{t}^{High} - P_{t-1}^{Close}), abs(P_{t}^{Low} - P_{t-1}^{Close})]$$

$$ATR_{t} = \frac{1}{n} \sum_{i=1}^{n}TR_{t-i+1}$$

![image](https://raphaellederman.github.io/assets/images/atr.png){:height="50%" width="100%"}

```python
def average_true_range(data, trend_periods=14, open_col='adj_open', high_col='adj_high', low_col='adj_low', close_col='adj_close', drop_tr = True):
    for index, row in data.iterrows():
        prices = [row[high_col], row[low_col], row[close_col], row[open_col]]
        if index > 0:
            val1 = np.amax(prices) - np.amin(prices)
            val2 = abs(np.amax(prices) - data.at[index - 1, close_col])
            val3 = abs(np.amin(prices) - data.at[index - 1, close_col])
            true_range = np.amax([val1, val2, val3])

        else:
            true_range = np.amax(prices) - np.amin(prices)

        data.at[index, 'true_range']= true_range
    data['atr'] = data['true_range'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    if drop_tr:
        data = data.drop(['true_range'], axis=1)
        
    return data
```


## Chaikin Oscillator

It applies MACD to the accumulation-distribution line rather than closing price. To calculate the Chaikin Oscillator, we subtract a 10-day EMA of the accumulation-distribution line from a 3-day EMA of the accumulation-distribution line. This measures momentum predicted by oscillations around the accumulation-distribution line.

$$Chaikin\;Oscillator_{t} = EMA_{3}(ADL) - EMA_{10}(ADL)$$

![image](https://raphaellederman.github.io/assets/images/chaikin.png){:height="50%" width="100%"}

```python
def chaikin_oscillator(data, periods_short=3, periods_long=10, high_col='adj_high',
                       low_col='adj_low', close_col='adj_close', vol_col='adj_volume'):
    ac = pd.Series([])
    val_last = 0
    
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            val = val_last + ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            val = val_last
        ac.at[index]= val
    val_last = val

    ema_long = ac.ewm(ignore_na=False, min_periods=0, com=periods_long, adjust=True).mean()
    ema_short = ac.ewm(ignore_na=False, min_periods=0, com=periods_short, adjust=True).mean()
    data['ch_osc'] = ema_short - ema_long

    return data
```

## Ease of Movement

This oscillator shows the relationship between price and volume and fluctuates around zero. In general, prices are advancing with relative ease when the oscillator is in positive territory. Conversely, prices are declining with relative ease when the oscillator is in negative territory. There are two parts to the EMV formula: 
* Distance Moved : it is calculated by comparing the current period's midpoint with the prior period's midpoint, which is the mean between the high and the low. Distance Moved is positive when the current midpoint is above the prior midpoint and negative when the current midpoint is below the prior midpoint.
$$Distance\;Moved_{t} = ((P_{t}^{High} + P_{t}^{Low}) / 2 - (P_{t-1}^{High} + P_{t-1}^{Low})/2)$$
* Box Ratio : it increases with volume and decreases with the high-low range
$$Box\;Ratio_{t} = ((Volume_{t}/100,000,000)/(P_{t}^{High} + P_{t}^{Low}))$$

$$EMV_{t}^{1\;period} = Distance\;Moved_{t} / Box\;Ratio_{t}$$

$$EMV_{t}^{n\;period} = \frac{1}{n} \sum_{i=1}^{n}EMV_{t-i+1}^{1\;period}$$

![image](https://raphaellederman.github.io/assets/images/eom.png){:height="50%" width="100%"}

```python
def ease_of_movement(data, period=14, high_col='adj_high', low_col='adj_low', vol_col='adj_volume'):
    for index, row in data.iterrows():
        if index > 0:
            midpoint_move = (row[high_col] + row[low_col]) / 2 - (data.at[index - 1, high_col] + data.at[index - 1, low_col]) / 2
        else:
            midpoint_move = 0
        
        diff = row[high_col] - row[low_col]
        
        if diff == 0:
            #this is to avoid division by zero below
            diff = 0.000000001          
            
        vol = row[vol_col]
        if vol == 0:
            vol = 1
        box_ratio = (vol / 100000000) / (diff)
        emv = midpoint_move / box_ratio
        
        data.at[index, 'emv']= emv
        
    data['emv_ema_'+str(period)] = data['emv'].ewm(ignore_na=False, min_periods=0, com=period, adjust=True).mean()
        
    return data
```

