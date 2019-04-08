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

## Mass Index

It uses the high-low range to identify trend reversals based on range expansions. In this sense, it is a volatility indicator that does not have a directional bias. Instead, the Mass Index identifies range bulges that can foreshadow a reversal of the current trend. There are four parts involved in the Mass Index calculation:
* Single EMA : 9-period EMA of the high-low differential.
$$Single\;EMA_{t} = EMA_{9}(P^{High} + P{Low})$$
* Double EMA : 9-period EMA of the 9-period EMA of the high-low differential.
$$Double\;EMA_{t} = EMA_{9}(EMA_{9}(P^{High} + P{Low}))$$
* EMA Ratio : Single EMA divided by Double EMA.
$$EMA\;Ratio_{t} = \frac{Single\;EMA_{t}}{Double\;EMA_{t}}$$ 
* Mass Index : 25-period sum of the EMA Ratio.
$$Mass\;Index_{t} = \sum_{i=1}^{25}EMA\;Ratio_{t-i+1}$$

```python
def mass_index(data, period=25, ema_period=9, high_col='adj_high', low_col='adj_low'):
    high_low = data[high_col] - data[low_col] + 0.000001    #this is to avoid division by zero below
    ema = high_low.ewm(ignore_na=False, min_periods=0, com=ema_period, adjust=True).mean()
    ema_ema = ema.ewm(ignore_na=False, min_periods=0, com=ema_period, adjust=True).mean()
    div = ema / ema_ema

    for index, row in data.iterrows():
        if index >= period:
            val = div[index-25:index].sum()
        else:
            val = 0
        data.at[index, 'mass_index']= val
         
    return data
```

## Average Directional Movement Index

It identifies in which direction the price of an asset is moving by comparing prior highs and lows and drawing two lines: a positive directional movement line ($$DI^+$$) and a negative directional movement line ($$DI^-$$). An optional third line, called directional movement ($$DX$$) gives the signal strength. When $$DI^+$$ is above $$DI^-$$, there is more upward pressure than downward pressure in the price.  Crossovers between the lines are sometimes used as trade signals to buy or sell by technical traders.

$$DI^{+} = \frac{EMA_{14}(DM_{t}^{+})}{TR_{t}}$$
$$DI^{-} = \frac{EMA_{14}(DM_{t}^{-})}{TR_{t}}$$
$${\displaystyle DX=100.{\frac {|DI^{+}-DI^{-}|}{DI^{+}+DI^{-}}}}$$
$${\displaystyle {\textit {ADX}}={EMA_{100}}(DX)}$$

$$\textit{with}$$
$${\displaystyle DM_{t}^{+}=\left\{{\begin{matrix}M_{t}^{+},&\mathrm {if} \ M_{t}^{+}>M_{t}^{-}\ {\textit {and}}\ M_{t}^{+}>0\\0,&\mathrm {if} \ M_{t}^{+}<M_{t}^{-}\ {\textit {or}}\ +M_{t}^{+}<0\end{matrix}}\right.}$$
$${\displaystyle DM_{t}^{-}=\left\{{\begin{matrix}0,&\mathrm {if} \ M_{t}^{-}<M_{t}^{+}\ {\textit {or}}\ M_{t}^{-}<0\\M_{t}^{-},&\mathrm {if} \ M_{t}^{-}>M_{t}^{+}\ {\textit {and}}\ M_{t}^{-}>0\end{matrix}}\right.}$$

$$\textit{where}$$
$${\displaystyle M_{t}^{+}={\textit {P}}_{t}^{High}-{\textit {P}}_{t-1}^{High}}$$
$${\displaystyle M_{t}^{-}={\textit {P}}_{t-1}^{Low}-{\textit {P}}_{t}^{Low}}$$

$$\textit{and}$$
$$TR_{t} = max[(P_{t}^{High} - P_{t}^{Low}), abs(P_{t}^{High} - P_{t-1}^{Close}), abs(P_{t}^{Low} - P_{t-1}^{Close})]$$

![image](https://raphaellederman.github.io/assets/images/adi.png){:height="50%" width="100%"}

```python
def directional_movement_index(data, periods=14, high_col='adj_high', low_col='adj_low'):
    remove_tr_col = False
    if not 'true_range' in data.columns:
        data = average_true_range(data, drop_tr = False)
        remove_tr_col = True

    data['m_plus'] = 0.
    data['m_minus'] = 0.
    
    for i,row in data.iterrows():
        if i>0:
            data.at[i, 'm_plus'] = row[high_col] - data.at[i-1, high_col]
            data.at[i, 'm_minus'] =  row[low_col] - data.at[i-1, low_col]
    
    data['dm_plus'] = 0.
    data['dm_minus'] = 0.
    
    for i,row in data.iterrows():
        if row['m_plus'] > row['m_minus'] and row['m_plus'] > 0:
            data.at[i, 'dm_plus']= row['m_plus']
            
        if row['m_minus'] > row['m_plus'] and row['m_minus'] > 0:
            data.at[i, 'dm_minus']= row['m_minus']
    
    data['di_plus'] = (data['dm_plus'] / data['true_range']).ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data['di_minus'] = (data['dm_minus'] / data['true_range']).ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    
    data['dxi'] = np.abs(data['di_plus'] - data['di_minus']) / (data['di_plus'] + data['di_minus'])
    data.at[0, 'dxi']=1.
    data['adx'] = data['dxi'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    data = data.drop(['m_plus', 'm_minus', 'dm_plus', 'dm_minus'], axis=1)
    if remove_tr_col:
        data = data.drop(['true_range'], axis=1)
         
    return data
```

## Typical Price

Sometimes called the pivot point, it refers to the arithmetic average of the high, low, and closing prices for a given period. Some investors use the Typical Price rather than the closing price when creating moving average penetration systems.

$$Typical\;Price_t = (P_{t}^{High} + P_{t}^{Low} + P_{t}^{Close})/3$$

```python
def typical_price(data, high_col = 'adj_high', low_col = 'adj_low', close_col = 'adj_close'):
    
    data['typical_price'] = (data[high_col] + data[low_col] + data[close_col]) / 3

    return data
```

## Money Flow Index

It is an oscillator that uses price and volume for identifying overbought or oversold conditions in an asset. Unlike conventional oscillators such as the Relative Strength Index (RSI), the Money Flow Index incorporates both price and volume data, as opposed to just price. For this reason, some analysts call MFI the volume-weighted RSI.

$$Positive\;Money\;Flow_{t} = \begin{cases}Typical\;Price_{t} \cdot Volume_{t}, & \text{if $Typical\;Price_{t} - Typical\;Price_{t-1} > 0$}.\\0, & \text{if $Typical\;Price_t - Typical\;Price_{t-1} < 0$}.\end{cases}$$

$$Negative\;Money\;Flow_{t} = \begin{cases}Typical\;Price_{t} \cdot Volume_{t}, & \text{if $Typical\;Price_t - Typical\;Price_{t-1} < 0$}.\\0, & \text{if $Typical\;Price_t - Typical\;Price_{t-1} > 0$}.\end{cases}$$

$$Money\;Flow\;Ratio_t = (\frac{1}{14}\sum_{i=1}^{14}Positive\;Money\;Flow_{t-i+1})/(\frac{1}{14}\sum_{i=1}^{14}Negative\;Money\;Flow_{t-i+1})$$

$$Money\;Flow\;Index_t = 100 - \frac{100}{(1 + Money\;Flow\;Ratio_t)}$$

![image](https://raphaellederman.github.io/assets/images/mfi.png){:height="50%" width="100%"}

```python
def money_flow_index(data, periods=14, vol_col='adj_volume'):
    remove_tp_col = False
    if not 'typical_price' in data.columns:
        data = typical_price(data)
        remove_tp_col = True
    
    data['money_flow'] = data['typical_price'] * data[vol_col]
    data['money_ratio'] = 0.
    data['money_flow_index'] = 0.
    data['money_flow_positive'] = 0.
    data['money_flow_negative'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            if row['typical_price'] < data.at[index-1, 'typical_price']:
                data.at[index, 'money_flow_positive']= row['money_flow']
            else:
                data.at[index, 'money_flow_negative']= row['money_flow']
    
        if index >= periods:
            period_slice = data['money_flow'][index-periods:index]
            positive_sum = data['money_flow_positive'][index-periods:index].sum()
            negative_sum = data['money_flow_negative'][index-periods:index].sum()

            if negative_sum == 0.:
                #this is to avoid division by zero below
                negative_sum = 0.00001
            m_r = positive_sum / negative_sum

            mfi = 1-(1 / (1 + m_r))

            data.at[index, 'money_ratio'] =m_r
            data.at[index, 'money_flow_index']= mfi
        
    data = data.drop(['money_flow', 'money_ratio', 'money_flow_positive', 'money_flow_negative'], axis=1)
    
    if remove_tp_col:
        data = data.drop(['typical_price'], axis=1)

    return data
```

## Negative and Positive Volume Index

The Negative Volume Index (NVI) is a cumulative indicator that uses the change in volume to decide when institutional investors are active. It works under the assumption that institutional investors are active on days when volume decreases and the more retail investors are active on days when volume increases. The NVI is computed by first forming a cumulative line by adding the percentage price change when volume decreases from one period to the other. The cumulative NVI line stays unchanged when volume increases from one period to the other. Starting at 1000, the NVI Values are applied each period to create a cumulative indicator that only changes when volume decreases. A 255-day EMA is then computed on this cumulative NVI. The Positive Value Index fulfils the same use as the NVI, but increases on days when volume has increased from the previous trading day.

![image](https://raphaellederman.github.io/assets/images/pvi.png){:height="50%" width="100%"}

```python
def negative_volume_index(data, periods=255, close_col='adj_close', vol_col='adj_volume'):
    data['nvi'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            prev_nvi = data.at[index-1, 'nvi']
            prev_close = data.at[index-1, close_col]
            if row[vol_col] < data.at[index-1, vol_col]:
                nvi = prev_nvi + (row[close_col] - prev_close / prev_close * prev_nvi)
            else: 
                nvi = prev_nvi
        else:
            nvi = 1000
        data.at[index, 'nvi'] =nvi
    data['nvi_ema'] = data['nvi'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()
    
    return data

def positive_volume_index(data, periods=255, close_col='adj_close', vol_col='adj_volume'):
    data['pvi'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            prev_pvi = data.at[index-1, 'pvi']
            prev_close = data.at[index-1, close_col]
            if row[vol_col] > data.at[index-1, vol_col]:
                pvi = prev_pvi + (row[close_col] - prev_close / prev_close * prev_pvi)
            else: 
                pvi = prev_pvi
        else:
            pvi = 1000
        data.at[index, 'pvi']= pvi
    data['pvi_ema'] = data['pvi'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean()

    return data
```
## TRIX

It is a technical analysis oscillator showing the slope (i.e. derivative) of a triple-smoothed exponential moving average. It is obtained by smoothing prices a first time using an N-day EMA, then smoothing that series using another N-day EMA, and finally smoothing the resulting series using a further N-day EMA. The TRIX at time $$t$$ is then the percentage difference between today's and yesterday's value in the final smoothed series.

$$TRIX_{t} = \Delta\;EMA_{10}(EMA_{10}(EMA_{10}(P^{Close})))$$

![image](https://raphaellederman.github.io/assets/images/trix.png){:height="50%" width="100%"}

## Discrete Fourier Transforms

It is commonly used in order to display several long and short-term trends and eliminate noise in the data. Mathematically speaking, these transforms take a time series and map it into a frequency spectrum. It decomposes a function into sinusoids of different frequencies. Given a discrete-time signal $$x_n$$, $$n=0$$,$$ … $$, $$N-1$$, the Discrete Fourier Transform can be defined as :
$$ X_k =  \sum\limits_{n=0}^{N-1} x_ke^{\frac{-i2\pi}{N}kn} \hspace{1cm} k=0, ..., N-1 $$
The Discrete Fourier Transform outputs sequence of N coefficient $$X_k$$ constituting the frequency domain representation of a signal. 

```python
def fourier(data, close_col='adj_close',  n_components = [3, 6, 9, 100]):
    close_fft = np.fft.fft(np.asarray(data[close_col].tolist()))
    data['fft'] = close_fft
    fft_list = np.asarray(data['fft'].tolist())
    for num_ in n_components:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        data['ifft_'+str(num_)]=np.real(np.fft.ifft(fft_list_m10))
    return data
```

![image](https://raphaellederman.github.io/assets/images/fourier.png){:height="50%" width="100%"}

> **Conclusion** : in this second article about stock market prediction, we have presented our data retrieving methodology using the Quandl API. Moreover, we have presented some technical indicators and features that might enrich our classification model. In the following article, we will present a state-of-the-art method for unsupervised feature extraction applied to time series using a custom Bidirectional Generative Adversarial Network.
