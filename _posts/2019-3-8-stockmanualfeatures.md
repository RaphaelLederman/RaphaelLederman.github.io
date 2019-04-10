---
published: true
title: Manual Feature Extraction
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

In this second article, we are going to describe some of the additional features and indicators that will be used as input to our deep learning classifier. I am personally not convinced by the reliability of technical indicators in the context of trading strategies, but technical analysts believe that patterns in the ups and downs of equity prices can be valuable indicators of the security's future price movements. This statement goes against the efficient market hypothesis (EMH), stating that asset prices fully reflect all available information and challenging the notion that past price and volume data can have any relationship with future movements. Nevertheless, some traders contend that if enough investors are using similar technical valuation techniques, a self-fulfilling prophesy might emerge : this is the reason why we chose to add such indicators in our dataset in order to exploit existing patterns.

In order to understand the patterns that might affect whether a stock price will move up or down, we need to build the most complete dataset possible, including the various technical indicators and features we chose to include in our analysis.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Discrete Fourier Transforms

It is commonly used in order to display several long and short-term trends and eliminate noise in the data. Mathematically speaking, these transforms take a time series and map it into a frequency spectrum. It decomposes a function into sinusoids of different frequencies. Given a discrete-time signal $$x_n$$, $$n=0$$,$$ … $$, $$N-1$$, the Discrete Fourier Transform can be defined as :
$$ X_k =  \sum\limits_{n=0}^{N-1} x_ke^{\frac{-i2\pi}{N}kn} \hspace{1cm} k=0, ..., N-1 $$
The Discrete Fourier Transform outputs a sequence of N coefficient $$X_k$$ constituting the frequency domain representation of a signal. 

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

## Exponential Moving Average

It is a particular type of moving average that places a greater weight to the most recent data points. Compared to a simple equally weighted moving average, this indicator will react more notably to recent price shocks (it will turn before simple moving averages). Simple moving averages may be better suited to identify support or resistance levels. Like all moving average indicators, this EMA is a lagging indicator, and is therefore better suited for trending markets :  conclusions drawn from applying a moving average should be to confirm a market move or to indicate its strength.

$$S_{t}=\begin{cases}Y_{1}, & \text{if $t=1$}.\\\alpha \cdot Y_{t} + (1-\alpha) \cdot S_{t-1}, & \text{if $t>1$}.\end{cases}$$

Where the coefficient $$\alpha$$ represents the degree of weighting decrease, a constant smoothing factor between 0 and 1. A higher $$\alpha$$ discounts older observations faster. $$Y_{t}$$ is the value at a time period $$t$$. $$S_{t}$$ is the value of the EMA at any time period $$t$$.

![image](https://raphaellederman.github.io/assets/images/ema.png){:height="50%" width="100%"}

```python
def ema(data, period=12, column='adj_close'):
    data['ema' + str(period)] = data[column].ewm(ignore_na=False, min_periods=period, com=period, adjust=True).mean()
    
    return data
```

## Moving Average Convergence Divergence (MACD)

It is a trend following momentum indicator calculated by subtracting the 26-period EMA from the 12-period EMA. We also use a nine-day EMA of this MACD, called the "signal line", which can function as a trigger for buy and sell signals. Traders may buy the security when the MACD crosses above its signal line and short sell the security when the MACD crosses below the signal line. The relationship between the MACD and the signal line can be used to identify crossovers (for instance when the MACD crosses above its signal line following a brief correction within a longer-term uptrend), divergence (when the MACD forms highs or lows that diverge from the corresponding highs and lows on the price), rapid rises (a shorter-term moving average pulling away from the longer-term moving average is sometimes seen by traders as a signal that the security is overbought or oversold, and will soon return to normal levels).

$$MACD = EMA_{12}(P^{Close}) - EMA_{26}(P^{Close})$$

![image](https://raphaellederman.github.io/assets/images/macd.png){:height="50%" width="100%"}

```python
def macd(data, period_long=26, period_short=12, period_signal=9, column='adj_close'):
    remove_cols = []
    if not 'ema' + str(period_long) in data.columns:
        data = ema(data, period_long)
        remove_cols.append('ema' + str(period_long))

    if not 'ema' + str(period_short) in data.columns:
        data = ema(data, period_short)
        remove_cols.append('ema' + str(period_short))

    data['macd_val'] = data['ema' + str(period_short)] - data['ema' + str(period_long)]
    data['macd_signal_line'] = data['macd_val'].ewm(ignore_na=False, min_periods=0, com=period_signal, adjust=True).mean()

    data = data.drop(remove_cols, axis=1)
        
    return data
```

## Momentum

It compares where the current price is in relation to where the price was in the past. The choice of the lag period can vary depending on the security and the strategy.

$$Momentum_{t}^{N} = P_{t}^{Close} - P_{t-N}^{Close}$$

![image](https://raphaellederman.github.io/assets/images/momentum.png){:height="50%" width="100%"}

```python
def momentum(data, periods=14, close_col='adj_close'):
    data['momentum'] = 0.
    
    for index,row in data.iterrows():
        if index >= periods:
            prev_close = data.at[index-periods, close_col]
            val_perc = (row[close_col] - prev_close)/prev_close

            data.at[index, 'momentum']= val_perc

    return data
```

## Bollinger bands

It is a technical analysis tool defined by a set of lines plotted two standard deviations away from the simple moving average of the security's price. The first step is to compute the simple moving average of the security in question. Next, the rolling standard deviation of the security's price has to be computed. Rolling standard deviation can be calculated by taking the square root of the variance, which itself is the average of the squared differences of the mean. Next, we multiply that standard deviation value by two and both add and subtract that amount from each point along the simple moving average. These computations produce the upper and lower bands. Some traders believe the closer the prices move to the upper band, the more overbought the market, and the closer the prices move to the lower band, the more oversold the market. A set of different rules can be defined and followed when using the bands as a trading system. For instance, a $$\textit{squeeze}$$ (when the bands come close together, constricting the moving average) can signal a period of low volatility and therefore represent a potential sign of future increased volatility. A breakout above or below the bands, while not providing any clue as to the direction and extent of future price movement, can also be used as an indicator providing information regarding price volatility.

Bollinger bands consist of the three following lines :
* An n-period moving average : $$MA_{n}$$
* An upper band at K times an n-period standard deviation above the moving average :
$$Up = (MA_{n} + K \cdot \sigma_{n})$$
* A lower band at K times an n-period standard deviation below the moving average :
$$Down = (MA_{n} - K \cdot \sigma_{n})$$

![image](https://raphaellederman.github.io/assets/images/bollinger.png){:height="50%" width="100%"}

```python
def bollinger_bands(data, trend_periods=20, close_col='adj_close'):

    data['bol_bands_middle'] = data[close_col].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    for index, row in data.iterrows():

        s = data[close_col].iloc[index - trend_periods: index]
        sums = 0
        middle_band = data.at[index, 'bol_bands_middle']
        for e in s:
            sums += np.square(e - middle_band)

        std = np.sqrt(sums / trend_periods)
        d = 2
        upper_band = middle_band + (d * std)
        lower_band = middle_band - (d * std)

        data.at[index, 'bol_bands_upper'] =upper_band
        data.at[index, 'bol_bands_lower']= lower_band

    return data
```

## Typical Price

Sometimes called the pivot point, it refers to the arithmetic average of the high, low, and closing prices for a given period. Some investors use the Typical Price rather than the closing price when creating moving average penetration systems.

$$Typical\;Price_t = (P_{t}^{High} + P_{t}^{Low} + P_{t}^{Close})/3$$

![image](https://raphaellederman.github.io/assets/images/typical.png){:height="50%" width="100%"}


```python
def typical_price(data, high_col = 'adj_high', low_col = 'adj_low', close_col = 'adj_close'):
    
    data['typical_price'] = (data[high_col] + data[low_col] + data[close_col]) / 3

    return data
```

## On-Balance-Volume

It is a technical trading momentum indicator that uses volume flow to predict changes in stock price. OBV is generally used to confirm price moves: the idea is that volume is higher on days where the price move is in the dominant direction, for example in a strong uptrend there is more volume on up days than down days.

$$OBV=  OBV_{prev} + 
\begin{cases}
    volume,& \text{if } close > close_{prev}\\
    0,&       \text{if } close = close_{prev}\\
    -volume,& \text{if } close < close_{prev}
\end{cases}$$

![image](https://raphaellederman.github.io/assets/images/obv.png){:height="50%" width="100%"}

```python
def on_balance_volume(data, trend_periods=21, close_col='adj_close', vol_col='adj_volume'):
    for index, row in data.iterrows():
        if index > 0:
            last_obv = data.at[index - 1, 'obv']
            if row[close_col] > data.at[index - 1, close_col]:
                current_obv = last_obv + row[vol_col]
            elif row[close_col] < data.at[index - 1, close_col]:
                current_obv = last_obv - row[vol_col]
            else:
                current_obv = last_obv
        else:
            last_obv = 0
            current_obv = row[vol_col]

        data.at[index, 'obv'] = current_obv

    data['obv_ema' + str(trend_periods)] = data['obv'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    
    return data
```

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

![image](https://raphaellederman.github.io/assets/images/mass.png){:height="50%" width="100%"}

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

## Relative Strength Index (RSI)

It is a momentum oscillator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions. RSI is considered overbought when above 70 and oversold when below 30 (these thresholds can be adjusted to better fit a particular security). For each trading period an upward change $$U_{t}$$ and downward change $$D_{t}$$ is calculated (if the last close is the same as the previous, both $$U_{t}$$ and $$U_{t}$$ are zero). 

$$U_{t}=\begin{cases}S_{t} - S_{t-1}, & \text{if $S_{t}>S_{t-1}$}.\\0, & \text{if $S_{t}<S_{t-1}$}.\end{cases}$$

$$D_{t}=\begin{cases}0, & \text{if $S_{t}>S_{t-1}$}.\\S_{t-1} - S_{t}, & \text{if $S_{t}<S_{t-1}$}.\end{cases}$$

The average $$U$$ and $$D$$ are then calculated using an n-period smoothed or modified moving average (SMMA) which is an exponentially smoothed moving average with $$\alpha = \frac{1}{period}$$. The ratio of these averages is the relative strength factor.

$$RSF = \frac{SMMA(U)}{SMMA(D)}$$. 

The relative strength factor is then converted to a relative strength index between 0 and 100:

$$RSI = 100 - \frac{100}{1+RSF}$$. 

If the average of $$D$$ values is zero, then according to the equation, the RS value will approach infinity, so that the resulting RSI will approach 100.

![image](https://raphaellederman.github.io/assets/images/rsi.png){:height="50%" width="100%"}

```python
def rsi(data, periods=14, close_col='adj_close'):
    data['rsi_u'] = 0.
    data['rsi_d'] = 0.
    data['rsi'] = 0.
    
    for index,row in data.iterrows():
        if index >= periods:
            
            prev_close = data.at[index-periods, close_col]
            if prev_close < row[close_col]:
                data.at[index, 'rsi_u']= row[close_col] - prev_close
            elif prev_close > row[close_col]:
                data.at[index, 'rsi_d']= prev_close - row[close_col]
            
    data['rsi'] = data['rsi_u'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean() / (data['rsi_u'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean() + data['rsi_d'].ewm(ignore_na=False, min_periods=0, com=periods, adjust=True).mean())
    
    data = data.drop(['rsi_u', 'rsi_d'], axis=1)
        
    return data
```

## Accumulation Distribution Line

It is a volume-based indicator designed to measure the cumulative flow of money into and out of a security. It is a running total of each period's Money Flow Volume. First, a multiplier is calculated based on the relationship between the close and the high-low range. Second, the Money Flow Multiplier is multiplied by the period's volume to come up with a Money Flow Volume. Finally, a running total of the Money Flow Volume forms the Accumulation Distribution Line.

$$Money\;Flow\;Multiplier_{t} = [(P_{t}^{Close}  -  P_{t}^{Low}) - (P_{t}^{High} - P_{t}^{Close})] /(P_{t}^{High} - P_{t}^{Low})$$ 

$$Money\;Flow\;Volume_{t} = Money\;Flow\;Multiplier_{t} \cdot Volume_{t}$$

$$ADL_{t} = ADL_{t-1} + Money\;Flow\;Volume_{t}$$

The indicator is used to either reinforce the underlying trend or cast doubts on its sustainability : an uptrend in prices with a downtrend in the Accumulation Distribution Line suggests underlying selling pressure that could foreshadow a bearish reversal on the price chart. A downtrend in prices with an uptrend in the Accumulation Distribution Line indicate underlying buying pressure that could foreshadow a bullish reversal in prices.

![image](https://raphaellederman.github.io/assets/images/acc_dist.png){:height="50%" width="100%"}

```python
def acc_dist(data, trend_periods=21, open_col='adj_open', high_col='adj_high', low_col='adj_low', close_col='adj_close', vol_col='adj_volume'):
    for index, row in data.iterrows():
        if row[high_col] != row[low_col]:
            ac = ((row[close_col] - row[low_col]) - (row[high_col] - row[close_col])) / (row[high_col] - row[low_col]) * row[vol_col]
        else:
            ac = 0
        data.at[index, 'acc_dist'] = ac
    data['acc_dist_ema' + str(trend_periods)] = data['acc_dist'].ewm(ignore_na=False, min_periods=0, com=trend_periods, adjust=True).mean()
    
    return data
```

## Williams Accumulation Distribution

It helps determining the trend direction: when price makes a new high and the indicator fails to exceed its previous high, distribution is taking place. When price makes a new low and the WAD fails to make a new low, accumulation is occurring. Usually, technical investors go long when there is a bullish divergence between WAD and price, go short on a bearish divergence. This indicator can be computed as follows:

$$AD_{t}=\begin{cases}P_{t}^{Close}-min(P_{t-1}^{Close}, P_{t}^{Low}), & \text{if $P_{t}^{Close}>P_{t-1}^{Close}$}\\P_{t}^{Close}-max(P_{t-1}^{Close}, P_{t}^{High}), & \text{if $P_{t}^{Close}<P_{t-1}^{Close}$}\\0, & \text{else}\end{cases}$$

$$Williams\;AD_{t} =Williams\;AD_{t-1} + AD_{t} * Volume_t$$

![image](https://raphaellederman.github.io/assets/images/wad.png){:height="50%" width="100%"}

```python
def williams_ad(data, high_col='adj_high', low_col='adj_low', close_col='adj_close'):
    data['williams_ad'] = 0.
    
    for index,row in data.iterrows():
        if index > 0:
            prev_value = data.at[index-1, 'williams_ad']
            prev_close = data.at[index-1, close_col]
            if row[close_col] > prev_close:
                ad = row[close_col] - min(prev_close, row[low_col])
            elif row[close_col] < prev_close:
                ad = row[close_col] - max(prev_close, row[high_col])
            else:
                ad = 0.
                                                                                                        
            data.at[index, 'williams_ad'] = (ad+prev_value)
        
    return data
```

## Williams R

It is a dynamic indicator which determines whether the market is overbought or oversold. It compares where a security's price closed relative to its price range over a given time period. Indicator values ranging between 80 and 100% indicate that the market is oversold. Indicator values ranging between 0 and 20% indicate that the market is overbought.

$$\%R_{t} = -100 \cdot \frac{Highest\;High_t - P_{t}^{Close}}{Highest\;High_t - Lowest\;Low_{t}}$$

$$\%R_{t} = -100 \cdot \frac{max\{P_{t-i+1}^{High}:\;i =1..n\} - P_{t}^{Close}}{max\{P_{t-i+1}^{High}:\;i =1..n\} - min\{P_{t-i+1}^{Low}:\;i =1..n\} }$$

![image](https://raphaellederman.github.io/assets/images/wr.png){:height="50%" width="100%"}

```python
def williams_r(data, periods=14, high_col='adj_high', low_col='adj_low', close_col='adj_close'):
    data['williams_r'] = 0.
    
    for index,row in data.iterrows():
        if index > periods:
            data.at[index, 'williams_r'] = ((max(data[high_col][index-periods:index]) - row[close_col]) / 
                                                 (max(data[high_col][index-periods:index]) - min(data[low_col][index-periods:index])))
        
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

## Average Directional Movement Index

It identifies in which direction the price of an asset is moving by comparing prior highs and lows and drawing two lines: a positive directional movement line ($$DI^+$$) and a negative directional movement line ($$DI^-$$). An optional third line, called directional movement ($$DX$$) gives the signal strength. When $$DI^+$$ is above $$DI^-$$, there is more upward pressure than downward pressure in the price.  Crossovers between the lines are sometimes used as trade signals to buy or sell by technical traders.

$$DI^{+} = \frac{EMA_{14}(DM_{t}^{+})}{TR_{t}}$$
$$DI^{-} = \frac{EMA_{14}(DM_{t}^{-})}{TR_{t}}$$
$${\displaystyle DX=100.{\frac {|DI^{+}-DI^{-}|}{DI^{+}+DI^{-}}}}$$
$${\displaystyle {\textit {ADX}}={EMA_{100}}(DX)}$$

$$\textit{with}$$
$$DM_{t}^{+}=
\begin{cases}
    M_{t}^{+},& \text{if } M_{t}^{+}>M_{t}^{-} {\text{and}}\ M_{t}^{+}>0\\
    0,&       \text{if } M_{t}^{+}<M_{t}^{-} {\text{or}}\ M_{t}^{+}<0\\
\end{cases}$$
$$DM_{t}^{-}=
\begin{cases}
    0,&       \text{if } M_{t}^{-}<M_{t}^{+} {\text{or}}\ M_{t}^{-}<0\\
    M_{t}^{+},& \text{if } M_{t}^{-}>M_{t}^{+} {\text{and}}\ M_{t}^{-}>0\\
\end{cases}$$

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


> **Conclusion** : in this second article about stock market prediction, we have presented some technical indicators and features that might enrich our classification model. In the following article, we will present a method for unsupervised feature extraction applied to time series using a custom Bidirectional Generative Adversarial Network.