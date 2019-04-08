---
published: true
title: Manual Feature Extraction (1/2)
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

In this second article, we will then describe some of the additional features and indicators that will be used in our deep learning classifier. Finally, we will provide the code for sending requests to the Quandl API, and computing various technical indicators and features on stock data. I am personally not convinced by the reliability of technical indicators in the context of trading strategies, but technical analysts believe that patterns in the ups and downs of equity prices can be valuable indicators of the security's future price movements. This statement goes against the efficient market hypothesis (EMH) stating that asset prices fully reflect all available information and challenging the notion that past price and volume data can have any relationship with future movements. Nevertheless, some traders contend that if enough investors are using similar technical valuation techniques, a self-fulfilling prophesy might emerge : this is the reason why we chose to add such indicators in our dataset in order to exploit existing patterns.

In order to understand the patterns that might affect whether a stock price will move up or down, we need to build the most complete dataset possible, including various technical indicators and features. Let's define some of the features we chose to incorporate in our analysis.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

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

## On-Balance-Volume

It is a technical trading momentum indicator that uses volume flow to predict changes in stock price. OBV is generally used to confirm price moves: the idea is that volume is higher on days where the price move is in the dominant direction, for example in a strong uptrend there is more volume on up days than down days.

$${\displaystyle OBV=OBV_{prev}+\left\{{\begin{matrix}volume&\mathrm {if} \ close>close_{prev}\\0&\mathrm {if} \ close=close_{prev}\\-volume&\mathrm {if} \ close< close_{prev}\end{matrix}}\right.}$$

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
