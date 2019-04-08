---
published: true
title: Expected Return Classification using CNN and LSTM
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

In this fifth article related to stock market predictions, we will go through a deep learning classification algorithm inspired by both Convolutional Neural Networks and Recurrent Neural Networks. This model is innovative in its essence : it takes as input not only regular stock data like daily open, low, high, close prices and volumes, but also manually computed features (including conventional indicators such as short-term and long-term Exponential Moving Averages, Moving Average Convergence Divergence, Bollinger Bands, Price-Volume Trends, ARMA, Fourier Transforms etc.) and a representation of all this data learned through a custom Bidirectional Generative Adversarial Network.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## The inputs of the model

The first input of the model is the original stock data including the normalized daily adjusted open, low, high, close prices and volumes. We then concatenated a set of 50 additional features to this original data, including conventional technical indicators. We finally chose to set the number of historical days to 20 as it proved to obtain the best scores in our preliminary exploration. Therefore we end up with a sequence of 20 days of historical data, including 55 different series, from raw closing prices to Momentum Indicators (see my first article for more information). We will call this input In-1.

The second input of the model is a representation of In-1 learned in an unsupervised fashion through a Bidirectional Generative Adversarial Network. This representation is the concatenation of both the activation from the last layer of convolutional lays of the BiGAN's discriminative model and its encoding into the generating distribution of the BiGAN's generative model (see my second article for a more detailed description). We will call this input In-2.

## The model architecture

The inputs In-1 and In-2 are first treated separately in order to obtain respective intermediate representations. These two separate representations are then summed and fed into the rest of the model.

### First branch : original sotck data and additional features (In-1) 

As In-1 is a structured sequence of prices, volumes and indicators, we chose to apply 2D convolutions to it in order to extract patterns through time. 

Why using CNNs for patterns extraction ? CNNs have seen a lot of success in the domain of constructing features using raw data, often outperforming expert features (computer vision, signal processing etc.). This can be explained by some interesting characteristics of CNNs, such as :
* Local connectivity : during a convolution operation, each output value is not required to be connected to every neuron in the previous layer but only to those, called receptive fields, where the convolution kernel is applied currently
* Parameters sharing : the same weights are applied over the convolving until the next parameters update. Local connectivity and parameters sharing reduce the number of parameters needed compared to usual Artificial Neural Network structures.
* Shift/translation in-variance : invariance means that you can recognize a pattern even when its appearance varies in some way. In the field of computer vision for instance, this is crucial as it preserves an object's identity across changes in the specifics of the visual input (relative positions of the viewer/camera and the object etc.). Translation invariance means that sliding a specific pattern over will have the same output but simply translated over. Convolving $$\textit{f}$$ with $$\textit{g}$$, it doesn't matter if you translate the convolved output$$\textit{f * g}$$, or if you translate $$\textit{f}$$ or $$\textit{g}$$ first, then convolve them.

Back to our classification model, the convolutions's outputs are activated with the ReLU function ($$A(x) = max(0,x)$$), and a dropout layer is then applied. The outputs are then fed into a particular type of Recurrent Neural Network cell : a Long Short Term Memory cell (for a detailed description of the mechanism behind LSTM cells, have a look at my short [article](https://raphaellederman.github.io/articles/rnn/#)). 

Why using an LSTM cell for time series classification ? Long Short Term Memory architectures have an edge over conventional feed-forward neural networks and RNNs as they have the property of selectively remembering patterns for long durations of time. This is made possible by what is called a $$\textit{memory cell}$$. Its unique structure is composed of four main elements : an input gate, a neuron with a self-recurrent connection, a forget gate and an output gate. The self-recurrent connection ensures that the state of a memory cell can remain constant from one timestep to another. The role of the gates is to fine-tune the interactions between the memory cell and its environment using a sigmoid layer and a point-wise multiplication operation.

Back to our classification model again, we use the last output of the LSTM cell, multiply it by a weight matrix and add a bias in order to obtain our intermediate representation of the In-1 input.

### Second branch : BiGAN features (In-2) 

We also apply convolutions to the representation of In-1 obtained through our BiGAN. The convolutions outputs are then activated using the ReLU function, and a dropout layer is applied. The outputs correspond to the intermediate representation of the In-2 input.

### Central branch

Following the sum of the two intermediate representations, we apply two other blocks of convolution-activation-dropout, and flatten the output in order to extract it as a feature representing all the inputs of the model (this could be used in order to investigate the way our final model represents the data). Finally, the flattened output goes through a series of transformation in order to obtain logits that match the label format (two classes). The loss is then computed with the softmax cross-entropy function using the previously computed logits, and an L2 regularization term is added to the total loss. Finally, the ADAM (Adaptive Moment Estimation) is used to minimize the loss.

Here is my code for the definition of the classifier.

```python
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn

SEED = 42
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class CNN():

    def __init__(self, num_features, num_features_gan, num_historical_days, is_train=True, keep_rate = 0.8):

        self.keep_rate = keep_rate
        self.X = tf.placeholder(tf.float32, shape=[None, num_historical_days, num_features])
        self.X_gan = tf.placeholder(tf.float32, shape=[None, num_features_gan])
        X = tf.reshape(self.X, [-1, num_historical_days, 1, num_features])
        X_gan = tf.reshape(self.X_gan, [-1, 1,1, num_features_gan])
        self.Y = tf.placeholder(tf.int32, shape=[None, 2])
        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope("cnn",  reuse = True) as scope:
            scope.reuse_variables()

            k1 = tf.Variable(tf.truncated_normal([3, 1, num_features, 16],
                stddev=0.1,seed=SEED, dtype=tf.float32))
            b1 = tf.Variable(tf.zeros([16], dtype=tf.float32))
            conv = tf.nn.conv2d(X,k1,strides=[1, 1, 1, 1],padding='SAME')
            relu_manual = tf.nn.relu(tf.nn.bias_add(conv, b1))
            if is_train:
                relu_manual = tf.nn.dropout(relu_manual, keep_prob = self.keep_prob)

            n_units = 320
            w_lstm = tf.Variable(tf.truncated_normal([n_units, 16],stddev=0.1,seed=SEED, dtype=tf.float32))
            b_lstm = tf.Variable(tf.zeros([16], dtype=tf.float32))
            lstm_cell = rnn.BasicLSTMCell(n_units, reuse = tf.AUTO_REUSE)
            dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_rate, output_keep_prob=self.keep_rate, state_keep_prob=self.keep_rate)
            reshaped_layer = tf.reshape(relu_manual,[-1,num_historical_days, 16])
            outputs, states = tf.nn.dynamic_rnn(dropout_cell, reshaped_layer, dtype=tf.float32)
            output_lstm = tf.matmul(outputs[-1], w_lstm) + b_lstm      

            k1_gan = tf.Variable(tf.truncated_normal([1, 1,num_features_gan, 16],
                stddev=0.1,seed=SEED, dtype=tf.float32))
            b1_gan = tf.Variable(tf.zeros([16], dtype=tf.float32))
            conv_gan = tf.nn.conv2d(X_gan,k1_gan,strides=[1, 1, 1, 1],padding='SAME')
            relu_gan = tf.nn.relu(tf.nn.bias_add(conv_gan, b1_gan))
            if is_train:
                relu_gan = tf.nn.dropout(relu_gan, keep_prob = self.keep_prob)

            relu =output_lstm+relu_gan

            k2 = tf.Variable(tf.truncated_normal([3, 1, 16, 32],
                stddev=0.1,seed=SEED, dtype=tf.float32))
            b2 = tf.Variable(tf.zeros([32], dtype=tf.float32))
            conv = tf.nn.conv2d(relu, k2,strides=[1, 1, 1, 1],padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b2))
            if is_train:
                relu = tf.nn.dropout(relu, keep_prob = self.keep_prob)

            k3 = tf.Variable(tf.truncated_normal([3, 1, 32, 64],
                stddev=0.1,seed=SEED, dtype=tf.float32))
            b3 = tf.Variable(tf.zeros([64], dtype=tf.float32))
            conv = tf.nn.conv2d(relu, k3, strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b3))
            if is_train:
                relu = tf.nn.dropout(relu, keep_prob=self.keep_prob)

            flattened_convolution_size = int(relu.shape[1]) * int(relu.shape[2]) * int(relu.shape[3])
            flattened_convolution = features = tf.reshape(relu, [-1, flattened_convolution_size])
            if is_train:
                flattened_convolution =  tf.nn.dropout(flattened_convolution, keep_prob=self.keep_prob)

            W1 = tf.Variable(tf.truncated_normal([20*1*64, 32]))
            b4 = tf.Variable(tf.truncated_normal([32]))
            h1 = tf.nn.relu(tf.matmul(flattened_convolution, W1) + b4)
            W2 = tf.Variable(tf.truncated_normal([32, 2]))
            logits = tf.matmul(h1, W2)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.Y, 1), tf.argmax(logits, 1)), tf.float32))
            self.confusion_matrix = tf.confusion_matrix(tf.argmax(self.Y, 1), tf.argmax(logits, 1))
            tf.summary.scalar('accuracy', self.accuracy)
            theta = [k1, b1, k1_gan, b1_gan, w_lstm, b_lstm, k2, b2, k3, b3, W1, b4, W2]           

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=logits))
        self.l2_loss = (0.0001 * tf.add_n([tf.nn.l2_loss(t) for t in theta]) / len(theta))
        tf.summary.scalar('loss', self.loss)
        self.total_loss = self.loss + self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.total_loss)
        self.summary = tf.summary.merge_all()
```

## Training the model with Tensorflow

The BiGAN is trained on 96 Nasdaq stocks. The first inputs are from the concatenation of the original time series (including adjusted open, low, high, close price and volume) normalized using a 20-day rolling window and the additional features. Sequences of 20 days are randomly picked from this dataset (30% of the dataset is excluded in order to test the model). The second inputs are the BiGAN's representation of 20 days sequences picked from the original time series and indicators. 

```python
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import confusion_matrix

SEED = 42
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def compute_features(num_historical_days, df_all = df_stocks, days=1, pct_change=0):
    
    data_gan = []
    data = []
    labels = []
    test_data = []
    test_data_gan = []
    test_labels = []
    cnn = CNN(num_features=41, num_historical_days=num_historical_days,  num_features_gan = 392, is_train=False)
    gan = BiGAN(num_features=41, num_historical_days=num_historical_days,
                    generator_input_size=200)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        with open('./bigan_models/checkpoint', 'rb') as f:
            model_name = next(f).split(b'"')[1]
        saver = tf.train.import_meta_graph("./bigan_models/{}.meta".format(model_name.decode("utf-8")))
        saver.restore(sess, "./bigan_models/{}".format(model_name.decode("utf-8") ))
        stocks = set(df_all.ticker)
        
        for stock in list(stocks)[:10]: 
            
            df = df_all[df_all.ticker == stock][['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
            df.sort_index(inplace=True)
            labels_hist = df.adj_close.pct_change(days).map(lambda x: [int(x > pct_change/100.0), int(x <= pct_change/100.0)])
            df = ((df -
            df.rolling(num_historical_days).mean().shift(-num_historical_days))
            /(df.rolling(num_historical_days).max().shift(-num_historical_days)
            -df.rolling(num_historical_days).min().shift(-num_historical_days)))
            df['labels'] = labels_hist
            df_features = df_all[df_all.ticker == stock][['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']].reset_index()
            add_features= compute_manual_features(df_features)
            add_features = add_features.drop(['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'], axis = 1).set_index('date')
            df = pd.concat( [df, add_features], axis=1)
            n_columns = df.shape[1]
            df = df.dropna()
            test_df = df[:365]
            df = df[400:]
            data_hist = df.drop(['labels'], axis = 1).values
            labels_hist = df['labels'].values
            for i in range(num_historical_days, len(df), num_historical_days):
                features = sess.run(gan.features, feed_dict={gan.X:[data_hist[i-num_historical_days:i]], gan.keep_prob:0.8})
                data_gan.append(features[0])
                data.append(data_hist[i-num_historical_days:i])
                labels.append(labels_hist[i-1])
            data_hist_test = test_df.drop(['labels'], axis = 1).values
            labels_hist_test = test_df['labels'].values
            for i in range(num_historical_days, len(test_df), 1):
                features = sess.run(gan.features, feed_dict={gan.X:[data_hist_test[i-num_historical_days:i]], gan.keep_prob:1})
                test_data_gan.append(features[0])
                test_data.append(data_hist_test[i-num_historical_days:i])
                test_labels.append(labels_hist_test[i-1])
    return data_gan, data, labels, test_data, test_data_gan, test_labels


class TrainCNN:

    def __init__(self, data_gan, data, labels, test_data, test_data_gan, test_labels, num_historical_days, df = df_stocks, days=1, pct_change=0):
        self.data_gan = data_gan
        self.data = data
        self.df = df
        self.labels = labels
        self.test_data = test_data
        self.test_data_gan = test_data_gan
        self.test_labels = test_labels
        self.cnn = CNN(num_features=41, num_historical_days=num_historical_days,  num_features_gan = 392, is_train=False)
        self.gan = BiGAN(num_features=41, num_historical_days=num_historical_days,
                        generator_input_size=200)
        
    def random_batch(self, batch_size=128):
        batch = []
        batch_gan = []
        labels = []
        length = len(self.data)
        data = list(zip(self.data, self.data_gan, self.labels))
        i = 0
        while True:
            i+= 1
            while True:
                d = list(data)[np.random.randint(0,length)]
                if(d[2][0]== int(i%2)):
                    break
            batch.append(d[0])
            batch_gan.append(d[1])
            labels.append(d[2])
            if (len(batch) == batch_size):
                yield batch, batch_gan, labels
                batch = []
                batch_gan = []
                labels = []

    def train(self, restore_mode, print_steps=100, display_steps=100, save_steps=1000, batch_size=128, keep_prob=0.6):
        
        if not os.path.exists('./cnn_models'):
            os.makedirs('./cnn_models')
        
        if restore_mode: 
            
            with tf.Session() as sess:
                
                total_loss = 0
                accuracy = 0
                sess.run(tf.global_variables_initializer())
                graph = tf.get_default_graph()
                if os.path.exists('./cnn_models/checkpoint'):
                    with open('./cnn_models/checkpoint', 'rb') as f:
                        model_name = next(f).split(b'"')[1]
                else:
                    print("No checkpoint found")
                saver = tf.train.import_meta_graph("./models/{}.meta".format(model_name.decode("utf-8")))
                saver.restore(sess, "./models/{}".format(model_name.decode("utf-8") ))
                step = int(re.search("(?<=-).*", model_name.decode("utf-8")).group(0))
                
                for i, [X, X_gan, y] in enumerate(self.random_batch(batch_size)):
                    
                    _, total_loss_curr, accuracy_curr = sess.run([self.cnn.optimizer, self.cnn.total_loss, self.cnn.accuracy], feed_dict=
                            {self.cnn.X:X, self.cnn.X_gan:X_gan, self.cnn.Y:y, self.cnn.keep_prob:keep_prob})
                    
                    total_loss += ttloss_curr
                    accuracy += accuracy_curr
                    
                    if (i+1) % print_steps == 0:
                        print('Step={} loss={}, accuracy={}'.format(i, total_loss/print_steps, accuracy/print_steps))
                        total_loss = 0
                        accuracy = 0
                        test_loss, test_accuracy, confusion_matrix = sess.run([self.cnn.total_loss, self.cnn.accuracy, self.cnn.confusion_matrix], feed_dict={self.cnn.X:self.test_data, self.cnn.X_gan:self.test_data_gan, self.cnn.Y:self.test_labels, self.cnn.keep_prob:1})
                        print("Test loss = {}, Test accuracy = {}".format(test_loss, test_accuracy))
                        print(confusion_matrix)
                    
                    if (i+step+1) % save_steps == 0:
                        saver.save(sess, './cnn_models/cnn.ckpt', step+i)

                    if (i+1) % display_steps == 0:
                        summary = sess.run(self.cnn.summary, feed_dict=
                            {self.cnn.X:X,self.cnn.X_gan:X_gan,  self.cnn.Y:y, self.cnn.keep_prob:keep_prob})
                        summary = sess.run(self.cnn.summary, feed_dict={
                            self.cnn.X:self.test_data, self.cnn.X_gan : self.test_data_gan, self.cnn.Y:self.test_labels, self.cnn.keep_prob:1})
                                   
        else: 

            sess = tf.InteractiveSession()
            total_loss = 0
            accuracy = 0
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            
            for i, [X, X_gan, y] in enumerate(self.random_batch(batch_size)):
                
                _, total_loss_curr, accuracy_curr = sess.run([self.cnn.optimizer, self.cnn.total_loss, self.cnn.accuracy], feed_dict=
                        {self.cnn.X:X, self.cnn.X_gan:X_gan, self.cnn.Y:y, self.cnn.keep_prob:keep_prob})
                total_loss += total_loss_curr
                accuracy += accuracy_curr
                
                if (i+1) % print_steps == 0:
                    print('Step={} loss={}, accuracy={}'.format(i, total_loss/print_steps, accuracy/print_steps))
                    total_loss = 0
                    accuracy = 0
                    test_loss, test_accuracy, confusion_matrix = sess.run([self.cnn.total_loss, self.cnn.accuracy, self.cnn.confusion_matrix], feed_dict={self.cnn.X:self.test_data, self.cnn.X_gan:self.test_data_gan, self.cnn.Y:self.test_labels, self.cnn.keep_prob:1})
                    print("Test loss = {}, Test accuracy = {}".format(test_loss, test_accuracy))
                    print(confusion_matrix)
                
                if (i+1) % save_steps == 0:
                    saver.save(sess, './cnn_models/cnn.ckpt', i)
            sess.close()

if __name__ == '__main__':

    data_gan, data, labels, test_data, test_data_gan, test_labels = compute_features(num_historical_days = 20, df_all = df_stocks, days=1, pct_change=0)
    tf.keras.backend.clear_session()
    cnn = TrainCNN(data_gan, data, labels, test_data, test_data_gan, test_labels, num_historical_days=20, days=1, pct_change=0)
    cnn.train(restore_mode= True)
    
```

> **Conclusion** : in this fourth article about stock market prediction, we have presented a multi-input deep learning classifier using CNNs and LSTM in order to predict whether a stock price will go up or down on the next day considering time-series of historical prices/volumes, technical indicators and features of 20-day period, as well as a representation of this data using a BiGAN.