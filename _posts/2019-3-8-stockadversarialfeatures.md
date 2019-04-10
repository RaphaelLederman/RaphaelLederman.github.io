---
published: true
title: Unsupervised Feature Extraction with a custom Bidirectional GAN
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

In this third article related to stock market predictions, we will go through an unsupervised learning methodology used in order to generate features. The main purpose of this approach is to train a custom Bidirectional Generative Adversarial Network to represent data in an unsupervised fashion through a competitive learning process so that the resulting features are not overfitted to the training set. Classification algorithms trained on these features will generalize on a smaller amount of data as the learning process for GANs results in more of the possible feature space being explored. In other words, we are going to use the fact that GANs promote generalization beyond the training set in order to construct a rich and relevant representation of our time series. By way of introduction, here is a [tutorial](https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/) for implementing a basic GAN. For more detailed information on GANs, have a look at the reference [paper](https://arxiv.org/pdf/1406.2661.pdf).

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## A bit of theory

GANs are based on two components : the generative model and the discriminative model.
The generative model, which acts like a counterfeiter, trying to produce fake time series, is pitted against the discriminative model, which acts as the police, trying to detect the fake time series by determining whether a sample is from the model distribution or the data distribution (thereby distinguishing random from real variability). Competition in this adversarial context drives both models to improve their scores until the generated data is indistinguishable from the original data.

To learn the generator’s distribution $$p_{g}$$ over data $$x$$, we define a prior on
input noise variables $$p_{z}(z)$$, then represent a mapping to data space as $$G(z; \theta_{g})$$, where G is a differentiable function represented by a multilayer perceptron with parameters  $$\theta_{g}$$. We also define a second model $$D(x; \theta_{d})$$ that outputs a single scalar. $$D(x)$$ represents the probability
that $$x$$ came from the data rather than $$p_{g}$$. We train D to maximize the probability of assigning the correct label to both training examples and samples from G. 

We simultaneously train G to minimize $$log(1 − D(G(z)))$$. In other words, D and G play the following two-player minimax game with value function $$V(G, D)$$:

$$\underset{G}{\text{min}} {\underset{G}{\;\text{max}}} V(G,D) = \mathbb{E}_{x\sim p_{data}(x)}[log D(x)]+\mathbb{E}_{x\sim p_{z}(z)}[log (1-D(G(z)))]$$

![image](https://raphaellederman.github.io/assets/images/gan_train.png){:height="100%" width="200%"}

Generative adversarial nets are trained by simultaneously updating the discriminative distribution ($$D$$, blue, dashed line) so that it discriminates between samples from the data generating distribution (black, dotted line) $$p_{x}$$ from those of the generative distribution $$p_{g}$$ ($$G$$) (green, solid line). The lower horizontal line is the domain from which $$z$$ is sampled, in this case uniformly. The horizontal line above is part of the domain of $$x$$. The upward arrows show how the mapping $$x = G(z)$$ imposes the non-uniform distribution $$p_{g}$$ on
transformed samples. G contracts in regions of high density and expands in regions of low density of $$p_{g}$$.

* (a) Consider an adversarial pair near convergence: $$p_{g}$$ is similar to $$p_{data}$$ and $$D$$ is a partially accurate classifier.
* (b) In the inner loop of the algorithm D is trained to discriminate samples from data, converging to $$D^{* }(x) = \frac{p_{data}(x)}{p_{data}(x)+p_{g}(x)}$$ 
* (c) After an update to $$G$$, gradient of $$D$$ has guided $$G(z)$$ to flow to regions that are more likely to be classified as data. 
* (d) After several steps of training, if $$G$$ and $$D$$ have enough capacity, they will reach a point at which both cannot improve because $$p_{g} = p_{data}$$. The discriminator is unable to differentiate between the two distributions, i.e. $$D(x) = \frac{1}{2}$$.

## Choice of Generative and Discriminative models for features extraction

In the reference paper written by Ian J. Goodfellow $$\textit{et al.}$$, the generative model generates samples by passing random noise through a multilayer perceptron, and the discriminative model is also a multilayer perceptron. Both models are then trained using backpropagation and dropout algorithms.

In our case, we choose to keep a multilayer perceptron as the generative model, but instead of sampling $$z$$ from a uniform distribution, we sample it from a standard normal distribution. In order to extract features that represent the data in the richest way possible, we use a Convolutional Neural Network as the discriminative model. With CNNs, the learned representation of the data often outperforms expert features for many modalities including radio frequency, computer vision and audio classification. We used 2D convolutions (which implies convolving both horizontal and vertical directions in two-dimensional spatial domain, the impulse functions and responses also being in 2D space), and the Rectified Linear Unit (or ReLU) activation function ($$A(x) = max(0,x)$$). With this type of architecture, the activations from the last layer of convolutional lays can be used as the new data representation.

Other choices of generative and discriminative models are also valid and have proven to yield promising results, for instance Recurrent Neural Networks and especially Long Short Term Memory cells can be used as a generator for structured data.

## Extension of the model : Custom Bidirectional Generative Adversarial Networks

In order to improve further the unsupervised features learning, we chose to implement a custom version of Bidirectional Generative Adversarial Networks. Have a look at this [paper](https://arxiv.org/pdf/1605.09782.pdf) if you want more detailed information on the theoretical foundations of BiGANs, and more precisely on both the usefulness of the resulting learned feature representation for auxiliary supervised discrimination tasks, and their competitiveness with contemporary approaches to unsupervised and self-supervised feature learning.

The main difference between a GAN and a BiGAN is that the BiGAN's discriminative model learns to determine the joint probability $$P(x, z) = real$$ (where x is the sample and z is the generating distribution). This, in turn, means that the generative model learns to encode a real sample into its generating distribution. Once the BiGAN is finished training, both the encoding of the real sample into the generating distribution and the activations from the last layer of convolutional lays can be used as the representation of the data.

![image](https://raphaellederman.github.io/assets/images/bigan.png){:height="100%" width="200%"}

We chose to make a modification to the BiGAN : rather than learning to encode a real sample into its generating distribution, our model learns to encode the features learned by the discriminative model (the activation from the last layer of convolutional lays) into the generating distribution. The concatenation of both the activation from the last layer of convolutional lays and its encoding into the generating distribution can be used as the new representation of the data. We added an L2 regularization term to the model's loss, and used the ADAM (Adaptive Moment Estimation) optimizer. This architecture outperformed the regular BiGAN architecture for our dataset.

In the following block, you will find my code for the definition of the custom BiGAN.

```python
import os
import numpy as np
import tensorflow as tf

SEED = 42
tf.set_random_seed(SEED)
np.random.seed(SEED)

class BiGAN():

    def sample_Z(self, batch_size, n):

        return np.random.normal(loc=0.0, scale=1.0, size=(batch_size, n))

    def __init__(self, num_features, num_historical_days, generator_input_size=200):

        self.X = tf.placeholder(tf.float32, shape=[None, num_historical_days, num_features])
        X = tf.reshape(self.X, [-1, num_historical_days, 1, num_features])
        self.Z = tf.placeholder(tf.float32, shape=[None, generator_input_size])
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        generator_output_size = num_features*num_historical_days

        with tf.variable_scope("generator"):

            W1 = tf.Variable(tf.truncated_normal([generator_input_size, 128]))
            b1 = tf.Variable(tf.truncated_normal([128]))

            h1 = tf.nn.sigmoid(tf.matmul(self.Z, W1) + b1)
            h1 = tf.nn.dropout(h1, keep_prob=self.keep_prob)

            W2 = tf.Variable(tf.truncated_normal([128, 256]))
            b2 = tf.Variable(tf.truncated_normal([256]))

            h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)
            h2 = tf.nn.dropout(h2, keep_prob=self.keep_prob)

            W3 = tf.Variable(tf.truncated_normal([256, generator_output_size]))
            b3 = tf.Variable(tf.truncated_normal([generator_output_size]))

            g_out = tf.matmul(h2, W3) + b3
            g_out = tf.reshape(g_out, [-1, num_historical_days, 1, num_features])
            self.gen_data = tf.reshape(g_out, [-1, num_historical_days, num_features])

            theta_G = [W1, b1, W2, b2, W3, b3]

        with tf.variable_scope("discriminator"):

            k1 = tf.Variable(tf.truncated_normal([3, 1, num_features, 16],
                stddev=0.1, dtype=tf.float32))
            b1 = tf.Variable(tf.zeros([16], dtype=tf.float32))

            k2 = tf.Variable(tf.truncated_normal([5, 1, 16, 32],
                stddev=0.1, dtype=tf.float32))
            b2 = tf.Variable(tf.zeros([32], dtype=tf.float32))

            k3 = tf.Variable(tf.truncated_normal([5, 1, 32, 64],
                stddev=0.1, dtype=tf.float32))
            b3 = tf.Variable(tf.zeros([64], dtype=tf.float32))

            W1 = tf.Variable(tf.truncated_normal([3*1*64 + generator_input_size, 64]))
            b4 = tf.Variable(tf.truncated_normal([64]))

            W2 = tf.Variable(tf.truncated_normal([64, 1]))

            theta_D = [k1, b1, k2, b2, k3, b3, W1, b4, W2]


        def discriminator_conv(X):

            conv = tf.nn.conv2d(X,k1,strides=[1, 2, 1, 1],padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b1))
            relu = tf.nn.dropout(relu, self.keep_prob)

            conv = tf.nn.conv2d(relu, k2,strides=[1, 2, 1, 1],padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b2))
            relu = tf.nn.dropout(relu, self.keep_prob)

            conv = tf.nn.conv2d(relu, k3, strides=[1, 2, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, b3))
            relu = tf.nn.dropout(relu, self.keep_prob)

            flattened_convolution_size = int(relu.shape[1]) * int(relu.shape[2]) * int(relu.shape[3])
            flattened_convolution = tf.reshape(relu, [-1, flattened_convolution_size])
            
            return flattened_convolution
        
        
        def discriminator(X, Z):

            flattened_convolution = discriminator_conv(X)
            flattened_convolution = features = tf.concat([flattened_convolution, Z], axis=1)
            flattened_convolution = tf.nn.dropout(flattened_convolution, self.keep_prob)

            h1 = tf.nn.relu(tf.matmul(flattened_convolution, W1) + b4)

            D_logit = tf.matmul(h1, W2)
            D_prob = tf.nn.sigmoid(D_logit)
            
            return D_prob, D_logit, features

        with tf.variable_scope("encoder"):

            flattened_convolution = discriminator_conv(X)
            encoder_input = tf.concat([flattened_convolution, tf.reshape(self.X, [-1, num_features*num_historical_days])], axis=1)
            
            e_h1_size = 128
            e_W1 = tf.Variable(tf.truncated_normal([int(flattened_convolution.shape[1]), e_h1_size]))
            e_b1 = tf.Variable(tf.truncated_normal([e_h1_size]))
            e_h1 = tf.nn.relu(tf.matmul(flattened_convolution, e_W1) + e_b1)
            e_h1 = tf.nn.dropout(e_h1, keep_prob=self.keep_prob)

            e_h2_size = 64
            e_W2 = tf.Variable(tf.truncated_normal([e_h1_size, e_h2_size]))
            e_b2 = tf.Variable(tf.truncated_normal([e_h2_size]))
            e_h2 = tf.nn.relu(tf.matmul(e_h1, e_W2) + e_b2)
            e_h2 = tf.nn.dropout(e_h2, keep_prob=self.keep_prob)


            e_h3_size = 32
            e_W3 = tf.Variable(tf.truncated_normal([e_h2_size, e_h3_size]))
            e_b3 = tf.Variable(tf.truncated_normal([e_h3_size]))
            e_h3 = tf.nn.tanh(tf.matmul(e_h2, e_W3) + e_b3)


            e_W4 = tf.Variable(tf.truncated_normal([e_h3_size, generator_input_size]))
            e_b4 = tf.Variable(tf.truncated_normal([generator_input_size]))

            self.encoding = tf.nn.sigmoid(tf.matmul(e_h3, e_W4) + e_b4)
            theta_E = [e_b1, e_b2, e_b3, e_b4, e_W1, e_W2, e_W3, e_W4]

        D_real, D_logit_real, self.features = discriminator(X, self.encoding)
        D_fake, D_logit_fake, _ = discriminator(g_out, self.Z)

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        
        self.D_l2_loss = (0.0001 * tf.add_n([tf.nn.l2_loss(t) for t in theta_D]) / len(theta_D))
        self.D_loss = D_loss_real + D_loss_fake + self.D_l2_loss
        self.G_l2_loss = (0.00001 * tf.add_n([tf.nn.l2_loss(t) for t in theta_G]) / len(theta_G))
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))+self.G_l2_loss

        self.D_solver = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.D_loss, var_list=theta_D)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.G_loss, var_list=theta_G + theta_E)
        
        self.D_real_mean = tf.reduce_mean(D_real)
        self.D_fake_mean = tf.reduce_mean(D_fake)
```

## Training the model with Tensorflow

The BiGAN is trained on 96 Nasdaq stocks. Original time series (including adjusted open price, low price, high price, close price and volume) are normalized using a 20-day rolling window (data-mean)/(max-min). Additional features (see first article) are then computed based on the original series, and the whole dataset (original time series and additional features) is used as input to the BiGAN.

```python
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class TrainBiGan:

    def __init__(self, num_historical_days, dataframe = df_stocks, batch_size = 128, generator_input_size=50):
        
        self.data = []
        self.df = df_stocks
        self.batch_size = batch_size
        self.generator_input_size = generator_input_size
        stocks = list(set(self.df.ticker))
        
        for stock in stocks: 

            df = self.df[self.df.ticker == stock][['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
            df.sort_index(inplace=True)
            df = ((df -
            df.rolling(num_historical_days).mean().shift(-num_historical_days))
            /(df.rolling(num_historical_days).max().shift(-num_historical_days)
            -df.rolling(num_historical_days).min().shift(-num_historical_days)))
            
            df_features = self.df[self.df.ticker == stock][['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']].reset_index()
            add_features= compute_manual_features(df_features)
            add_features = add_features.drop(['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume'], axis = 1).set_index('date')
            
            df = pd.concat([df, add_features], axis=1)
            n_columns = df.shape[1]
            df = df.dropna()

            for i in range(num_historical_days, len(df), num_historical_days):

                self.data.append(df.values[i-num_historical_days:i])
        
        self.gan = BiGAN(num_features=n_columns, num_historical_days=num_historical_days,
                        generator_input_size=self.generator_input_size)

    def random_batch(self, batch_size=128):
        
        batch = []
        while True:
            batch.append(random.choice(self.data))
            if (len(batch) == batch_size):
                yield batch
                batch = []
              
                
    def train(self,restore_mode, print_steps=100, display_data=100, save_steps=100):
        
        if not os.path.exists('../bigan_models'):
            os.makedirs('../bigan_models')
        
        if restore_mode: 
            with tf.Session() as sess:
                G_loss = 0
                D_loss = 0
                D_last_loss = 0
                G_last_loss = 0
                D_real = 0
                D_fake = 0
                D_real_curr = 0
                D_fake_curr = 0
                D_l2_loss = 0
                G_l2_loss = 0
                sess.run(tf.global_variables_initializer())
                graph = tf.get_default_graph()
                if os.path.exists('./bigan_models/checkpoint'):
                    with open('./bigan_models/checkpoint', 'rb') as f:
                        model_name = next(f).split(b'"')[1]
                else:
                    print("No checkpoint found")
                saver = tf.train.import_meta_graph("./models/{}.meta".format(model_name.decode("utf-8")))
                saver.restore(sess, "./models/{}".format(model_name.decode("utf-8") ))
                step = int(re.search("(?<=-).*", model_name.decode("utf-8")).group(0))
                for i, X in enumerate(self.random_batch(self.batch_size)):

                    _, D_loss_curr, D_l2_loss_curr, D_real_curr = sess.run([self.gan.D_solver, self.gan.D_loss, self.gan.D_l2_loss, self.gan.D_real_mean], feed_dict=
                                {self.gan.X:X, self.gan.keep_prob:1.0, self.gan.Z:self.gan.sample_Z(self.batch_size, self.generator_input_size)})

                    D_loss += D_loss_curr
                    D_l2_loss += D_l2_loss_curr
                    D_last_loss = D_loss_curr 

                    D_real += D_real_curr
  
                    _, G_loss_curr, G_l2_loss_curr, D_fake_curr = sess.run([self.gan.G_solver, self.gan.G_loss,self.gan.G_l2_loss, self.gan.D_fake_mean], feed_dict=
                        {self.gan.keep_prob:1.0, self.gan.Z:self.gan.sample_Z(self.batch_size, self.generator_input_size)})

                    G_loss += G_loss_curr
                    G_l2_loss += G_l2_loss_curr
                    G_last_loss = G_loss_curr
                
                    D_fake += D_fake_curr

                    if (i+1) % print_steps == 0:
                        print('Step={} D_loss={}, G_loss={}, D_real={}, D_fake={}'.format(i, D_loss/print_steps, G_loss/print_steps, D_real/print_steps, D_fake/print_steps))   
                        G_loss = 0
                        D_loss = 0
                        D_real = 0
                        D_fake = 0

                    if (i+step+1) % save_steps == 0:
                        save_path = saver.save(sess, './bigan_models/bigan.ckpt', i+step)
            
        else: 
             with tf.Session() as sess:
                G_loss = 0
                D_loss = 0
                D_last_loss = 0
                G_last_loss = 0
                D_real = 0
                D_fake = 0
                D_real_curr = 0
                D_fake_curr = 0
                D_l2_loss = 0
                G_l2_loss = 0
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                for i, X in enumerate(self.random_batch()):

                    _, D_loss_curr, D_l2_loss_curr, D_real_curr = sess.run([self.gan.D_solver, self.gan.D_loss, self.gan.D_l2_loss, self.gan.D_real_mean], feed_dict=
                                {self.gan.X:X, self.gan.keep_prob:1.0, self.gan.Z:self.gan.sample_Z(self.batch_size, self.generator_input_size)})

                    D_loss += D_loss_curr
                    D_l2_loss += D_l2_loss_curr
                    D_last_loss = D_loss_curr 

                    D_real += D_real_curr

                    _, G_loss_curr, G_l2_loss_curr, D_fake_curr = sess.run([self.gan.G_solver, self.gan.G_loss,self.gan.G_l2_loss, self.gan.D_fake_mean], feed_dict=
                        {self.gan.keep_prob:1.0, self.gan.Z:self.gan.sample_Z(self.batch_size, self.generator_input_size)})

                    G_loss += G_loss_curr
                    G_l2_loss += G_l2_loss_curr
                    G_last_loss = G_loss_curr
                
                    D_fake += D_fake_curr

                    if (i+1) % print_steps == 0:
                        print('Step={} D_loss={}, G_loss={}, D_real={}, D_fake={}'.format(i, D_loss/print_steps, G_loss/print_steps, D_real/print_steps, D_fake/print_steps))   
                        G_loss = 0
                        D_loss = 0
                        D_real = 0
                        D_fake = 0

                    if (i+1) % save_steps == 0:
                        save_path = saver.save(sess, './bigan_models/bigan.ckpt', i)

if __name__ == '__main__':
    bigan = TrainBiGan(num_historical_days=20, batch_size=32, generator_input_size=200)
    bigan.train(restore_mode = False)
```

> **Conclusion** : in this third article about stock market prediction, we have presented an efficient unsupervised features extration methodology using a custom Bidirectional Generative Adversarial Networks. In the next article, we will show a deep learning classification algorithm used to predict positive or negative returns based on the original time series, the manual features and the extracted representation.