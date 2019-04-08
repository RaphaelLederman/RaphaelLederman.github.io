---
published: true
title: Embedding
collection: articles
layout: single
author_profile: false
read_time: true
categories: [articles]
excerpt : "Natural Language Processing"
header :
    overlay_image: "https://raphaellederman.github.io/assets/images/night.jpg"
    teaser_image: "https://raphaellederman.github.io/assets/images/night.jpg"
toc: true
toc_sticky: true
---

In this article, we will describe two of the main approaches to word embedding : bag-of-word methods and Word2Vec embedding.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Bag-of-Word approaches

In order to run machine learning algorithms we need to convert the text files into numerical feature vectors : we convert a collection of text documents to a matrix of token counts, the number of features being equal to the vocabulary size found by analyzing the data (each unique word in our dictionary corresponding to a descriptive feature). The easiest and simplest way of counting this tokens is to use raw counts (term frequencies).

$$  tf_{t,d}=  f_{t,d} $$

Instead of using the raw frequencies of occurrence of tokens in a given document it is possible to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus. In order to do this, we can use term frequencies adjusted for document length, also called TF-IDF
(or term frequency times inverse document frequency). Common words like (a,
an, the, etc.) will then have a lower weight.

$$ tf_{t,d}=  \sum\limits_{t'\in d} f_{t',d} $$

This choice of embedding using a document-term matrix has some disadvantages. As with most embedding methodologies, it requires to choose the vocabulary size (which is a parameter that will greatly impact the sparsity of the document representations) and the document-term matrix can end-up being very sparse. The $$\textit{a priori}$$ choice of vocabulary is not the problem $$\textit{per se}$$, but it is this scarcity, intrinsically linked to the structure of the word representation in bag-of-words approaches, that is the biggest disadvantage of this method, both from the point of view of computational efficiency and information retrieval (as there is little information in a large representation space). Finally, there can be a loss of valuable information through discarding word order.

## Word2Vec Embedding

The Word2Vec embedding was first proposed by Mikolov et al. in “Efficient Estimation of Word Representations in Vector Space” (2013). It generates distributed representations by assigning a real-valued vector for each word and representing the word by the vector : we call the vector $$\textit{word embedding}$$. The idea is to introduce dependency between words : words with similar context should occupy close spatial positions. This is very different from the document-term matrix where all words were considered independent from each others. 
The Word2Vec method constructs the embedding using two methods in the context of neural networks: Skip Gram and Common Bag Of Words (CBOW). Both architectures can be used in order to produce embeddings.

Using one-hot encoding and considering the context of each word, the goal of the CBOW methodology is to predict the word corresponding to the context. For a single input word, for instance the word "sunny" in the sentence "What a sunny weather today!", the objective is to predict the one-hot encoding of the target word "weather" and minimize the output error between the predicted vector and the target one. 

The vector representation of the target word is then learned in the prediction process. More precisely, the neural network first takes the V-dimensional one-hot encoding of the input word and maps it to the hidden layer using a first weight matrix. Then, another weight matrix is used to map the hidden layer outputs to the final V-dimensional prediction vector constructed with the softmax values. 

It is important to note that there is no use of non-linear activation functions (tanh, sigmoid, ReLu etc.) outside of the softmax calculations in the last layer: the outputs are passed as simple weighted combination of the inputs. More precisely : 

$$Output\;from\;hidden\;node\; 'j' = u_j = \sum\limits_{i = 1}^{V}{w_{i,j}x_i}$$

With $$u_j$$ being the input to the j-th hidden node, $$w_ij$$ is the weight of the connection from the i-th input node to the j-th hidden node and $x_i$ is the value of the i-th input node (in the case of word vectors, only one element of $x_i$ is equal to 1, remaining all are 0. The output layer has V output nodes, one for each unique word (V corresponds here to the vocabulary size). The final output from each node is the softmax.

$$Value\;at\;output\;node\;'k' = O_k = \frac{exp(u^{\prime}_k)}{\sum\limits_{q=1}^{V}exp(u_q^{\prime})}$$


where $$u_k^{\prime}=\sum\limits_{j=1}^{N}w_{jk}^{\prime}h_j$$

$$u_k^{\prime}$$ is the input to the k-th output node, $$w_{jk}^{\prime}$$ is the weight of the connection from j-th hidden node to the k-th output node and $$h_j$$ is the output of the j-th hidden node. The cross-entropy function is used to compute the log-loss at each output node 'k':

$$Loss\;at\;output\;node\;'k' = {-y_k}\times log(O_k)-(1-y_k) \times log(1-O_k)$$

where $$y_k=1$$ if the actual output is the word at the k-th index else $$y_k=0$$.
The cross-entropy function is used to compute the log-loss at each output node 'k':

$$E=\sum\limits_{k=1}^{V}[-y_k\times log(O_k)-(1-y_k)\times log(1-O_k)]$$

The weights $$w_{jk}^{\prime}$$ between hidden layer and the output layer are progressively updated using stochastic gradient descent technique as follows (the second equation being obtained by solving for the partial derivative in the first equation) : 

$$w^{\prime(t+1)}_{jk}=w^{\prime(t)}_{jk} - \eta \times [\frac{\partial E}{\partial w^{\prime}_{jk}}] = w^{\prime(t)}_{jk}- \eta \times (O_k - y_k) \times h_j$$
 
where $$\eta$$ is the learning rate.

In the previous equation, $$(O_k−y_k)$$ denotes the error in prediction at the k-th node. Similarly, using back-propagation, it is possible to obtain the update equations for the input weights $$w_{ij}$$. Since only one of the inputs is active at each training iteration, we need to update the input weights $$w_{ij}$$ only for the node 'i' (the one for which the input  $$x_i=1$$) as follows: 
The above update equation is applied only for the node 'i', for which the input  $$x_i=1$$

$$w^{(t+1)}_{jk} =  w^{(t)}_{jk}- \eta \times \sum \limits_{k=1}^{V}(O_k - y_k)\times w_{jk}^{\prime(t+1)}$$

This model can be extended to non-single context words : it is possible to use multiple input context vectors, or a combination of them (sum or mean for instance) in order to improve predictions. Indeed, if we define a context size of 2, we will consider a maximum of 2 words on the left and 2 words on the right as the surrounding words for each target word.

For the sentence "read books instead of playing video games", the context words for "playing" with context size of 2 would be :

("instead", "of", "video", "games")

Using the CBOW methodology, if the input is ("instead", "of", "video", "games"), then desired output for this precise example is ("playing"). In the case where we have multiple input context words, it is assumed that the same set of weights $$w_ij$$ between input nodes and hidden nodes are used. This leads to computing the output of the hidden node $$j$$ as the average of the weighted inputs for all the context words. 

Assuming we have C number of context words, the output from hidden layer 'j' is given as :

$$h_j =  \frac{1}{C} \times [\sum \limits_{q=1}^{C}w_{i_{q}j}x_{i_{q}}]$$

where the inputs $$x_{i_{q}}$$ is the value of $$x_i$$ in the q-th input layer. The update equation for the output weights $$w_{jk}^{\prime}$$ remains the same as above. The update equations for the input weights $$w_{ij}$$ is modified to be :

$$w^{(t+1)}_{i_{q}j} =  w^{(t)}_{i_{q}j} - \eta \times \frac{1}{C} \times \sum \limits_{k=1}^{V}(O_k - y_k)\times w_{jk}^{\prime(t+1)}$$

The concept behind the Skip Gram architecture is the exact opposite of the CBOW architecture : taking a word vector as input, the objective is to predict the surrounding words (the number of context words to predict being a parameter of the model). In the context of our previous example sentence with context size of 2, if the input is ("playing") then the desired output would be ("instead", "of", "video", "games"). For the Skip Gram architecture, we therefore assume that we have multiple output layers and all output layers share the same set of output weights $$w_{jk}^{\prime}$$. The update equations for the output weights are modified as follows :

$$w^{(t+1)}_{jk} =  w^{(t)}_{jk} - \eta \times h_j \times \sum \limits_{k_{q}}(O_{k_{q}} - y_{k_{q}})$$

where $$k_q$$ denotes the k-th output node in the q-th output layer. The term $$\sum \limits_{k_{q}}(O_{k_{q}} - y_{k_{q}})$$ represents the sum of all the errors from C different context words predictions. In this way, the model is penalized for each context word mis-prediction. The update equation for the input weights remains unchanged.

According to various experimentations, it seems that the Skip Gram architecture performs slightly better than the CBOW architecture at constructing word embeddings : this might be linked to the fact that the varying impacts of different input context vectors are averaged out in the CBOW methodology. For words which co-occur together many times in the text corpus, the model is more likely to predict one of them at the output layer when the other one is given as the input. 
The first one is the Hierarchical Softmax and the second one is Negative Sampling.

With Hierarchical Softmax, it is possible to reduce the complexity of the output probabilities computation for each output node, from $$O(V)$$ to $$O(log(V))$$. In HS, the output layer is arranged in the form of a binary tree, with the leaves being the output nodes, thus there are V leaf nodes in the tree (and V-1 internal nodes). Therefore there are no output weights with Hierarchical Softmax, but instead we need to update weights for each internal node (each internal node having a weight associated with it). The probability of a particular output node (leaf node in this case) is given by the product of the internal node probabilities on the path from the root to this leaf node. There is no need to normalize this probabilities as the sum of all the leaf node probabilities sum to 1.

Let n(k, j) for an internal node denote that it is the j-th node from the root to the word at index k in the vocabulary, along the path in the tree, then the output probabilities are given as :

$$O_k =  \prod_{j=1}^{L(k) - 1} F(k,j)$$

$$where\; F(k,j) =
    \begin{cases}
      \sigma(v(k,j)), & \textit{if n(k,j+1) is the left child of n(k,j)}\ \\
      \sigma( - v(k,j)), & \textit{otherwise}
    \end{cases}$$

where $$v(k,j)=\sum\limits^{N}_{i=1}w^{\prime}_{i,n(k,j)}h_i$$ and $$L(k)$$ denotes the length of the path from the root of the tree to the word at index k, $$w_{i,n(k,j)}^{\prime}$$ denotes the weight of the connection from the hidden node 'i' to the internal node n(k, j) in the tree. It can be shown that :

$$sum_{k=1}^{V}O_k =  1$$

Instead of using a balanced binary tree for Hierarchical Softmax, the authors have used Huffman Trees. In theses trees, words which are more frequent are placed closer to the root whereas words which are less frequent are placed farther from the root. This is done using the frequency of each output word from the training instances, and results in the weights of the internal nodes updating faster.

In the Negative Sampling approach, few words are sampled from the vocabulary and only these words are used to update the output weights. The words that are sampled (apart from the actual output word) should be ones that are less likely to be predicted given the input word, so that input word vector is most affected by the actual output words (and least affected by the output vectors of the remaining sampled words). The main objective being to maximize the similarity between the input word vector and the output context word vectors, it seems appropriate to discard some of the words in the vocabulary that do not contribute much but only increase time complexity. The negative samples are selected from the vocabulary list using their "unigram distribution", such that more frequent words are more likely to be negative samples.

One approach for converting the word vector representations into the document-term matrix is to take the sum (average, min/max etc.) of the word vectors of all the words present in the document and use this as the vector representation for the document. The authors of Word2Vec have also developed another version of their methodology called Doc2Vec for directly training sentences and paragraphs with the Skip Gram architecture instead of averaging the word vectors in the text. We tried a similar approach in order to improve the accuracy of our classification: instead of giving the whole list of token vectors to our classifier, we gave it instead an average vector based on the TF-IDF weights. This method, while not improving our accuracy, yielded satisfying results considering the magnitude of the dimension reduction and therefore the potential information loss.

Fortunately, we don't have t train our own CBOW and Skip Gram models in order to obtain word embeddings : it is possible to load pretrained embeddings and use them into our models. 
In order to load the Google Word2Vec embedding dictionnary and select the relevant items corresponding to a preprocessed corpus, it is possible to use a code similar to the following. Here, max_features represents the maximum number of unique words, max_sentence_len the length of each sequence, and embed_dim the embedding dimension of each word.

```python
def prepare_embedding(max_features, max_sentence_len, embed_dim, corpus):
    """
    Returns a word-vector dictionnary fitted on our text data : it contains only the max_features most important words in our corpus, and each vector has a dimension of embed_dim
    """
    # Load full Google Word2Vec dictionnary
    word2vec = load_google_vec()

    # Get the preprocessed corpus and the corresponding word index
    corpus_preprocessed, train_word_index = preprocess(X, max_features, max_sentence_len)
    print('Found %s unique tokens.' % len(train_word_index))

    # Create the embedding matrix
    train_embedding_weights = np.zeros((len(train_word_index)+1, embed_dim))

    # Populate the matrix with the relevant items from the Google Word2Vec dictionnary
    for word,index in train_word_index.items():
        train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(embed_dim)
    
    word_vector_dict = dict(zip(pd.Series(list(train_word_index.keys())),pd.Series(list(train_word_index.keys())).apply(lambda x: train_embedding_weights[train_word_index[x]])))
    
    return train_embedding_weights, train_word_index, word_vector_dict

def load_google_vec():
    """
    Returns the full Google Word2Vec dictionnary with 300-dimensionnal embeddings
    """
    return KeyedVectors.load_word2vec_format('/Users/raphaellederman/GoogleWord2Vec300.bin', binary=True)

```

This embedding dictionnary can then be used in a custom Keras embedding layer, which is generally the first layer of a deep learning architecture for instance in the context of documents classification (sentiment analysis, emotion recognition...). A potential architecture in Keras for a multi-label (5 in the following code) text emotion recognition could be as following (we will not provide more details on such models here as it is not the purpose of this short article):

```python
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv1D, MaxPooling1D, SpatialDropout1D, LSTM, Dense 

Input_words = Input(shape=(max_sentence_len,), name='input')
x = Embedding(len(train_word_index)+1,embed_dim,weights=[train_embedding_weights],input_length= max_sentence_len,trainable=True)(Input_words)
x = Conv1D(filters=50, kernel_size=8, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = SpatialDropout1D(0.2)(x)
x = BatchNormalization()(x)
x = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(x)
out = Dense(5,activation='softmax')(x)
classifier = Model(inputs=Input_words, outputs=[out])
print(classifier.summary())
```

