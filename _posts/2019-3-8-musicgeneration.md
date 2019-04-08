---
published: true
title: Generating Music Sequences using Deep Recurrent Neural Networks
collection: articles
layout: single
author_profile: false
read_time: true
categories: [articles]
excerpt : "Music"
header :
    overlay_image: "https://raphaellederman.github.io/assets/images/night.jpg"
    teaser_image: "https://raphaellederman.github.io/assets/images/night.jpg"
toc: true
toc_sticky: true
---

In this article, we will train a network to learn a language model and then use it to generate new sequences. We will train it to learn the language of the music of [Johann_Sebastian_Bach](https://en.wikipedia.org/wiki/Johann_Sebastian_Bach). For this, we will learn how J. S. Bach's "Cello suite" hav been composed. Here is an example of a "Cello suite" [Link](https://www.youtube.com/watch?v=mGQLXRTl3Z0). Rather than analyzing the audio signal, we use a symbolic representation of the "Cello suite" through their [MIDI files](https://en.wikipedia.org/wiki/MIDI#MIDI_files). A MIDI file encodes in a file, the set of musical notes, their duration, and intensity which have to be played by each instrument to "render" a musical piece. The "rendering" is usually operated by a MIDI synthesizer (such as VLC, QuickTime). We will first train a language model on the whole set of MIDI files of the "Cello suites", and will then sample this language model to create a new MIDI file which will be a brand new "Cello suite" composed by the computer.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Collecting data

First let's start with some imports. We also define the parameters of our model and collect the 36 MIDI files corresponding to the 36 "Cello suites" composed by J. S. Bach. The parameters are the following :
* n_x : number of possible different keys on a keyboard
* max_T_x : the maximum length that we are going to use for each suite (all suites originally having different lengths)
* sequence_length : the sequence length that we are going to use to define the inputs of our model
* T_y_generated : the length of the sequence that we are going to generate with our model

```python
import os
import pretty_midi
from scipy.io import wavfile 
import IPython

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Dropout, Activation
from tensorflow.keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import glob

n_x = 79
max_T_x = 1000
sequence_length = 20
T_y_generated = 200

DIR = './'
import urllib.request
midiFile_l = ['cs1-2all.mid', 'cs5-1pre.mid', 'cs4-1pre.mid', 'cs3-5bou.mid', 'cs1-4sar.mid', 'cs2-5men.mid', 'cs3-3cou.mid', 'cs2-3cou.mid', 'cs1-6gig.mid', 'cs6-4sar.mid', 'cs4-5bou.mid', 'cs4-3cou.mid', 'cs5-3cou.mid', 'cs6-5gav.mid', 'cs6-6gig.mid', 'cs6-2all.mid', 'cs2-1pre.mid', 'cs3-1pre.mid', 'cs3-6gig.mid', 'cs2-6gig.mid', 'cs2-4sar.mid', 'cs3-4sar.mid', 'cs1-5men.mid', 'cs1-3cou.mid', 'cs6-1pre.mid', 'cs2-2all.mid', 'cs3-2all.mid', 'cs1-1pre.mid', 'cs5-2all.mid', 'cs4-2all.mid', 'cs5-5gav.mid', 'cs4-6gig.mid', 'cs5-6gig.mid', 'cs5-4sar.mid', 'cs4-4sar.mid', 'cs6-3cou.mid']
for midiFile in midiFile_l:
  #if os.path.isfile(DIR + midiFile) is None:
  urllib.request.urlretrieve ("http://www.jsbach.net/midi/" + midiFile, DIR + midiFile)
nbExample = len(midiFile_l)

midiFile_l = glob.glob(DIR + 'cs*.mid')
print(midiFile_l)
```

## Reading and converting all MIDI files

Let's now read all MIDI files and convert their content to one-hot-encoding matrix X_ohe of dimensions (T_x, n_x) where as we said n_x is the number of possible musical notes.

```python
# We truncate the duration of each example to the first T_x data

X_list = []

for midiFile in midiFile_l:
    # read the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midiFile)
    note_l = [note.pitch for note in midi_data.instruments[0].notes]
    # convert to one-hot-encoding
    T_x = len(note_l)
    if T_x > max_T_x:
      T_x = max_T_x
    X_ohe = np.zeros((T_x, n_x))
    for t in range(T_x): 
      X_ohe[t, note_l[t]-1] = 1
    # add to the list  
    X_list.append(X_ohe)
    
print(len(X_list))
print(X_list[0].shape)
print(X_list[1].shape)
print(X_list[2].shape)
```

We can now display the set of notes over time for a specific track.

```python
plt.figure(figsize=(16, 6))
plt.imshow(X_list[2].T, aspect='auto')
plt.set_cmap('gray_r')
plt.grid(True)
```

![image](https://raphaellederman.github.io/assets/images/midinotes.png){:height="100%" width="100%"}

## Shaping data for the training of language model

To proceed, for each example/sequence and each possible starting note in this example/sequence, we create two sequences :
* An input sequence which contains a sub-sequence of length sequence_length. This sub-sequence ranges from the note $$\textit{t}$$ to the note $$\textit{t+sequence\_length-1}$$
* An output sequence which contains the following note to be predicted, the one at position  $$\textit{t+sequence\_length}$$
The training is therefore performed by giving to the model a set of sequences as input and asking the network to predict each time the note that should come right after this sequence.

```python
X_train_list = []
y_train_list = []

# CODE-RNN2-1
# --- START CODE HERE
X_train_list = [X_list[i][t:t+sequence_length] for i in range(len(X_list)) for t in range(len(X_list[i])-sequence_length)]
y_train_list = [X_list[i][t+ sequence_length] for i in range(len(X_list)) for t in range(len(X_list[i])-sequence_length)]
# --- END CODE HERE

X_train = np.asarray(X_train_list)
y_train = np.asarray(y_train_list)

print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
```
## Training the language model

The language model will be learned by training an RNN with input X_train and output Y_train: for each of the examples of sequences, we give to the network a sequence of notes of sequence_length duration, and ask the network to predict the following note of each sequence.

The network will have the following structure :
* a layer of LSTM with $$\textit{na = 256}$$
* a DropOut layer with rate 0.3 (the probability to "drop-out" one neuron is 0.3)
* a second layer of LSTM with $$\textit{na = 256}$$
* a DropOut layer with rate 0.3 (the probability to "drop-out" one neuron is 0.3)
* a third layer of LSTM with $$\textit{na = 256}$$
* a Dense layer with 256 units
* a DropOut layer with rate 0.3 (the probability to "drop-out" one neuron is 0.3)
* a Dense layer with a softmax activation which predict the probability of each of the $$\textit{nx}$$ notes as output

Note that because we will stack three LSTM layers on top of each other (deep-RNN), we need to tell the first two LSTM to output their hidden states at each time $$\textit{t}$$. This is done by the option return_sequences=True that has to be given to the first two LSTM.

This is not the case of the third LSTM since we are only interrested by its final prediction (hence return_sequences=False, which is the default behaviour).


```python
X_train_list = []
y_train_list = []

# CODE-RNN2-1
# --- START CODE HERE
X_train_list = [X_list[i][t:t+sequence_length] for i in range(len(X_list)) for t in range(len(X_list[i])-sequence_length)]
y_train_list = [X_list[i][t+ sequence_length] for i in range(len(X_list)) for t in range(len(X_list[i])-sequence_length)]
# --- END CODE HERE

X_train = np.asarray(X_train_list)
y_train = np.asarray(y_train_list)

print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
```

![image](https://raphaellederman.github.io/assets/images/lstmsummary.png){:height="100%" width="100%"}

We can now compile and run the model on TPU.

```python
def model_to_tpu(model):
  return tf.contrib.tpu.keras_to_tpu_model(model,
      strategy=tf.contrib.tpu.TPUDistributionStrategy(
          tf.contrib.cluster_resolver.TPUClusterResolver(
              tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    )
)

Xtpu_model = model_to_tpu(model)
tpu_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
tpu_model.fit(X_train, y_train, epochs=50, batch_size=64)
```

## Generating a new sequence from sampling the language model

To generate a new sequence from the language model, we simply give it as input a randomly selected sequence out of the 2880 sequences used for training of duration sequence_length and ask the trained network to predict the output (using model.predict). The output of the network is a vector of probability of dimension $$\textit{nx}$$ which represents the probability of each note to be the next note of the melody given as input. From this vector, we select the note which has the maximum probability.

We then concatenate this new note (its one-hot-encoding representation) at the end of the input sequence. We finally remove the first element of the input sequence to keep its duration constant (sequence_length).

```python
# Select a random starting pattern
start = np.random.randint(0, len(X_train_list)-1)
pattern = X_train_list[start]
print(start)
print(pattern.shape)
print(np.expand_dims(pattern, 0).shape)

# Cast our model to CPU for single prediction
model = tpu_model.sync_to_cpu()
note_l = []
prediction_l = []

# Generate T_y_generated notes
for note_index in range(T_y_generated):
    pred = model.predict(np.expand_dims(pattern[note_index:,:], 0))
    prediction_l.append(pred)
    note = np.argmax(pred, axis=1)
    note_l.append(note)
    note_ohe = np.zeros(79)
    note_ohe[note] = 1
    pattern = np.vstack((pattern, note_ohe))
```

## Display the generated sequence

```python
plt.figure(figsize=(16, 6))
plt.imshow(np.asarray(prediction_l)[:,0,:].T, aspect='auto')
plt.plot(note_l)
plt.set_cmap('gray_r')
plt.grid(True)
```

![image](https://raphaellederman.github.io/assets/images/finalsequence.png){:height="100%" width="100%"}

## Create a MIDI file and an audio file which correspond to the generated sequence

Once the new sequence has been generated (note_l) we transform it to a new MIDI file and perform a very basic rendering of it in an audio file.

```python
new_midi_data = pretty_midi.PrettyMIDI()
cello_program = pretty_midi.instrument_name_to_program('Cello')
cello = pretty_midi.Instrument(program=cello_program)
time = 0
step = 0.3
for note_number in note_l:
    myNote = pretty_midi.Note(velocity=100, pitch=note_number, start=time, end=time+step)
    cello.notes.append(myNote)
    time += step
new_midi_data.instruments.append(cello)
%matplotlib inline

audio_data = new_midi_data.synthesize()
IPython.display.Audio(audio_data, rate=44100)
```

> **Conclusion** : this short article shows a very basic way to generate music sequences using deep recurrent neural networks. Of course it could be improved by generating polyphonic sequences that takes into account harmony.