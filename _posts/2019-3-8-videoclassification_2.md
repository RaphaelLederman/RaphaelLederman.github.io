---
published: true
title: Video Classification using Deep Learning -  Classification (2)
collection: articles
layout: single
author_profile: false
read_time: true
categories: [articles]
excerpt : "Computer Vision"
header :
    overlay_image: "https://raphaellederman.github.io/assets/images/night.jpg"
    teaser_image: "https://raphaellederman.github.io/assets/images/night.jpg"
toc: true
toc_sticky: true
---

In this second article on personality traits recognition through computer vision, we will show how to transform video inputs into sequences of preprocessed images, and feed these sequences to a deep learning model using CNN and LSTM in order to perform personality traits detection.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Converting videos to sequences of preprocessed images

The first step is to build a function that transforms any video into a sequence of 48x48 images centered around a detected face. In order to do so, we use VideoCapture from OpenCV in order to split a video input into a sequence of successive images, and then apply our preprocessing in order to obtain standardized images.

```python
def FrameCapture(path):  
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
    count = 0
    success = 1   
    images =[]  
    while success: 
        success, image = vidObj.read()
        if image is not None: 
            image = extract_face_features(detect_face(image))[0] 
            if image !=[]:
                images.append(image)
                count += 1        
        else:
            success = False
    result = np.array([images[i][0] for i in range(len(images))])
    return result
```

We now have a sequence of preprocessed images. In order to build a consistent dataset for video classification, we could reformat each video so that it has a predetermined shape, let's say (420, 48, 48) (each video being therefore composed of 420 different images). In order to do so, we could apply some kind of padding :
* if the video is longer than 420 images, we truncate it and only take the first 420 images.
* if the video is shorter than 420 images, we can sample a sequence from itself and add it at the end in order to fill the gap (for instance if a video is composed of 400 images, we can "repeat" the last 20 images). 
Such a method is not optimal as it affects the temporal consistency of the sequence. Hopefully some deep learning frameworks (like PyTorch) do not need sequences to be padded in order to perform classification or other tasks, but Keras requires sequences to have a constant shape. As we will use Keras in this short article, I will provide a very basic example of such padding for video inputs.

```python
def pad_video(video):
    if video.shape[0] < 420:
        while video.shape[0] < 420:
            video_padded = np.concatenate((video, video[-(420-len(video)):]), axis = 0)
            if video_padded.shape[0] == 420:
                break
    else:
        video_padded = video[:420] 
    return video_padded
```

This preprocessing pipeline can then be applied to the whole First Impression V2 dataset in order to start training a classification model. In order to do so, we create a dictionary with the video names as keys and the corresponding sequences of images as values (it is better to use ordered dictionnaries).

```python
shape_x = 48
shape_y = 48
videonames = []
video_di = {}
local ='XXX/data/train_data/'
directory_names = [f for f in listdir(local) if not(isfile(join(local, f)))]
for di in directory_names:
    mypath = local + di + "/"
    subdirectory_names = [f for f in listdir(mypath) if not(isfile(join(mypath, f)))]
    for subdi in subdirectory_names:
        finalpath = mypath + subdi + "/"
        video_names = [f for f in listdir(finalpath) if isfile(join(finalpath, f))]
        for video in video_names:
            video_di[video] = pad_video(FrameCapture(finalpath+video))
            videonames.append(video)
```

## Building an appropriate classification model

Now that we know how to properly preprocess our video data, we can start building a neural network architecture in order to perform classification. Here, our objective is to choose an architecture that is consistent with the temporal nature of our data : we will use a convolution layer along with an LSTM cell. For more information on Long-Short Term Memory cells, you can have a look at this short [article](https://raphaellederman.github.io/articles/rnn/#) I wrote on Recurrent Neural Networks.

```python
dim = (420,48,48,1)
inputShape = (dim)
Input_words = Input(shape=inputShape, name='input_vid')
x = TimeDistributed(Conv2D(filters=50, kernel_size=(8,8), padding='same', activation='relu'))(Input_words)
x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
x = TimeDistributed(SpatialDropout2D(0.2))(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(x)
out = Dense(5,activation='softmax')(x)
model = Model(inputs=Input_words, outputs=[out])
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss = 'categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])
```

Now we can retrieve our training data and fit the model. We add one dimension to our training sequences as all images are in grayscale (therefore the last dimension is 1, compared to 3 in the case of RGB images). Finally, we retrieve the labels from ordered dictionnaries.

```python
X = np.expand_dims(np.stack(X), axis = 4)
y = np.asarray(list(zip(list(labels['extraversion'].values()), list(labels['neuroticism'].values()), list(labels['agreeablenes'].values()),list(labels['conscientiousness'].values()),list(labels['openness'].values()))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = model.fit(X_train, y_train)
pred = model.predict(X_test)
score = multilabel_accuracy(pred,y_test)
print(score)
```

We could perform some tuning in order to provide better predictions, trying different architectures and finding the best set of hyperparameters, but this is not the aim of this short articles.

> **Conclusion** : in this series of articles, we went through the most important steps in building a video classification pipeline in the context of personality traits detection with inputs consisting of videos of people facing and speaking in English to a camera. We divided each video into a padded sequence of preprocessed images (scaled and centered around a detected face), and fed these sequences to a deep learning architecture in order to perform classification.