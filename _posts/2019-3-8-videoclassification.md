---
published: true
title: Video Classification using Deep Learning
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

In this article, we will present some of the tools available in order to preprocess video data for personality traits detection and propose a deep learning classification model adapted to this task. The personality traits we are going to detect are the following : openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism (see [here](https://en.wikipedia.org/wiki/Big_Five_personality_traits) for more information on the Big Five model in psychology). Our original dataset is the First Impression V2 [dataset](http://chalearnlap.cvc.uab.es/dataset/24/description/), comprising 10,000 clips (with an average duration of 15 seconds) extracted from more than 3,000 different YouTube high-definition videos of people facing and speaking in English to a camera. Theses videos were labeled with the personality traits from the Big Five model using [Amazon Mechanical Turk](https://www.mturk.com/) (AMT).

We will first focus on single images preprocessing (detecting faces, extracting facial features...). Then we will show how to transform video inputs into sequences of preprocessed images, and feed these sequences to a deep learning model using CNN and LSTM in order to perform personality traits detection.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Preprocessing : facial detection using the Viola–Jones object detection framework.

Let's start using OpenCV (Open Source Computer Vision Library), an open source library for computer vision written in C/C++ with interfaces in C++, Python and Java. These are a few imports that are going to be useful for our classification.

```python
import cv2
import matplotlib.pyplot as plt
from imutils import face_utils
```

First we will define a function that is able to detect a face on a given image. In order to do so efficiently, we will use a Casacade classifier available with the OpenCV library. These types of classifiers are based on boosting, a family of ensemble learning algorithms which converts weak learner to strong learners by training each weak learner sequentially (each one trying to correct its predecessor). These weak learners can typically be decision trees with a single split (called decision stumps), as in the case of AdaBoost (Adaptive Boosting). 

The Viola–Jones object detection framework is an effective method proposed by Paul Viola and Michael Jones in their [paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It relies on such classifiers : it is the methodology that we will use in order to detect faces on our sequences of images. This algorithm proceeds in four different steps in order to perform accurate detection : Haar feature selection, integral image creation, Adaboost training and finally classifiers cascading.

The Haar features are first extracted from positively and negatively labelled images (with faces and witout faces in it) : these features are just like convolutional kernels, each one being a single value obtained by subtracting the sum of pixels under the white areas from the sum of pixels under the black areas. There are several rectangle structures that can be applied in this fashion, for instance two-rectangle features are mainly used for detecting edges and three-rectangle features mainly used for detecting lines.

![image](https://raphaellederman.github.io/assets/images/haar_features.png){:height="100%" width="100%"}

All possible sizes and locations of each kernel are then used to calculate hundreds of thousands features, and the extraction process roughly corresponds to the following image. It is important to note that these Haar features basically correspond to common human face image features, for instance a dark eye region compared to upper-cheeks.

![image](https://raphaellederman.github.io/assets/images/haar_features_2.png){:height="100%" width="100%"}

In order to improve the efficiency of the rectangle features computation, the authors proposed an intermediate representation for the image : the Integral Image (reducing the calculations for a given pixel to an operation involving just four pixels). This method can also be used for calculating the average intensity within a given image. For a more detailed explanation on the integral image, go visit this [website](https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/).

Then, in order to avoid calculating too many features, and make the extraction process more time efficient, the authors use the AdaBoost algorithm to select the best set of features (single rectangle features which split best negative and positive examples). This is really important because most of the features primarily calculated are irrelevant : if a window relies on the property that the eyes region is generally darker than the bridge of the nose (line feature 2.b for instance in the first image above), it should be applied essentially to this particular region (eyes/nose) and not to other regions (for instance the mouth region). In this way, they combined a set of weak classifiers in order to create an accurate ensemble model with fewer features (the authors achieved an accuracy of 95% while retaining only 200 of the original features, and finally chose to select 6,000 features).

The last crucial step in the Viola–Jones object detection framework is the Cascade of Classifiers. This concept helps discarding non-face regions in an image, so that more time is spent evaluating possible face regions. Instead of applying all 6000 features on a window, the features are grouped into different stages of classifiers and applied one-by-one : if a window fails the first stage, it is automatically discarded and if it passes, the second stage of features is applied and the process goes on. A particular window is considered to be a face region only if it passes all stages (on average only 10 features out of more than 6000 are evaluated per sub-window thanks to this filtering). The authors finally organized the features in 38 stages with 1, 10, 25, 25 and 50 features in the first five stages. The Cascade of Classifiers were trained using Adaboost and adjusting the threshold to minimize the false rate, using the following hyperparameters : the number of classifier stages, the number of features in each stage, the threshold of each stage. In the following image, we can see the two features obtained as the best two features from Adaboost). 

![image](https://raphaellederman.github.io/assets/images/haar_features_3.png){:height="100%" width="100%"}

In the following function, we first locate the pre-trained weights and load them in order to use the model. We then convert our image to grayscale, and apply our cascade classifier in order to detect multiple faces (the scaleFactor and minNeighbors correspond respectively to how much the image size is reduced at each image scale and how many neighbors each candidate rectangle should have to retain it). We then add a rectangle around the detected face on our image, and get back different elements (grayscale picture and detected faces coordinates).

```python
def detect_face(frame):
    
    #Cascade classifier pre-trained model
    cascPath = "/anaconda3/envs/py35/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    #BGR -> Gray conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Cascade MultiScale classifier
    detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10,
                                                  minSize=(shape_x, shape_y),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    coord = []
    
    for x, y, w, h in detected_faces :
        if w>500:
            sub_img=frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
            coord.append([x,y,w,h])
        
    return gray, detected_faces, coord
```

Here is an example : we first take a photo of Jimi Hendrix.

![image](https://raphaellederman.github.io/assets/images/hendrix.jpg){:height="100%" width="100%"}

We then apply our function in order to obtain a grayscale image, detect the face and add a rectangle around it.

![image](https://raphaellederman.github.io/assets/images/hendrix_frame.png){:height="100%" width="100%"}

This object extraction method can also be applied to eyes or mouth detection.

## Extracting facial features

We will now extract from our original image a new one, scaled and centered around the detected face, with a predefined resolution (here we chose to produce a 48x48 image).

```python
def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
    gray = faces[0]
    detected_face = faces[1]
    new_face = []
    for det in detected_face :
        if det != ():
            x, y, w, h = det
        	
        	#Offset coefficient, np.floor takes the lowest integer (delete border of the image)
            horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
            vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
            extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]

            #Zoom on the extracted face
            new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
            
            #cast type float
            new_extracted_face = new_extracted_face.astype(np.float32)
            
            #scale
            new_extracted_face /= float(new_extracted_face.max())
            new_face.append(new_extracted_face)
            
        else:
            pass
    
    return new_face,
```

Here is what we obtain when we apply the extract_face_features function to the output of our detect_face function based on our original Jimi Hendrix picture :

![image](https://raphaellederman.github.io/assets/images/zoom_hendrix.png){:height="100%" width="100%"}

## Converting videos to sequences of preprocessed images

The next step is to build a function that transforms any video into a sequence of 48x48 images centered around a detected face. In order to do so, we use VideoCapture from OpenCV in order to split a video input into a sequence of successive images, and then apply our preprocessing in order to obtain standardized images.

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
We could perform some tuning in order to provide better predictions, trying different architectures and using Bayesian optimization to find the best set of hyperparameters, but this is not the aim of this short articles.

> **Conclusion** : in this short article, we went through the most important steps in building a video classification pipeline in the context of personality traits detection with inputs consisting of videos of people facing and speaking in English to a camera. We divided each video into a padded sequence of preprocessed images (scladed and centered around a detected face), and fed these sequences to a deep learning architecture in order to perform classification.