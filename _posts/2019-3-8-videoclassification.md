---
published: true
title: Video Classification using Deep Learning - Facial Detection and Feature Extraction (1)
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

In this series of articles, we will present some of the tools available in order to preprocess video data for personality traits detection and propose a deep learning classification model adapted to this task. The personality traits we are going to detect are the following : openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism (see [here](https://en.wikipedia.org/wiki/Big_Five_personality_traits) for more information on the Big Five model in psychology). Our original dataset is the First Impression V2 [dataset](http://chalearnlap.cvc.uab.es/dataset/24/description/), comprising 10,000 clips (with an average duration of 15 seconds) extracted from more than 3,000 different YouTube high-definition videos of people facing and speaking in English to a camera. Theses videos were labeled with the personality traits from the Big Five model using [Amazon Mechanical Turk](https://www.mturk.com/) (AMT).

In this first article, we will focus on single images preprocessing (detecting faces and extracting facial features). 

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

> **Conclusion** : in this first article, we went through some preprocessing tools in order to perform video classification. In the next articlen we will show how to divide each video into a padded sequence of preprocessed images (scladed and centered around a detected face), and feed these sequences to a deep learning architecture in order to perform classification.