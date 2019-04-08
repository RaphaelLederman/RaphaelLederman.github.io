---
layout: archive
title: "Data Science"
permalink: /datascience/
author_profile: true
classes: wide
header :
    image: "https://raphaellederman.github.io/assets/images/mountain.jpg"
---


## GitHub Projects

### Deep Learning : Multimodal Emotion Recognition (Text, Audio, Video)

This research project is made in the context of an exploratory analysis for the French employment agency (Pole Emploi), and is part of the Big Data program at Telecom ParisTech. This project is currently being developed and should be finished in May 2019.

The aim of this project is to provide candidates seeking for a job a platform that analyses their interview performance, emotions, and personality traits through text, audio and video processing. The platform should finally provide a detailed report to the candidates in order to help them improve their interview skills.

Through this project, we explore state of the art multimodal emotion recognition methods, and work on developing an ensemble model that gathers the information from all these sources and displays it in a clear and interpretable way.

For a more detailed description of this research project, you can have a look at the preliminary report. It describes some of the statistical tools and model architectures used in order to develop our platform. 

<embed src="https://raphaellederman.github.io/assets/images/multimodal.pdf" type="application/pdf" width="600px" height="500px" />

See the GitHub page for more information : <span style="color:blue">[https://github.com/maelfabien/Mutlimodal-Sentiment-Analysis](https://github.com/RaphaelLederman/Multimodal-Emotion-Recognition)</span>

### Big Data : GDELT Project

The GDELT Project monitors broadcasts, prints, and web news from all over the world in more than 100 different languages, identifying a vast amount of characteristics linked to these events such as the people, locations or emotions. It is a free open platform with new files uploaded every 15 minutes (the GDELT database contains more than 700 Gb of zipped data for the single year of 2018).

This project consisted in building a resilient architecture for storing large amount of data from the GDELT database allowing fast responding queries. We have chosen to work with the following architecture :

* NoSQL Database : Cassandra
* AWS : EMR to transfer the data to Cassandra, and EC2 for the resiliency for the requests
* Visualization : Zeppelin Notebook

Here is a graphic of our architecture :

![image](https://raphaellederman.github.io/assets/images/archi.jpg)

For more details on our data processing approach, have a look at the GitHub page :  <span style="color:blue">[https://github.com/maelfabien/Mutlimodal-Sentiment-Analysis](https://github.com/RaphaelLederman/Cassandra-GDELT-Queries)</span>


