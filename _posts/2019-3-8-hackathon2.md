---
published: true
title: Hackathon : what is the next hit of 2019 ? (2)
collection: articles
layout: single
author_profile: false
read_time: true
categories: [articles]
excerpt : "Hackathon"
header :
    overlay_image: "https://raphaellederman.github.io/assets/images/night.jpg"
    teaser_image: "https://raphaellederman.github.io/assets/images/night.jpg"
toc: true
toc_sticky: true
---

The next step of this hackathon was to explore thoroughly the music features of the hit songs using the Spotify API in order to obtain a kind of "music profile" for the next hit. The features that we retrieved from the API are the following:

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

* duration_ms : The duration of the track in milliseconds.
* key : The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
* mode : Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0 (-1 for no result). Note that the major key (e.g. C major) could more likely be confused with the minor key at 3 semitones lower (e.g. A minor) as both keys carry the same pitches.
* time_signature : An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of “3/4”, to “7/4”.
* acousticness : A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
* danceability : Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
* energy : Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
* instrumentalness :Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
* liveness : Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
* loudness : The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
* speechiness : Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
* valence : A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). The distribution of values for this feature look like this: Valence distribution tempo float The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
* tempo : The overall estimated tempo of the section in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
* key : The estimated overall key of the section. The values in this field ranging from 0 to 11 mapping to pitches using standard Pitch Class notation (E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on). If no key was detected, the value is -1.
* mode_confidence: The confidence, from 0.0 to 1.0, of the reliability of the mode.

```python
import requests, json, logging
import pandas as pd
import base64
import six

def get_info(song_name = 'africa', artist_name = 'toto', req_type = 'track'):
    client_id = '***'
    client_secret = '***'
    auth_header = {'Authorization' : 'Basic %s' % base64.b64encode(six.text_type(client_id + ':' +                              client_secret).encode('ascii')).decode('ascii')}
    r = requests.post('https://accounts.spotify.com/api/token', headers = auth_header, data= {'grant_type': 'client_credentials'})
    token = 'Bearer {}'.format(r.json()['access_token'])
    headers = {'Authorization': token, "Accept": 'application/json', 'Content-Type': "application/json"}

    payload = {"q" : "artist:{} track:{}".format(artist_name, song_name), "type": req_type, "limit": "1"}

    res = requests.get('https://api.spotify.com/v1/search', params = payload, headers = headers)
    res = res.json()['tracks']['items'][0]
    year = res['album']['release_date'][:4]
    month = res['album']['release_date'][5:7]
    day = res['album']['release_date'][8:10]
    artist_id = res['artists'][0]['id']
    artist_name = res['artists'][0]['name'].lower()
    song_name = res['name'].lower()
    track_id = res['id']
    track_pop = res['popularity']

    res = requests.get('https://api.spotify.com/v1/audio-analysis/{}'.format(track_id), headers = headers)
    res = res.json()['track']
    duration = res['duration']
    end_fade = res['end_of_fade_in']
    key = res['key']
    key_con = res['key_confidence']
    mode = res['mode']
    mode_con = res['mode_confidence']
    start_fade = res['start_of_fade_out']
    temp = res['tempo']
    time_sig = res['time_signature']
    time_sig_con = res['time_signature_confidence']

    res = requests.get('https://api.spotify.com/v1/audio-features/{}'.format(track_id), headers = headers)
    res = res.json()
    acousticness =  res['acousticness']
    danceability = res['danceability']
    energy = res['energy']
    instrumentalness = res['instrumentalness']
    liveness = res['liveness']
    loudness = res['loudness']
    speechiness = res['speechiness']
    valence = res['valence']

    res = requests.get('https://api.spotify.com/v1/artists/{}'.format(artist_id), headers = headers)
    artist_hot = res.json()['popularity']/100

    return pd.Series([artist_name, song_name, duration, key,mode,temp,artist_hot,end_fade, start_fade, mode_con,key_con,time_sig,time_sig_con,acousticness,danceability,energy ,instrumentalness,liveness,loudness,speechiness,valence, year, month, day, track_pop], index = ['artist_name', 'song_name', 'duration','key','mode','tempo','artist_hotttnesss','end_of_fade_in','start_of_fade_out','mode_confidence','key_confidence','time_signature','time_signature_confidence','acousticness','danceability','energy' ,'instrumentalness','liveness','loudness','speechiness','valence','year','month', 'day', 'track_popularity'])
```
The following function tests whether a song request through the API is successful or not.

```python
def test(song_name = 'africa', artist_name = 'toto', req_type = 'track'):
    client_id = '***'
    client_secret = '***'
    auth_header = {'Authorization' : 'Basic %s' % base64.b64encode(six.text_type(client_id + ':' + client_secret).encode('ascii')).decode('ascii')}
    r = requests.post('https://accounts.spotify.com/api/token', headers = auth_header, data= {'grant_type': 'client_credentials'})
    token = 'Bearer {}'.format(r.json()['access_token'])
    headers = {'Authorization': token, "Accept": 'application/json', 'Content-Type': "application/json"}

    payload = {"q" : "artist:{} track:{}".format(artist_name, song_name), "type": req_type, "limit": "1"}

    res = requests.get('https://api.spotify.com/v1/search', params = payload, headers = headers)
    if not res.json()['tracks']['items']:
        return False
    else:
        return True
```

This part of the code iterates over our dataset of hit songs (.csv file) in order to request the Spotify API and retrieve the audio features. Everything is gathered in a single dataframe, and we create a feature by combining the mode confidence and the mode together. We then proceeed to some data cleaning :

```python
song_list = pd.read_csv('/Users/raphaellederman/Downloads/Tracks_Hackathon_treated (4).csv', sep = ';')
print(type(song_list['Track Name']))

rows= []
features = ['artist_name', 'song_name', 'duration','key','mode','tempo','artist_hotttnesss','end_of_fade_in','start_of_fade_out','mode_confidence','key_confidence','time_signature','time_signature_confidence','acousticness','danceability','energy' ,'instrumentalness','liveness','loudness','speechiness','valence','year','month', 'day', 'track_popularity']

for index, row in song_list.iterrows():
    print(row['Track Name'].replace('\'','') + ' - ' + row['Artist'])
    if test(row['Track Name'].replace('\'',''), row['Artist'], req_type = 'track') == True :
        rows.append(get_info(row['Track Name'].replace('\'',''), row['Artist'], req_type = 'track'))

data = pd.DataFrame(rows, columns=features)
data['mode_confidence'] = np.where(data['mode'] == 1, data['mode']* data['mode_confidence'], (data['mode']- 1)* data['mode_confidence'])
data = data.drop('mode', axis=1)

data_songs = data_songs.reset_index().drop(['index', 'artist_name', 'song_name'], axis=1).replace('', np.nan).dropna()
```

We chose to display a correlation matrix between features in order to analyze if there were some kind of redundancies or dependancies.

```python
import seaborn as sn
fig = plt.figure(figsize=(20, 20))
sn.heatmap(data.corr(),  annot=True)
plt.title('Correlation of every features', fontsize=20)
```
![image](https://raphaellederman.github.io/images/corr.png){:height="100%" width="100%"}

Our goal was to build a model predicting the likelihood of an artist being considered as hot in 2019, this is the reason why we filtered our dataset, retaining only features that had a correlation of more than 0.1 or less than -0.1 with the artist's "hotness".

```python
features = [ 'duration', 'tempo', 'danceability', 'end_of_fade_in', 'start_of_fade_out','energy', 'speechiness','valence','track_popularity']
i=1

for feature in features : 
    plt.figure(figsize=(15,15))
    plt.subplot(3,3, i )
    plt.scatter(data['artist_hotttnesss'], data[feature])
    plt.title("correlation between artist_hotttnesss and " + feature)
    i+=1
```

We found an interesting relation between an artist's hotness and his speechiness that could be related to the relative popularity of rap music.

![image](https://raphaellederman.github.io/images/speechiness.png){:height="100%" width="100%"}

Among the most important features, we observe that the ideal tempo should be around 120 BPM, and the ideal duration should be around 213 seconds for a hit (using the music features values for the most popular songs of 2018).

We used a Random Forest Regressor to evaluate features importance when predicting the track popularity :

```python

features_columns = [col for col in data.drop("track_popularity", axis = 1).columns]
X = data[features_columns].apply(pd.to_numeric, errors='coerce')
y = data['track_popularity'].apply(pd.to_numeric, errors='coerce')

# Split the data in order to compute the accuracy score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

rnd_clf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

importances_rf = rnd_clf.feature_importances_
indices_rf = np.argsort(importances_rf)

n = len(indices_rf)
sorted_features_rf = [0] * n;  
for i in range(0,n): 
sorted_features_rf[indices_rf[i]] = features_columns[i] 

plt.figure(figsize=(140,120) )
plt.title('Random Forest Features Importance')
plt.barh(range(len(indices_rf)), importances_rf[indices_rf], color='b', align='center')
plt.yticks(range(len(indices_rf)), sorted_features_rf)
plt.xlabel('Relative Importance')
plt.tick_params(axis='both', which='major', labelsize=100)
plt.tick_params(axis='both', which='minor', labelsize=100)

plt.show()
```
![image](https://raphaellederman.github.io/images/importance.png){:height="100%" width="100%"}

We finally used XGBoost to predict the popularity of a song based on its features, but the overall training set was too small (26% r-squared only).

```python
clf = xgboost.XGBRegressor(colsample_bytree = 0.44, n_estimators=30000, learning_rate=0.07,max_depth=9,alpha = 5)
model = clf.fit(X_train.drop(worst_features, axis=1), y_train)
pred = model.predict(X_test.drop(worst_features, axis=1))
score_rf = metrics.r2_score(y_test, pred)
print(score_rf)
```

Our last thought was to determine the optimal release time in the year, using Spotify's API to gather release dates of hit songs : March seemed to be the best option.
 
```python 
s = pd.Series(data['month']).dropna()
fig, ax = plt.subplots(figsize = (12,12))
ax.hist(s, alpha=0.8, color='blue', bins = 25)
ax.xaxis.set_ticks(range(13))
ax.xaxis.set_ticklabels( [' ','Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre'])
plt.title("Historgram of the number of hit by month")
```
![image](https://raphaellederman.github.io/images/release.png){:height="100%" width="100%"}
