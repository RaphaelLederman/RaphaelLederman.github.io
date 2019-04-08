---
published: true
title: Hackathon - the next hit of 2019
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

Here is a article describing a 10 hours Hackathon to which I participated with a team of 5 friends from Télécom ParisTech this year around the following theme : what are the characteristics of the hit song of 2019 ?
I will provide some of the code that we write and the main outputs of this incredible experience.

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Which country represents the biggest market for music ?

We first tried to select the country that represented the biggest market by looking at the total number of streams in the 10 regions of the world that listen to the most music. We downloaded datas of streams from the Spotify Top 200 charts (https://spotifycharts.com/regional) and then gathered the data into one dataset, grouping some rows as the data was ordered by regions and weeks.

```python
folder = "**********/Hackathon/Total/"
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print("Working with {0} csv files".format(len(onlyfiles)))

data = []
for file in onlyfiles :
	if file != '.DS_Store' :
		df = pd.read_csv(folder + file, skiprows=1)
		df['country'] = file[9:11]
		df['week'] = file[19:20]
		data.append(df)
data = pd.concat(data, axis=0)

data_country = data.groupby(['country'])[["Streams"]].sum().sort_values('Streams', ascending=True)
data_artists = data.groupby(['Artist'])[["Streams"]].sum().sort_values('Streams', ascending=False)
```

![image](https://raphaellederman.github.io/assets/images/tabh1.png){:height="100%" width="100%"}

In the following code, we used ploty in order to display the data in a more esthetic way.

```python
trace0 = go.Bar(
	x=data_country.index,
	y=data_country['Streams'],
	text=data_country['Streams'],
	marker=dict(
		color='rgb(158,202,225)',
		line=dict(
			color='rgb(8,48,107)',
			width=1.5,
		)
	),
	opacity=0.6
)

data = [trace0]
layout = go.Layout(
	title='Regions of the world that consume most music',
)

fig1 = go.Figure(data=data, layout=layout)
py.iplot(fig1, filename='text-hover-bar')
```

![image](https://raphaellederman.github.io/assets/images/US.png){:height="100%" width="100%"}

From this visualization, the choice was pretty clear : the USA represent by far the largest market, and we therefore chose to focus on this country.

## Which artists are the most popular, and what are their nationalities ?

The next step was to determine who were the most popular artists in terms on streams at the time. You can see the ranking on the next picture, XXXTentacion leading the scoreboard.

![image](https://raphaellederman.github.io/assets/images/artist.png){:height="100%" width="100%"}

We then thought about determining which nationality seemed to be the most "popular" for artists : based on this list (https://www.thefamouspeople.com/singers.php), we have scrapped some data regarding the nationality of famous artists.

```python
def _handle_request_result_and_build_soup(request_result):
	if request_result.status_code == 200:
	html_doc =  request_result.text
	soup = BeautifulSoup(html_doc,"html.parser")
	return soup

def _convert_string_to_int(string):
	if "K" in string:
		string = string.strip()[:-1]
		return float(string.replace(',','.'))*1000
	else:
		return int(string.strip())

def get_all_links_for_query(query):
	url = website + "/rechercher/"
	res = requests.post(url, data = {'q': query })
	soup = _handle_request_result_and_build_soup(res)
	specific_class = "c-article-flux__title"
	all_links = map(lambda x : x.attrs['href'] , soup.find_all("a", class_= specific_class))
	return all_links

def get_share_count_for_page(page_url):
	res = requests.get(page_url)
	soup = _handle_request_result_and_build_soup(res)
	specific_class = "c-sharebox__stats-number"
	share_count_text = soup.find("span", class_= specific_class).text
	return  _convert_string_to_int(share_count_text)


def get_popularity_for_people(query):  
	url_people = get_all_links_for_query(query)
	results_people = []

	for url in url_people:
		results_people.append(get_share_count_for_page(website_prefix + url))
	return sum(results_people)

def get_name_nationality(page_url):
	res = requests.get(page_url)
	soup = _handle_request_result_and_build_soup(res)
	specific_class = "btn btn-primary btn-sm btn-block btn-block-margin"
	share_count_text = soup.find("a", class_= specific_class).text
	return  share_count_text

artists_dict = {}

for i in range(1, 17):
	website = 'https://www.thefamouspeople.com/singers.php?page='+str(i)
	res = requests.get(website)
	specific_class = "btn btn-primary btn-sm btn-block btn-block-margin"
	soup = _handle_request_result_and_build_soup(res)
	classes = soup.find_all("a", class_= specific_class)
	for i in classes:
		mini_array = i.text[:-1].split('(')
		artists_dict[mini_array[0]]=mini_array[1]

artists_df = pd.DataFrame.from_dict(artists_dict, orient='index', columns=['Country'])
artists_df.head(n=10)
```
The result was not a surprise : being American as an artist seems to be an advantage.

![image](https://raphaellederman.github.io/assets/images/nationality.png){:height="100%" width="100%"}

## Which type of music is the most popular ?

Our next objective was to determine which style of music was the most popular using Google Trends. For this purpose, we extracted a CSV file covering the past 5 years comparing internet research for different styles. As you can see, Rap music seems to be the winner.

![image](https://raphaellederman.github.io/assets/images/trends.png){:height="100%" width="100%"}


## What are the characteristics of hits' lyircs ?

We then focused on the lyrics : we chose to analyze the lyrics of the top 200 songs of the Spotify charts using the Genius.com API
Here are the tracks we chose to analyze.

![image](https://raphaellederman.github.io/assets/images/tracks.png){:height="100%" width="100%"}

Then, using `lyricsgenius`, we retrieved the lyrics and applied some preprocessing in order to standardize the text data. We finally obtained a list of list containing each word of each song.

```python
api = genius.Genius('VGxZYl4kHnoBcj_hMiUA0DtweOQvySa8c7hi_fvyqbKd__3or_Lkn75yCG6_immb')

i = 0
for track in zip(tracks['Track Name'], tracks['Artist']) :
    try :
        song = api.search_song(str(track[0]), str(track[1]))
        song.save_lyrics('***/Hackathon/New_songs/' + str(track[1] + str(i)))
    except : 
        pass
    i = i + 1

files = sorted(glob(op.join('***/Hackathon/New_songs/', '*.txt')))
songs = [open(f).read() for f in files]

for i in range(0, len(songs)) :
    songs[i] = songs[i].replace("\n", " ").replace("\'", " ")
    songs[i] = re.sub(r"\[(.*?)\]", " ", songs[i])

cachedStopWords = stopwords.words("english")

words = []
filtered = []

for i in range(0, len(songs)) :
    words.append(re.split("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)", songs[i]))
    filtered.append(' '.join([word for word in songs[i].split() if word not in cachedStopWords]))

voc = []
voc_unique = []

for i in range(0, len(songs)) :
    voc.append(len(words[i]))
    voc_unique.append(len(filtered[i].split()))
```

We then displayed some meaningful features, for instance the average number of words (490), the average number of unique words (161) or the number of times each disctinct word should be repeated (3).

```python
plt.figure(figsize=(12,5))
plt.hist(np.array(voc), bins=40)
plt.title('Number of words in a top 2018 song')
plt.show()

length = [len(set(word)) for word in words]
print(round(np.array(length).mean(),2))

plt.figure(figsize=(12,5))
plt.hist(np.array(length), bins=40)
plt.title('Number of unique words in a top 2018 song')
plt.show()
```
![image](https://raphaellederman.github.io/assets/images/words1.png){:height="100%" width="100%"}

![image](https://raphaellederman.github.io/assets/images/words2.png){:height="100%" width="100%"}

Performing the same task on the preprocessed text data without stopwords produces a very different result : the number of words declines (45.5 unique words left on average).

```python
length_fil = [len(set(word)) for word in filtered]
round(np.array(length_fil).mean(),2

plt.figure(figsize=(12,5))
plt.hist(np.array(length_fil), bins=40)
plt.title('Number of unique words in a top 2018 song, without stopwords')
plt.show()
```
![image](https://raphaellederman.github.io/assets/images/words3.png){:height="100%" width="100%"}

In order to display the most important words, we decided to use a WordCloud representation.

```python
word_cloud = list(itertools.chain.from_iterable(words))

str1 = ' '.join(word_cloud)
stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(str(str1))

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```
![image](https://raphaellederman.github.io/assets/images/wordcould.png){:height="100%" width="100%"}

We then wanted to have a view on the tone of the lyrics, using an NLTK pre-trained model identifying the positivity and negativity of text data. We concluded that within our dataset, the average positivity was only around 32%. We also used another pre-trained model in order to detect emotions (anger, sadness, happiness and relax) : relax came as the most frequent, followed by anger.

```python
filename = 'model_sentiment_analysis.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(songs)
print(result.mean())

filename = 'sentiment_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(songs)
unique, counts = np.unique(result, return_counts=True)
```

## What are the features of the most popular songs ?

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
![image](https://raphaellederman.github.io/assets/images/corr.png){:height="100%" width="100%"}

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

![image](https://raphaellederman.github.io/assets/images/speechiness.png){:height="100%" width="100%"}

Among the most important features, we observe that the ideal tempo should be around 120 BPM, and the ideal duration should be around 213 seconds for a hit (using the music features values for the most popular songs of 2018).

## Can we predict the likelihood of a song becoming a hit ?

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
![image](https://raphaellederman.github.io/assets/images/importance.png){:height="100%" width="100%"}

We finally used XGBoost to predict the popularity of a song based on its features, but the overall training set was too small in order to obtain promising results (as we were short on time during the hackathon, we chose not to extand this analysis).

```python
clf = xgboost.XGBRegressor(colsample_bytree = 0.44, n_estimators=30000, learning_rate=0.07,max_depth=9,alpha = 5)
model = clf.fit(X_train.drop(worst_features, axis=1), y_train)
pred = model.predict(X_test.drop(worst_features, axis=1))
score_rf = metrics.r2_score(y_test, pred)
print(score_rf)
```

## What is the best time of the year to release a song ?

Our last thought was to determine the optimal release time in the year, using Spotify's API to gather release dates of hit songs : March seemed to be the best option.
 
```python 
s = pd.Series(data['month']).dropna()
fig, ax = plt.subplots(figsize = (12,12))
ax.hist(s, alpha=0.8, color='blue', bins = 25)
ax.xaxis.set_ticks(range(13))
ax.xaxis.set_ticklabels( [' ','Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre'])
plt.title("Historgram of the number of hit by month")
```
![image](https://raphaellederman.github.io/assets/images/release.png){:height="100%" width="100%"}


