---
published: true
title: Hackathon : what is the next hit of 2019 ? (1)
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

Here is a article describing a 10 hours Hackathon to which I participated with a team of 5 friends from Télécom ParisTech this year around the following theme : what will be the hit song of 2019 ?
I will provide some of the code that we write and the main outputs of this incredible experience

<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

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

![image](https://raphaellederman.github.io/images/tabh1.png){:height="100%" width="100%"}


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

![image](https://raphaellederman.github.io/images/US.png){:height="100%" width="100%"}

From this visualization, the choice was pretty clear : the USA represent by far the largest market, and we therefore chose to focus on this country.

The next step was to determine who were the most popular artists in terms on streams at the time. You can see the ranking on the next picture, XXXTentacion leading the scoreboard.

![image](https://raphaellederman.github.io/images/artist.png){:height="100%" width="100%"}

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

![image](https://raphaellederman.github.io/images/nationality.png){:height="100%" width="100%"}

Our next objective was to determine which style of music was the most popular using Google Trends. For this purpose, we extracted a CSV file covering the past 5 years comparing internet research for different styles. As you can see, Rap music seems to be the winner.

![image](https://raphaellederman.github.io/images/trends.png){:height="100%" width="100%"}


We then focused on the lyrics : we chose to analyze the lyrics of the top 200 songs of the Spotify charts using the Genius.com API
Here are the tracks we chose to analyze.

![image](https://raphaellederman.github.io/images/tracks.png){:height="100%" width="100%"}

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
![image](https://raphaellederman.github.io/images/words1.png){:height="100%" width="100%"}

![image](https://raphaellederman.github.io/images/words2.png){:height="100%" width="100%"}

Performing the same task on the preprocessed text data without stopwords produces a very different result : the number of words declines (45.5 unique words left on average).

```python
length_fil = [len(set(word)) for word in filtered]
round(np.array(length_fil).mean(),2

plt.figure(figsize=(12,5))
plt.hist(np.array(length_fil), bins=40)
plt.title('Number of unique words in a top 2018 song, without stopwords')
plt.show()
```
![image](https://raphaellederman.github.io/images/words3.png){:height="100%" width="100%"}

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
![image](https://raphaellederman.github.io/images/wordcould.png){:height="100%" width="100%"}

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