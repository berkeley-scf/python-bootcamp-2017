# solutions below are based in part on:
# wget http://jarrodmillman.com/capstone/code/senators.py
# wget http://jarrodmillman.com/capstone/code/senators2.py

import json
from operator import itemgetter
import re
import numpy as np
import pandas as pd

# 1. Load the *senators-list.json* and *timelines.json* files as objects called *senators* and *timelines*.

# using 'with' here closes the file automatically
with open("senators-list.json") as f:
    senators = json.load(f)

with open("timelines.json") as f:
    timelines = json.load(f)

# 2. What type of datastructure is *timelines*? How many timelines are there? What does each timeline correspond to?

type(timelines)
type(timelines[0])
type(timelines[0][0])

len(timelines)  # 100 senators
len(timelines[0]) # 200 tweets
timelines[0][0]

timelines[0][0].keys()
timelines[0][0]["text"]

timelines[0][0]["user"].keys()
timelines[0][0]["user"]['screen_name']
timelines[0][0]["user"]["followers_count"]

# 3. Make a list of the number of followers each senator has.

popularity = [(s['name'], s['followers_count']) for s in senators['users']]


# 4. What is the screen name of the senator with the largest number of followers.

popularity.sort()  # this will only work if you have 'followers_count' as the first element of the tuples in popularity

# this works as is, but is a bit advanced
popularity.sort(key = lambda x: x[1], reverse = True)

popularity[0]
popularity[0:10]

# alternatively, put the info in a pandas DataFrame

import pandas as pd
popularity = pd.DataFrame(popularity).rename(index = str, columns = {0:'name',1:'followers_count'})
popularity.sort_values('followers_count', ascending = False)

# 5. Make a list of lists where the outer list represents senators and the inner list contains each senator's tweets, and call it *tweets*.

# approach #1: list comprehension within a loop
tweets = []
for timeline in timelines:  # loop over senators
    tweets.append([tweet['text'] for tweet in timeline])  # list comprehension to loop over a senator's tweets

# approach #2: double list comprehension
tweets = [ [tweet['text'] for tweet in timeline] for timeline in timelines]

# approach #3: double for loop
tweets = []
for timeline in timelines:  # loop over senators
    tmp = []
    for tweet in timeline:
        tmp.append([tweet['text'])  

# 6. Write a function that takes tweet and returns a cleaned up version of the tweet. Here is some example code to get you started:

def remove_punct(tweet):
    tweet = re.sub('[.,;?!)(:#]', '', tweet)
    return(tweet)

# 7. Write a function that takes tweet and returns a cleaned up version of the tweet. 

def clean(tweet):
     cleaned_words = [remove_punct(word).lower() for word in tweet.split() if
                      'http' not in word and
                      remove_punct(word).isalpha() and
                      word != 'RT' and
                      not re.match('^[&@]', word)]
     return ' '.join(cleaned_words)

# 8. Use the following file to create a list, called *stopwords*, that contains common english words.  <http://www.textfixer.com/resources/common-english-words.txt>

import requests

url = requests.get('http://www.textfixer.com/resources/common-english-words.txt')
stopwords = url.text.split(',')

# 9. Write a function, called *tokenize*, which takes a tweet, cleans it, and removes all punctuation and stopwords.

def tokenize(tweet):
    cleaned = clean(tweet)
    content = [word for word in cleaned.split(' ') if word not in stopwords]
    return(content)

# 10. Create a list of lists, tweets_content, using your *tokenize* function.

tweets_content = [[ tokenize(tweet) for tweet in senator] for senator in tweets]

# 11. Create a list, *tokens*, where each senator's tweets are made into a single string.

# approach #1: join tokens within each tweet to a single string, then join resulting strings
tokens = [" ".join([" ".join(tweet) for tweet in senator]) for senator in tweets_content]
# approach #2
tokens = [sum(senator, []) for senator in tweets_content]

# 12.) Create a Pandas dataFrame with the following columns: senator name or handle, party of the senator, and number of times a prominent politician is mentioned in each senator's tweets. You might count the number of 'Obama', 'Trump', or 'Clinton' references.

focus = 'obama' # or 'trump' or 'clinton'

counts = np.array([x.count(focus) for x in tokens])  # for obama note this also counts 'obamacare'

# 1 is Republican
party = np.array([1,0,0,1,1,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1])

mydf = pd.DataFrame({'party': party, 'counts': counts})

# 13.) Use a Poisson GLM to assess the relationship between party and number of Obama/Trump/Clinton mentions. Does one party tend to mention Obama/Trump/Clinton more in their tweets?

import statsmodels.api as sm
model = sm.GLM(endog = mydf.counts, exog = sm.tools.add_constant(mydf.party), family = sm.families.Poisson())
results = model.fit()
results.summary()

# simple count to compare to stat model result
for j in range(2): 
    print(sum(counts[i] for i in range(100) if party[i] == j))

mydf.to_csv('/tmp/counts.csv')

# 14.) Use *matplotlib* to make histograms of the number of Obama mentions by senator, stratified by party.

import matplotlib.pyplot as plt
plt.hist(counts, 20, normed=0)
plt.show()

plt.subplot(121)
plt.hist([counts[i] for i in range(100) if party[i] == 1], 20, normed=1)
plt.title('Republican')
plt.subplot(122)
plt.hist([counts[i] for i in range(100) if party[i] == 0], 20, normed=1)
plt.title('Democrat')
plt.show()


