# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sqlite3
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import Ridge
import random

conn = sqlite3.connect("./reddit_comments.db")
c = conn.cursor()

c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment, created_at FROM allData WHERE LENGTH(commentBody) < 25 ORDER BY Random() LIMIT 10000; ''')
data = c.fetchall()

df_train = pd.DataFrame(data)
df_train.columns = ['comment', 'subreddit', 'score', 'sentiment', 'created_utc']
# add categories
df_train['category_id'] = df_train['subreddit'].factorize()[0]
category_id_df = df_train[['subreddit', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','subreddit']].values)

#print df_train[:3]
# find all unique words in comment
comments = df_train['comment']
unique_words = list(set(" ".join(comments).split(" ")))
'''def make_matrix(comments, vocab):
    matrix = []
    for comment in comments:
        # count each word in the comment & make a ditionary
        counter = Counter(comment)
        # turn dictionary into matrix row using vocab
        row = [counter.get(w,0) for w in vocab]
        matrix.append(row)
    df = pd.DataFrame(matrix)
    df.columns = unique_words
    return df


#print(make_matrix(comments, unique_words)) -> sparse matrix (i.e a lot of 0s)

# Lowercase, then replace any non-letter, space or digit character in the headlines
new_comments = [re.sub(r'[^\w\s\d]','',c.lower()) for c in comments]
# replace sequences of whitespace with a space character
new_comments = [re.sub("\s+", " ", c) for c in new_comments]

unique_words = list(set(" ".join(new_comments).split(" ")))'''

#print(make_matrix(new_comments, unique_words))

# removing stopwords
'''with open("stopwords.txt", 'r') as f:
    stopwords = f.read().split("\n")

    # punctuation replacement
    stopwords = [re.sub(r'[^\w\s\d]','',s.lower()) for s in stopwords]
    unique_words = list(set(" ".join(new_comments).split(" ")))
    # remove stopwords from vocab
    unique_words = [w for w in unique_words if w not in stopwords]'''

    #print (make_matrix(new_comments, unique_words))


# construct bag of words matrix
# lowecase everything & ignore punctuation
# remove stop words


vectorizer = CountVectorizer(lowercase=True, stop_words="english")
matrix = vectorizer.fit_transform(comments)
#print(matrix.todense())

full_matrix = vectorizer.fit_transform(comments)
#print(full_matrix.shape)

# must reduce dimensionality in order to speed it up
# pick a subset of columns that are most informative  can use the chi-squared test
# create upvotes & downvotes column
up_list = []
for value in df_train['score']:
    if value > 3:
        up_list.append(value)

down_list = []
for v in df_train['score']:
    if v <= 0:
        down_list.append(v)

values = pd.Series(up_list)
df_train.insert(loc=4, column="upvotes", value=values)
#rint df_train.head()

down_values = pd.Series(down_list)
df_train.insert(loc=5, column="downvotes", value=down_values)
#print df_train.head()

# add in meta features
transform_functions = [
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count("."),
    lambda x: x.count("!"),
    lambda x: x.count("?"),
    lambda x: len(x) / (x.count(" ") + 1),
    lambda x: x.count(" ") / (x.count(".") + 1),
    lambda x: len(re.findall("\d", x)),
    lambda x: len(re.findall("[A-Z]", x)),
]

# apply each function & put the results into a list
columns = []
for func in transform_functions:
    columns.append(df_train["comment"].apply(func))

# convert meta features to a numpy array
meta = np.asarray(columns).T
features = np.hstack([meta, full_matrix.todense()])

train_rows = 7500
random.seed(1)

indices = list(range(features.shape[0]))
random.shuffle(indices)

# create train and test sets
train = features[indices[:train_rows],:]
test = features[indices[train_rows:],:]
train_upvotes = df_train['score'].iloc[indices[:train_rows]]
test_upvotes = df_train['score'].iloc[indices[train_rows:]]
train = np.nan_to_num(train)

reg = Ridge(alpha=.1)
reg.fit(train, train_upvotes)
predictions = reg.predict(test)

# use MAE
print(sum(abs(predictions - test_upvotes)) / len(predictions))
average_score = sum(test_upvotes)/len(test_upvotes)
print(sum(abs(average_score - test_upvotes)) / len(predictions))
