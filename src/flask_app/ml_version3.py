# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge

# getting data
conn = sqlite3.connect("./reddit_comments.db")
c = conn.cursor()

c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment, created_at FROM allData WHERE LENGTH(commentBody) < 100 and LENGTH(commentBody) > 10 and commentScore < 5  ORDER BY Random() LIMIT 50000; ''')
data = c.fetchall()

data_df = pd.DataFrame(data)
data_df.columns = ['comment', 'subreddit', 'score', 'sentiment', 'created_utc']
#print data_df.head()


#print data_df.shape


# Use CountVectorizer to transform the comments into a matrix
# matrix = (i,j) - the element of this matrix will be the number of times the jth word appears
# in the ith comment
# data is large so we will only sample 10% of it
vect = CountVectorizer(min_df=1)
rnds = data_df.sample(frac = 0.1, random_state=87824, axis=0)
bow = vect.fit_transform(rnds['comment'])

print (vect.transform(['asdfadj']).count_nonzero())
print (vect.transform(['great']).count_nonzero())

# now we have established the BoW, we can use AdaBoost to classify and predict
# Using Adaboost because the data is sparse - so we need a classifier
# that has a high bias bc most classifiers overfit sparse data
# Adaboost is quick and tends to not overfit

ab = AdaBoostClassifier(n_estimators=200)
rf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
rdg = Ridge(alpha=1.0)
nb = MultinomialNB()
lvc = LinearSVC()
rf.fit(bow, rnds['score'])
ab.fit(bow, rnds['score'])
nb.fit(bow, rnds['score'])
rdg.fit(bow, rnds['score'])
lvc.fit(bow, rnds['score'])

# test the data on a different random sample (the test data )
test_data = data_df.sample(frac = 0.1, random_state=824, axis=0)
prediction = ab.predict(vect.transform(test_data['comment']))

print ("Accuracy: " + str(100*sum(prediction == test_data['score'].values)/len(prediction)) + '%')
#print ("F1      : " + str(f1_score(test_data['score'], prediction, average="micro")))
print " "

test_data_svc = data_df.sample(frac = 0.1, random_state=42, axis=0)
pred_svc = lvc.predict(vect.transform(test_data_svc['comment']))

print ("LSCV Accuracy: " + str(100*sum(pred_svc == test_data['score'].values)/len(pred_svc)) + '%')
print " "
test_data = data_df.sample(frac = 0.1, random_state=814, axis=0)
rf_predition = rf.predict(vect.transform(test_data['comment']))
print ("RF Accuracy: " + str(100*sum(rf_predition == test_data['score'].values)/len(rf_predition)) + '%')


test_data = data_df.sample(frac = 0.1, random_state=422, axis=0)
nb_predition = nb.predict(vect.transform(test_data['comment']))
print ("NB Accuracy: " + str(100*sum(nb_predition == test_data['score'].values)/len(nb_predition)) + '%')
'''
test_data = data_df.sample(frac= 0.1, random_state=420, axis=0)
rdg_pred = rdg.predict(vect.transform(test_data['comment']))
print("Ridge Accuracy: " + str(100*sum(prediction == test_data['score'].values)/len(prediction)) + '%')'''
