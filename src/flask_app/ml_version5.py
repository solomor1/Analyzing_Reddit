# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from io import StringIO
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
from IPython.display import display
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot
import plotly.plotly as py
import plotly.graph_objs as go


conn = sqlite3.connect("./reddit_comments.db")
c = conn.cursor()

c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment, created_at FROM allData WHERE LENGTH(commentBody) < 100 and LENGTH(commentBody) > 10 and commentScore < 5 ORDER BY Random() LIMIT 50000; ''')
data = c.fetchall()

data_df = pd.DataFrame(data)
data_df.columns = ['comment', 'subreddit', 'score', 'sentiment', 'created_utc']

# add categories
data_df['category_id'] = data_df['subreddit'].factorize()[0]
category_id_df = data_df[['subreddit', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','subreddit']].values)
#print data_df[:]


# convert text to numerical feature vectors using BoW


'''count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data_df.comment)
print X_train_counts.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape

clf = MultinomialNB().fit(X_train_tfidf, data_df.subreddit)'''

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])

text_clf = text_clf.fit(data_df.comment, data_df.subreddit)



c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment, created_at FROM allData WHERE LENGTH(commentBody) < 100 and LENGTH(commentBody) > 10 and commentScore < 5 ORDER BY Random() LIMIT 50000; ''')
test_data = c.fetchall()

test_data_df = pd.DataFrame(test_data)
test_data_df.columns = ['comment', 'subreddit', 'score', 'sentiment', 'created_utc']

# add categories
test_data_df['category_id'] = test_data_df['subreddit'].factorize()[0]
category_id_df = test_data_df[['subreddit', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','subreddit']].values)


#print test_data_df[:5]
#print data_df[:5]

predicted = text_clf.predict(test_data_df.comment)
print "NAIVE BAYES: ", np.mean(predicted == test_data_df.subreddit)
'''con_matrix = confusion_matrix(test_data_df.subreddit, predicted)
fig, ax = plt.subplots()
sns.heatmap(con_matrix, annot=True, fmt='d', xticklabels=category_id_df.subreddit.values, yticklabels=category_id_df.subreddit.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')'''
#plt.show()


text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                        ('tfidf', TfidfTransformer()),
                        ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])

                                    # (X_train, Y_train)
text_clf_svm = text_clf_svm.fit(data_df.comment, data_df.subreddit)
                                    # X_test
predicted_svm = text_clf_svm.predict(test_data_df.comment)      #Y_test
print "LINEAR SVM (SGD CLASSIFIER): ", np.mean(predicted_svm == test_data_df.subreddit)
con_matrix = confusion_matrix(test_data_df.subreddit, predicted_svm)
fig, ax = plt.subplots()

'''sns.heatmap(con_matrix, annot=False, fmt='d', xticklabels=category_id_df.subreddit.values, yticklabels=category_id_df.subreddit.values, vmin=20, cmap="YlGnBu")
plt.xticks(rotation=45)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()'''


heatmap = sns.heatmap(con_matrix, annot=True, fmt="d", vmin=(-5.0))
heatmap.yaxis.set_ticklabels(category_id_df.subreddit.values, rotation=0, ha='right', fontsize=10)
heatmap.xaxis.set_ticklabels(category_id_df.subreddit.values, rotation=45, ha='right', fontsize=10)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


vect = CountVectorizer(min_df=1)
rnds = data_df.sample(frac=0.1, random_state=87824, axis=0)
bow = vect.fit_transform(rnds['comment'])

ab = AdaBoostClassifier(n_estimators=200)
ab.fit(bow, rnds['score'])

test_data = data_df.sample(frac = 0.1, random_state=824, axis=0)
prediction = ab.predict(vect.transform(test_data['comment']))

print "ADA BOOST for SCORE:", np.mean(prediction == test_data.score)



''' add in way for user interaction ie narrowing down by sub '''
# predict subreddit from user input comment based of of how the model has been trained
