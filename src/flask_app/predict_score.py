# -*- coding: utf-8 -*-


# machine learning attempt at predicting score < 5

import sqlite3
import pandas as pd
import numpy as np
import scipy.sparse

import codecs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, explained_variance_score, mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import Ridge
import nltk
from sklearn.ensemble import RandomForestClassifier

def predict(model, train_data, train_target, test_data, test_target):
    if(model == "GNB"):
        gnb_classifier = GaussianNB()
        X = train_data.todense()
        Y = train_target.values
        gnb_classifier.fit(X, Y)
        test_dense = test_data.todense()
        gnb_predict = gnb_classifier.predict(test_dense)
        print "GNB ACCURACY: ",(float(accuracy_score(test_target, gnb_predict)))
    elif(model == "NB"):
        classifier = MultinomialNB()
        classifier.fit(train_data, train_target)
        predict = classifier.predict(test_data)
        print "NB ACCURACY: ",(float(accuracy_score(test_target,predict)))
    elif(model == "SVM"):
        classifier = LinearSVC()
        classifier.fit(train_data, train_target)
        predict = classifier.predict(test_data)
        print "SVM ACCURACY: ",(float(accuracy_score(test_target,predict)))
    elif(model == "BRN"):
        classifier = GaussianNB()
        X = train_data.todense()
        Y = train_target.values
        classifier.fit(X,Y)
        test_dense = test_data.todense()
        predict = classifier.predict(test_dense)
        print "BRN ACCURACY: ", (float(accuracy_score(test_target, predict)))
    elif(model == "RFC"):
        classifier = RandomForestClassifier()
        X = train_data.todense()
        Y = train_target.values
        classifier.fit(X, Y)
        test_dense = test_data.todense()
        rfc_predict = classifier.predict(test_dense)
        print "RFC ACCURACY: ",(float(accuracy_score(test_target, rfc_predict)))

    '''elif(model == "SGD"):
        classifier = linear_model.SGDRegressor()
        classifier.fit(train_data, train_target)
        predict = classifier.predict(test_data)
        print(model)
        print "MAE: ",(mean_absolute_error(test_target, predict))
    elif(model == "RDG"):
        classifier = Ridge(alpha = 0.5)
        classifier.fit(train_data, train_target)
        predict = classifier.predict(test_data)
        print(model)
        print "MAE: ",(mean_absolute_error(test_target, predict))'''









        #classifier.fit(X[0].values,Y[0].values)
        #predict = classifier.predict(test_data)



    '''print(model)
    print(float(accuracy_score(test_target, predict)))'''
    #print(confusion_matrix(test_target, predict))
    #print(precision_score(test_target, predict, average=None))
    #print(recall_score(test_target, predict, average=None))
    '''print("Accuracy: %f" % float(accuracy_score(test_target,predict)))
    print("Precision: %f" % float(precision_score(test_target,predict, average=None)))
    print("Recall: %f" % float(recall_score(test_target,predict)))'''


def wordsInComments():
    # which words appear in successful comments?
    succ_comments = data.loc[data['commentScore'] < 21]
    cv = CountVectorizer(stop_words="english",min_df=2, ngram_range=(1,3))
    cv.fit_transform(succ_comments['commentBody'])
    return cv.vocabulary_


conn = sqlite3.connect("reddit_comments.db")
sql_cmd = "SELECT commentBody, Subreddit, commentScore FROM allData ORDER BY Random() LIMIT 10000"

data = pd.read_sql(sql_cmd, conn)
utf8 = [codecs.encode(body, 'utf-8') for body in data.commentBody]
data.commentBody = pd.Series(utf8)
#print(data.describe())

# Use words that appear in comments < 5
vocab = wordsInComments()
vectorizer = CountVectorizer(vocabulary=vocab)
body_terms = vectorizer.fit_transform(data['commentBody'])

# Add subreddit as a feature
dv = DictVectorizer()
subredditDict = data[['Subreddit']].T.to_dict().values()
subredditFeatures = dv.fit_transform(subredditDict)
features = scipy.sparse.hstack([body_terms, subredditFeatures])

#svd = TruncatedSVD(n_components=3)
#body_words = svd.fit_transform(body_terms)

text_train, text_test, target_train, target_test = train_test_split(features, data['commentScore'])

predict("NB", text_train, target_train, text_test, target_test)
predict("SVM", text_train, target_train, text_test, target_test)
#predict("SGD", text_train, target_train, text_test, target_test)
#predict("RDG", text_train, target_train, text_test, target_test)
predict("GNB", text_train, target_train, text_test, target_test)
predict("BRN", text_train, target_train, text_test, target_test)
predict("RFC", text_train, target_train, text_test, target_test)
