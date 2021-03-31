# -*- coding: utf-8 -*-
import nltk.data
import math
import sys
import json, ast
from collections import Counter,OrderedDict
from operator import itemgetter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, abort
from flask_cache import Cache
import pandas as pd
from textblob import TextBlob
import sys
import praw
from praw.models import MoreComments
import operator
import sqlite3
from flask import request
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import logging
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sentiment_analysis import *


reload(sys)
sys.setdefaultencoding('utf-8')
from collections import Counter
app = Flask(__name__)
#dash_app = dash.Dash(__name__,server=app, url_base_pathname='/dash')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})




@app.route("/")
def main():
        return render_template('index.html')



@app.route('/chart', methods=['POST','GET'])
@cache.cached(timeout=300)
def graphIt():
    conn = sqlite3.connect("./reddit_comments.db")
    #c = conn.cursor()
    total_count, avgscore, keys = fillLanguageSeries(conn)
    labels = []
    for k in keys:
        labels.append(k)

    # pie chart it up
    subreddit = createSubSeries()
    sarc, neg, pos, other = activeSubs(subreddit)

    # comments vs subreddits graph
    c = conn.cursor()
    c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment FROM allData ORDER BY Random(); ''')
    data = c.fetchall()

    data_df = pd.DataFrame(data)
    data_df.columns = ['comment', 'subreddit', 'score', 'sentiment']
    data_df['category_id'] = data_df['subreddit'].factorize()[0]
    category_id_df = data_df[['subreddit', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id','subreddit']].values)

    data_series = data_df.groupby('subreddit').comment.count()
    data_dict = data_series.to_dict()
    subreddits_list = []
    comment_count = []
    for key, value in data_dict.iteritems():
        subreddits_list.append(key)
        comment_count.append(value)

    return render_template('c3test.html', avgscore=avgscore, labels=labels, sarc=sarc, neg=neg, pos=pos,other=other, subreddits_list=subreddits_list, comment_count=comment_count)


@app.route('/wordcloud', methods=['GET', 'POST'])
@cache.cached(timeout=300)
def word_cloud():
    # get word cloud of frequent words per subreddit
    # perhaps have a way of user to pick which subreddit to visualize
    # use input from user to query subreddit
    conn = sqlite3.connect("reddit_comments.db")
    c = conn.cursor()
    word_df = wordcloud_words(c)
    df = word_df['Word']
    words = list(df.values.flatten())
    return render_template("wordcloud.html", words=words)

@app.route('/wordcloud_sub')
def word_cloud_query():
    return render_template("wordcloud_sub.html")

@app.route('/sub_wordclouds', methods=['POST', 'GET'])
@cache.cached(timeout=300)
def sub_wordclouds():
    if request.method == 'POST':
        result = request.form
        # gets subreddit, now to add this to sql query
        sub = str(result['mySubreddit'])
        conn = sqlite3.connect("reddit_comments.db")
        c = conn.cursor()
        string = "Subreddit"
        c.execute("SELECT commentBody FROM allData WHERE "+string+"=?",(sub,))
        sql = c.fetchall()
        conn.commit()
        sql = list(sql)
        w_list = subFreqWords(sql)
        return render_template("sub_wordclouds.html", sub=sub, w_list=w_list)
    elif request.method == 'GET':
        return render_template("sub_wordclouds.html")



@app.route('/machine_learning_sentence', methods=['POST','GET'])
def machine_learning_sentence():
    #test = request.json['sentence']
    sentence = request.json['sentence']
    prediction = predict_sentence(sentence)
    prediction = prediction.tolist()
    #app.logger.debug(request.json);
    return json.dumps(prediction)




def predict_sentence(sentence):
    # similar to machine_learning method but need to alter it to just predict for one sentence
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

    # SGD CLASSIFIER
    text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                            ('tfidf', TfidfTransformer()),
                            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])

                                        # (X_train, Y_train)
    text_clf_svm = text_clf_svm.fit(data_df.comment, data_df.subreddit)

    # for sentence
    sentence_list = []
    sentence_list.append(sentence)
    predicted_subreddit = text_clf_svm.predict(sentence_list)
    return predicted_subreddit


def get_predicted_subreddit(c,data_df):
    # test data
    c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment, created_at FROM allData WHERE LENGTH(commentBody) < 100 and LENGTH(commentBody) > 10 and commentScore < 5 ORDER BY Random() LIMIT 50000; ''')
    test_data = c.fetchall()

    test_data_df = pd.DataFrame(test_data)
    test_data_df.columns = ['comment', 'subreddit', 'score', 'sentiment', 'created_utc']

    # add categories
    test_data_df['category_id'] = test_data_df['subreddit'].factorize()[0]
    category_id_df = test_data_df[['subreddit', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id','subreddit']].values)

    # Naive Bayes
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])

    text_clf = text_clf.fit(data_df.comment, data_df.subreddit)
    # predict
    predicted = text_clf.predict(test_data_df.comment)
    prediction_accuracy = (np.mean(predicted == test_data_df.subreddit) * 100)

    # Linear SVM
    text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                            ('tfidf', TfidfTransformer()),
                            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])

                                        # (X_train, Y_train)
    text_clf_svm = text_clf_svm.fit(data_df.comment, data_df.subreddit)
                                        # X_test
    predicted_svm = text_clf_svm.predict(test_data_df.comment)

    svm_prediction_accuracy = (np.mean(predicted_svm == test_data_df.subreddit) * 100)

    return prediction_accuracy, svm_prediction_accuracy

def get_predicted_score(c,data_df):
    # test data
    c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment, created_at FROM allData WHERE LENGTH(commentBody) < 100 and LENGTH(commentBody) > 10 and commentScore < 5 ORDER BY Random() LIMIT 50000; ''')
    test_data = c.fetchall()

    test_data_df = pd.DataFrame(test_data)
    test_data_df.columns = ['comment', 'subreddit', 'score', 'sentiment', 'created_utc']

    # add categories
    test_data_df['category_id'] = test_data_df['subreddit'].factorize()[0]
    category_id_df = test_data_df[['subreddit', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id','subreddit']].values)

    # predicting score w AdaBoostClassifier
    vect = CountVectorizer(min_df=1)
    rnds = data_df.sample(frac=0.1, random_state=87824, axis=0)
    bow = vect.fit_transform(rnds['comment'])
    ab = AdaBoostClassifier(n_estimators=200)
    ab.fit(bow, rnds['score'])

    test_data = data_df.sample(frac = 0.1, random_state=824, axis=0)
    ab_prediction = ab.predict(vect.transform(test_data['comment']))
    ab_pred_acc = np.mean((ab_prediction == test_data.score) * 100)

    # with random forest classifier
    svc = LinearSVC()
    svc.fit(bow, rnds['score'])
    svc_test_data = data_df.sample(frac=0.1, random_state=424, axis=0)
    svc_pred = svc.predict(vect.transform(svc_test_data['comment']))
    svc_pred_acc = np.mean((svc_pred == svc_test_data.score) * 100)

    return ab_pred_acc, svc_pred_acc


@app.route('/machine_learning')
@cache.cached(timeout=300)
def machine_learning():
    # need to break into two for sub and score

    # training data
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


    prediction_accuracy, svm_prediction_accuracy = get_predicted_subreddit(c,data_df)
    ab_pred_acc, svc_pred_acc = get_predicted_score(c,data_df)

    return render_template("machine_learning.html", prediction_accuracy=prediction_accuracy, svm_prediction_accuracy=svm_prediction_accuracy, ab_pred_acc=ab_pred_acc, svc_pred_acc=svc_pred_acc)


def subFreqWords(sqlist):
    all_words = pd.DataFrame(sqlist)
    comm = all_words[0]
    comm = comm.str.lower().str.cat(sep=' ')
    allWords = nltk.tokenize.word_tokenize(comm)
    allWords = filter(lambda x: x.isalpha(), allWords)
    word_dist = nltk.FreqDist(allWords)
    stopwords = nltk.corpus.stopwords.words('english')
    stoplist = """https http there this let every looks people feel put things time said think go around since many point say said still time said used need play made must looks really go get got also text next see try lot first much use think know into like sure one two three four five six back take"""
    stopwords += stoplist.split()
    allWordsExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)
    rslt = pd.DataFrame(allWordsExceptStopDist.most_common(75), columns=['Word', 'Frequency'])
    words = rslt['Word']
    words_list = list(words.values.flatten())
    return words_list


def wordcloud_words(c):
    c.execute('''SELECT commentBody FROM allData;''')
    freq_words = c.fetchall()
    all_words = pd.DataFrame(freq_words)
    comm = all_words[0]
    comm = comm.str.lower().str.cat(sep=' ')
    allWords = nltk.tokenize.word_tokenize(comm)
    allWords = filter(lambda x: x.isalpha(), allWords)
    word_dist = nltk.FreqDist(allWords)
    stopwords = nltk.corpus.stopwords.words('english')
    stoplist = """ something good great right player going way could though want make even better well go https http there this let every looks people feel put would getting ca us things said think around since many point say said still time said used need play made must looks really go get got also text next see try lot first much use think know into like sure one two three four five six back take"""
    stopwords += stoplist.split()
    allWordsExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)
    rslt = pd.DataFrame(allWordsExceptStopDist.most_common(50), columns=['Word', 'Frequency'])


    return rslt



@app.route('/result', methods = ['POST', 'GET'])
@cache.cached(timeout=300) #cache this view for 5 mins
def result():
    conn = sqlite3.connect("reddit_comments.db")
    c = conn.cursor()
    if request.method == 'GET':
        sarc_data = detectSarcasm(c)
        freq_list = frequentPosters(c)
        freq_words = frequentWords(c)
        top_pos = get_top_pos_5(c)
        top_neg = get_top_neg_5(c)
        table_data = [[top_pos[i], top_neg[i]] for i in range(0, len(top_pos))]
        pos_comments = getPositiveComments(c)
        neg_comments = getNegativeComments(c)
        neutral = pos_comments - neg_comments
        return render_template("dashboard.html", neutral=neutral, neg_comments=neg_comments, pos_comments=pos_comments,sarc_data=sarc_data, freq_list=freq_list,freq_words=freq_words.to_html(index=False, classes="table table-sm"), table_data=table_data)


# need to make a separate route for this
@app.route('/subreddit_analysis', methods=['POST'])
def subredditAnalysis():
    reddit = praw.Reddit(client_id='SYLwpLpvyBQ4Xw', client_secret='QuyR-qIccKFkhKzdwr_ZMFdrkdw',
                          user_agent='subSentiment')
    conn = sqlite3.connect("reddit_comments.db")
    c = conn.cursor()
    if request.method == 'POST':
        '''use users input to gather 10 newest submissions '''
        result = request.form
        subreddit = str(request.form['Subreddit'])
        new_submissions, new_comments, total_neg, total_pos = getNewestPostsinSub(subreddit, conn)
        overall_sentiment_score = overall_sentiment(c)
        most_frequent_words = frequentWordsinSub(c)
        words_df = most_frequent_words['Word']
        words = list(words_df.values.flatten())
        active_user_dict = activePostersinSub(c)
        neutral = abs(total_pos - total_neg)

        total_comments, comment_avg, keys = language_style_subreddit(conn)
        labels = []
        for key in keys:
            labels.append(key)

    #    return render_template("querybysub.html", new_submissions, new_comments, total_neg, total_pos,subreddit=subreddit, overall_sentiment_score=overall_sentiment_score, most_frequent_words=most_frequent_words.to_html(index=False, classes="table table-sm"))
        return render_template("querybysub.html", neutral = neutral, new_submissions=new_submissions, new_comments=new_comments,total_pos=total_pos, total_neg=total_neg, subreddit=subreddit, overall_sentiment_score=overall_sentiment_score, words=words, active_user_dict=active_user_dict, labels=labels, comment_avg=comment_avg)


def language_style_subreddit(conn):
    passive_words = ['if you have the time','hmm','well','that was my fault','not sure', 'haha', 'yea',
                     'yeah', 'no', 'fine','aww',
                     'ok', 'okay', 'sorry', 'sure', 'thanks', 'thank you','funny','hilarious', 'cute']

    assertive_words = ['good idea','great idea','thanks for','good to know','really like', 'too',
                       'sorry for','I know', 'for sure',
                      'yes', 'and','smart','witty']

    aggressive_words = ['shot','fuck','motherfucker','fucking','ass','idiot','fuck off','stupid','but','dumb','lol'
                       'shit', 'bullshit', 'crap','loser','retard','faggot','bitch']

    meme_references = ['one does not simply','that was bad and you should feel bad','lousy smarch weather','hide the pain','cant believe youve done this','burgerking footlettuce', 'trololol','fuuu','dank memes','vape naysh','dab','power of god and anime','hewo','ma spaghet','ugandan knuckles','rage faces','me_irl','england is my city','salt bae','cash me outside','catch these hands']

    styles = pd.DataFrame({'Passive': pd.Series(passive_words),
                             'Assertive': pd.Series(assertive_words),
                             'Aggressive': pd.Series(aggressive_words),
                             'Meme References': pd.Series(meme_references),
                             'Sarcastic': pd.Series(['/s'])
                                    ## Reddit has a special tag '/s' widely used to identify sarcasm
                             })
    df = pd.read_sql("SELECT commentBody, User, commentScore, Sentiment FROM SUBREDDIT_DATA WHERE LENGTH(commentBody) > 5 AND LENGTH(commentBody) < 100",conn)

    style_scores = pd.DataFrame()
    contents = pd.DataFrame()
    if styles is not None:
        for style in styles:
            content = df[df.commentBody.apply(lambda x: any(word in x.split() for word in styles[style]))]
            style_scores[style] = content.describe().commentScore


    keys = style_scores.keys()
    summary = style_scores.transpose()
    total_num_comments = summary['count']   # the total number of comments per lang. style
    comment_avg_score = summary['mean']

    return total_num_comments, comment_avg_score, keys

def activePostersinSub(c):
    c.execute("SELECT User, COUNT(User)  FROM SUBREDDIT_DATA GROUP BY User HAVING COUNT(commentBody) > 0 ORDER BY COUNT(User) DESC")
    posters = c.fetchall()
    c.execute("SELECT commentBody FROM SUBREDDIT_DATA")
    total = c.fetchall()
    length = len(total)

    # add in case for if there no users who have not commented more than once
    top_posters = pd.DataFrame(posters)
    top_posters.columns = ['user', 'commentCount']
    print top_posters
    top_commenters = top_posters
    top_commenters = list(top_commenters.values.flatten())
    poster_freq = dict([(k, v) for k,v in zip (top_commenters[::2], top_commenters[1::2])])
    #print length
    user_activity = {}
    for key, value in poster_freq.iteritems():
        f_length = float(length)
        f_val = float(value)
        proportion_of_comments = f_val/f_length
        proportion = 100 * proportion_of_comments
        proportion = float(("%0.4f"%proportion_of_comments))
        user_activity.update({key : proportion})
    return user_activity


def overall_sentiment(c):
    c.execute("SELECT Sentiment FROM SUBREDDIT_DATA")
    sentiment_vals = c.fetchall()
    total_comments = len(sentiment_vals)
    total_sentiment = 0.0
    for value in sentiment_vals:
        value = value[0]
        total_sentiment += value

    rel_score = float(math.floor(total_sentiment/total_comments*100))
    return rel_score

def frequentWordsinSub(c):
    c.execute("SELECT commentBody FROM SUBREDDIT_DATA")
    freq_words = c.fetchall()
    all_words = pd.DataFrame(freq_words)
    comm = all_words[0]
    comm = comm.str.lower().str.cat(sep=' ')
    allWords = nltk.tokenize.word_tokenize(comm)
    allWords = filter(lambda x: x.isalpha(), allWords)
    word_dist = nltk.FreqDist(allWords)
    stopwords = nltk.corpus.stopwords.words('english')
    stoplist = """ something right player going way could though want make even go https http there this let every looks people feel put would getting ca us things said think around since many point say said still time said used need play made must looks really go get got also text next see try lot first much use think know into like sure one two three four five six back take"""
    stopwords += stoplist.split()
    allWordsExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)
    rslt = pd.DataFrame(allWordsExceptStopDist.most_common(30), columns=['Word', 'Frequency'])
    return rslt


def getNewestPostsinSub(sub, conn):
    reddit = praw.Reddit(client_id='SYLwpLpvyBQ4Xw', client_secret='QuyR-qIccKFkhKzdwr_ZMFdrkdw',
                          user_agent='subSentiment')

    # DROP TABLE IF EXISTS TABLE_NAME;
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS SUBREDDIT_DATA")
    c.execute(''' CREATE TABLE SUBREDDIT_DATA
            (commentBody text, User text, commentScore int, Sentiment float)''')

    sentiment_list = []
    count = 0
    neg_count = 0;
    pos_count = 0;
    neutral_count = 0;
    count_limit = 0

    if sub is " ":
        return "Invalid"
    # use a limit which is reasonable and wont take forever to process
    for submission in reddit.subreddit(sub).new(limit=5):
        comments = submission.comments.list()
        count_limit +=1

        for com in comments:
            if isinstance(com, MoreComments):
                continue

            comment = com.body
            score = com.score
            sentiment = getSentiment(com.body)
            sentiment_list.append(sentiment)
            author = com.author
            #print comment
            #print sentiment

            #print type(test)
            count +=1
            if author is None:
                username = "Null"
            else:
                username = author.name



            params = (comment, username, score, sentiment)
            c.execute("INSERT INTO SUBREDDIT_DATA VALUES(?,?,?,?)",params)
            conn.commit()


    new_submissions = count_limit
    new_comments = count

    for s in sentiment_list:
        if s < 0.0:
            neg_count += 1
        elif s > 0.0:
            pos_count += 1
        else:
            neutral_count += 1
    try:
        total_pos = 100 * float(pos_count)/float(count)
        total_pos = float(("%0.3f"%total_pos))
        #print "Out of those, %s were positive comments!" % total_pos

        total_neg = 100 * float(neg_count)/float(count)
        total_neg = float(("%0.3f"%total_neg))
        return new_submissions, new_comments, total_neg, total_pos
    except:
    #    print(("No comment sentiment") + '\n')
         ZeroDivisionError
    #print "Out of those, %s were negative comments!" % total_neg
    #print "The rest were neutral"











def detectSarcasm(c):
    #conn = sqlite3.connect("reddit_comments.db")
    #c = conn.cursor()
    c.execute('''SELECT commentBody, Subreddit from allData WHERE commentBody LIKE '% /s %';''')
    sarc_data = c.fetchall()
    c.execute('''SELECT commentBody from allData;''')
    total = c.fetchall()

    sarcasm = pd.DataFrame(sarc_data)
    sarc_count = len(sarcasm)
    count = pd.DataFrame(total)
    total_count = len(count)
    sarc_count - len(sarcasm)

    proportion = 100 * float(sarc_count)/float(total_count)
    proportion = float(("%0.4f"%proportion))
    return proportion




def getPositiveComments(c):
    c.execute('''SELECT commentBody, Sentiment FROM allData WHERE Sentiment > 0.0;''')
    pos_comments = c.fetchall()
    c.execute('''SELECT commentBody FROM allData;''')
    all_comments = c.fetchall()

    pos_count = pd.DataFrame(pos_comments)
    pos_count = len(pos_count)
    count = pd.DataFrame(all_comments)
    total_count = len(count)


    proportion = 100 * float(pos_count)/float(total_count)
    proportion = float(("%0.3f"%proportion))
    return proportion

def getNegativeComments(c):
    c.execute('''SELECT commentBody, Sentiment FROM allData WHERE Sentiment < 0.0;''')
    neg_comments = c.fetchall()
    c.execute('''SELECT commentBody FROM allData;''')
    all_comments = c.fetchall()

    neg_count = pd.DataFrame(neg_comments)
    neg_count = len(neg_count)
    count = pd.DataFrame(all_comments)
    total_count = len(count)


    proportion = 100 * float(neg_count)/float(total_count)
    proportion = float(("%0.3f"%proportion))
    return proportion






def frequentPosters(c):
    # this function is going to find the 10 most frequent commenters within the data
    c.execute(''' SELECT User FROM allData GROUP BY User HAVING COUNT(commentBody) > 10 AND User != 'AutoModerator' AND User != 'Null' ORDER BY COUNT(User) DESC;''')
    sql = c.fetchall()
    top_commenters = pd.DataFrame(sql)
    top_commenters.columns = ['user']
    top_10 = top_commenters[:10]
    top_10 = list(top_10.values.flatten())
    return top_10

# frequent words in whole dataset
def frequentWords(c):
    c.execute('''SELECT commentBody FROM allData;''')
    freq_words = c.fetchall()
    all_words = pd.DataFrame(freq_words)
    comm = all_words[0]
    comm = comm.str.lower().str.cat(sep=' ')
    allWords = nltk.tokenize.word_tokenize(comm)
    allWords = filter(lambda x: x.isalpha(), allWords)
    word_dist = nltk.FreqDist(allWords)
    stopwords = nltk.corpus.stopwords.words('english')
    stoplist = """ something good great right player going way could though want make even better well go https http there this let every looks people feel put would getting ca us things said think around since many point say said still time said used need play made must looks really go get got also text next see try lot first much use think know into like sure one two three four five six back take"""
    stopwords += stoplist.split()
    allWordsExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)
    rslt = pd.DataFrame(allWordsExceptStopDist.most_common(10), columns=['Word', 'Frequency'])
    #rslt.reset_index(level=0, drop=True, inplace=True)
    #rslt = rslt.set_index('Word', inplace=True)
    #rslt = ast.literal_eval(json.dumps(rslt))
    #r = rslt.set_index('Word').to_dict()

    return rslt


def get_top_pos_5(c):
    c.execute('''SELECT * from subsAndSent; ''')
    df = c.fetchall()
    data = pd.DataFrame(df)
    data.columns = ['Subreddit', 'Sentiment']

    subreddit_dictionary = {x[0]: x[1:] for x in data.itertuples(index=False)}

    # sort dictionary by max value
    sorted_dict = sorted(subreddit_dictionary.iteritems(), key=lambda (k,v): (v,k))
    ordered_dictionary = OrderedDict(sorted_dict)

    count = 0
    top_5_pos = []

    for k in reversed(ordered_dictionary):
        if count < 5:
            top_5_pos.append(k)
            count += 1

    return top_5_pos


def get_top_neg_5(c):
    c.execute(''' SELECT * from subsAndSent;''')
    df = c.fetchall()
    data = pd.DataFrame(df)

    data.columns = ['Subreddit', 'Sentiment']
    subreddit_dictionary = {x[0]: x[1:] for x in data.itertuples(index=False)}

    sorted_dict = sorted(subreddit_dictionary.iteritems(), key=lambda (k,v): (v,k))
    ordered_dictionary = OrderedDict(sorted_dict)

    count = 0
    top_5_neg = []
    for i in ordered_dictionary:
        if count < 5:
            top_5_neg.append(i)
            count += 1

    return top_5_neg




def fillLanguageSeries(sql_conn):

    #sql_conn = sqlite3.connect('./reddit_comments.db')

    passive_words = ['if you have the time','hmm','well','that was my fault','not sure', 'haha', 'yea',
                     'yeah', 'no', 'fine'
                     'ok', 'okay', 'sorry', 'sure', 'thanks', 'thank you','funny','hilarious']

    assertive_words = ['good idea','great idea','thanks for','good to know','really like', 'too',
                       'sorry for',
                      'yes', 'and','smart','witty']

    aggressive_words = ['shot','fuck','motherfucker','fucking','ass','idiot','fuck off','stupid','but','dumb','lol'
                       'shit', 'bullshit', 'crap','loser','retard','faggot','bitch']

    meme_references = ['one does not simply','that was bad and you should feel bad','lousy smarch weather','hide the pain','cant believe youve done this','burgerking footlettuce','ಠ_ಠ', 'trololol','fuuu','dank memes','vape naysh','dab','power of god and anime','hewo','ma spaghet','ugandan knuckles','rage faces','me_irl','england is my city','salt bae','cash me outside','catch these hands']

    styles = pd.DataFrame({'Passive': pd.Series(passive_words),
                             'Assertive': pd.Series(assertive_words),
                             'Aggressive': pd.Series(aggressive_words),
                             'Meme References': pd.Series(meme_references),
                             'Sarcastic': pd.Series(['/s'])
                                    ## Reddit has a special tag '/s' widely used to identify sarcasm
                             })

    df = pd.read_sql("""SELECT commentBody, Subreddit, User, commentScore, Sentiment
                     FROM allData WHERE LENGTH(commentBody) > 5 AND LENGTH(commentBody) < 100 LIMIT 1100000""", sql_conn)

    style_scores = pd.DataFrame()
    contents = pd.DataFrame()
    for style in styles:
        content = df[df.commentBody.apply(lambda x: any(word in x.split() for word in styles[style]))]
        style_scores[style] = content.describe().commentScore


    keys = style_scores.keys()
    summary = style_scores.transpose()
    total_num_comments = summary['count']   # the total number of comments per lang. style
    comment_avg_score = summary['mean']

    return total_num_comments, comment_avg_score, keys



def connDB():
    conn = sqlite3.connect("./reddit_comments.db")
    c = conn.cursor()
    return c

def createSubSeries():
    # get sarcastic
    c = connDB()
    c.execute('''SELECT Subreddit from allData WHERE commentBody LIKE '% /s %';''')
    sql = c.fetchall()
    sarcasm_subs = pd.DataFrame(sql)
    sarcasm_subs = sarcasm_subs.drop_duplicates()
    sarc_list = sarcasm_subs[0].tolist()

    # get positive
    c.execute('''SELECT * from subsAndSent; ''')
    df = c.fetchall()
    data = pd.DataFrame(df)
    data.columns = ['Subreddit', 'Sentiment']

    subreddit_dictionary = {x[0]: x[1:] for x in data.itertuples(index=False)}

    # sort dictionary by max value
    sorted_dict = sorted(subreddit_dictionary.iteritems(), key=lambda (k,v): (v,k))
    ordered_dictionary = OrderedDict(sorted_dict)

    count = 0
    top_5_pos = []

    for k in reversed(ordered_dictionary):
        if count < 5:
            top_5_pos.append(k)
            count += 1

    # get neg
    c.execute(''' SELECT * from subsAndSent;''')
    df = c.fetchall()
    data = pd.DataFrame(df)

    data.columns = ['Subreddit', 'Sentiment']
    subreddit_dictionary = {x[0]: x[1:] for x in data.itertuples(index=False)}

    sorted_dict = sorted(subreddit_dictionary.iteritems(), key=lambda (k,v): (v,k))
    ordered_dictionary = OrderedDict(sorted_dict)

    count = 0
    top_5_neg = []
    for i in ordered_dictionary:
        if count < 5:
            top_5_neg.append(i)
            count += 1

    # create dataframe
    subreddit = pd.DataFrame({'Sarcastic': pd.Series(sarc_list),
                                'Positive': pd.Series(top_5_pos),
                                'Negative': pd.Series(top_5_neg)})

    return subreddit


def activeSubs(subreddit):
        # correlate users to subs --- may change to maybe 5
    c = connDB()
    c.execute('''SELECT User,Subreddit FROM allData GROUP BY User HAVING COUNT(commentBody) > 10 AND User != 'AutoModerator' AND User != 'Null' ORDER BY Subreddit ASC;''')
    sql_cor = c.fetchall()

    user_sub_corr = pd.DataFrame(sql_cor)
    user_sub_corr.sort_values(1)
    dropped_user = user_sub_corr.drop(0,axis=1)
    dropped_user.columns = ['Most Active Commenters']
    # count duplicates, and find the 5 with most duplicates (most posters)
    # find the subs with the most counts
    size = dropped_user.groupby(dropped_user.columns.tolist(),as_index=False).size()

    active_subs = size.to_dict()
    sorted_active_subs = sorted(active_subs.iteritems(), key=lambda (k,v): (v,k))
    ordered_active_subs = OrderedDict(sorted_active_subs)

    # find top 5 active subs
    top_active = []
    count = 0
    for a in reversed(ordered_active_subs):
        if count < 5:
            top_active.append(a)
            count += 1



        # compare top_active to other df
        # find if the majority of top_active falls into pos,neg or sarc
    top_active = pd.DataFrame(top_active)
    top_active.columns = ['Most Active']


    sarc_match = top_active['Most Active'].map(subreddit['Sarcastic'].value_counts())
    neg_match = top_active['Most Active'].map(subreddit['Negative'].value_counts())
    pos_match = top_active['Most Active'].map(subreddit['Positive'].value_counts())


    sarc_perc = sarc_match.dropna(axis=0,how='all')
    sarctotal = float(len(sarc_perc))/float(len(sarc_match))

    sarc_percentage = sarctotal*100


    neg_per = neg_match.dropna(axis=0,how='all')
    negtotal = float(len(neg_per))/float(len(neg_match))

    negative_percentage = negtotal*100

    pos_perc = pos_match.dropna(axis=0,how='all')
    postotal = float(len(pos_perc))/float(len(pos_match))

    pos_percentage = postotal*100

    other = sarc_percentage + negative_percentage + pos_percentage
    other = 100.0 - other


    return sarc_percentage, negative_percentage, pos_percentage, other



# for error handling
@app.errorhandler(404)
def page_not_found(e):
    return render_template("errors/404.html")

@app.errorhandler(500)
def page_not_found(e):
    return render_template("errors/500.html")

@app.errorhandler(403)
def page_not_found(e):
    return render_template("errors/403.html")



if __name__ == '__main__':
    app.run(debug=False)
