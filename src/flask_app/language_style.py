import nltk.data
import sqlite3
from sentiment_analysis import *
from praw.models import MoreComments
import math
import pandas as pd
import praw
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


'''
Analyis per subreddit:
- get data (newest 100 submissions)
- popular language style
- overall sentiment per subreddit


'''

def main():


    conn = sqlite3.connect("./reddit_comments.db")
    conn.text_factory = str
    cur = conn.cursor()
    sub = raw_input("Enter a subreddit: ")
    getNew(sub, conn)
    print overall_sentiment(cur,sub)
    language_style(conn,sub)

    #rel_score = overall_sentiment(cur,sub)

    #result = frequent_words(cur, sub)
    #print result


def getNew(sub,conn):
    reddit = praw.Reddit(client_id='SYLwpLpvyBQ4Xw', client_secret='QuyR-qIccKFkhKzdwr_ZMFdrkdw',
                      user_agent='subSentiment')
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS "+sub+" (commentBody text, User text, commentScore int, Sentiment float)")

    sentiment_list = []
    count = 0
    neg_count = 0;
    pos_count = 0;
    neutral_count = 0;
    count_limit = 0

    # nb perhaps dont set limit to 500
    for submission in reddit.subreddit(sub).new(limit=10):
        #subreddit = reddit.subreddit(sub)
        #name = str(subreddit.display_name)
        comments = submission.comments.list()
        count_limit +=1
        #print count_limit

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
            cur.execute("INSERT INTO "+sub+" VALUES(?,?,?,?)",params)
            conn.commit()



    print "There are %s new comments!" % count
    print "There are %s new submissions" % count_limit

    for s in sentiment_list:
        if s < 0.0:
            neg_count += 1
        elif s > 0.0:
            pos_count += 1
        else:
            neutral_count += 1

    total_pos = 100 * float(pos_count)/float(count)
    total_pos = float(("%0.3f"%total_pos))
    print "Out of those, %s were positive comments!" % total_pos

    total_neg = 100 * float(neg_count)/float(count)
    total_neg = float(("%0.3f"%total_neg))
    print "Out of those, %s were negative comments!" % total_neg
    print "The rest were neutral"


    '''
    #########################################
    recycle frequent posters & language style
    #########################################
    '''

def activePosters(c, sub):
    #   c.execute("SELECT * FROM "+sub+" ORDER BY Sentiment ASC")
    #    sql = c.fetchall()

    ''' ugh fix this somehow lol '''
    # frequent posters
    c.execute("SELECT User, COUNT(User)  FROM "+sub+" GROUP BY User HAVING COUNT(commentBody) > 1 ORDER BY COUNT(User) DESC")
    posters = c.fetchall()
    c.execute("SELECT commentBody FROM "+sub+"")
    total = c.fetchall()
    length = len(total)

    top_posters = pd.DataFrame(posters)
    print top_posters
    top_posters.columns = ['user', 'commentCount']
    top_5 = top_posters
    top_5 = list(top_5.values.flatten())
    poster_freq = dict([(k, v) for k,v in zip (top_5[::2], top_5[1::2])])
    #print length
    #print poster_freq
    user_activity = {}
    for key, value in poster_freq.iteritems():
        #print key
        #print value
        #print length
        #print value
        f_length = float(length)
        f_val = float(value)
        proportion_of_comments =  f_val/f_length
        proportion = float(("%0.4f"%proportion_of_comments))
        user_activity.update({key : proportion})
        #print (key + ":" + 100 * proportion)
        print ("{0} : {1}%".format(key, 100 * proportion))
    print user_activity

    return 0

def overall_sentiment(c,sub):
    ''' overall sentiment in the sub '''
    c.execute("SELECT Sentiment FROM "+sub+"")
    sentiment_vals = c.fetchall()
    #print type(sentiment_vals)
    #sentiment_df = pd.DataFrame(sentiment_vals)
    total_comments = len(sentiment_vals)
    #print total_comments
    total_sentiment = 0.0
    for value in sentiment_vals:
        value = value[0]
        total_sentiment += value

    rel_score = float(math.floor(total_sentiment/total_comments*100))
    print "Overall sentiment of %s:" % sub, rel_score
    return rel_score



def frequent_words(c, sub):
    c.execute("SELECT commentBody FROM "+sub+"")
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
    return rslt


def language_style(sql_conn, sub):
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
    #c = sql_conn.cursor()
    #c.execute("SELECT commentBody, User, commentScore, Sentiment FROM "+sub+" WHERE LENGTH(commentBody) > 5 AND LENGTH(commentBody) < 100")
    #data = c.fetchall()
    #df = pd.DataFrame(data)
    df = pd.read_sql("SELECT commentBody, User, commentScore, Sentiment FROM "+sub+" WHERE LENGTH(commentBody) > 5 AND LENGTH(commentBody) < 100",sql_conn)

    style_scores = pd.DataFrame()
    contents = pd.DataFrame()
    for style in styles:
        content = df[df.commentBody.apply(lambda x: any(word in x.split() for word in styles[style]))]
        style_scores[style] = content.describe().commentScore


    keys = style_scores.keys()
    summary = style_scores.transpose()
    total_num_comments = summary['count']   # the total number of comments per lang. style
    comment_avg_score = summary['mean']

    print total_num_comments, comment_avg_score, keys


if __name__ == '__main__':
    main()
