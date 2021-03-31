# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import praw
from textblob import TextBlob
import math
from praw.models import MoreComments
from sentiment_analysis import *
import io
import unicodedata


reddit = praw.Reddit(client_id='SYLwpLpvyBQ4Xw', client_secret='QuyR-qIccKFkhKzdwr_ZMFdrkdw',
                      user_agent='subSentiment')
''' TO DO:
Experiment querying by subreddit - see how fast it is to gather comments for user given subreddit
add to database -- done
can recycle language style
and freq posters and posters vs types of comments (pos/neg)
overall sentiment per subreddit -- done
'''

conn = sqlite3.connect("./reddit_comments.db")
conn.text_factory = str
cur = conn.cursor()
sub = raw_input("Enter a subreddit: ")
cur.execute("CREATE TABLE %s (commentBody text, Subreddit text, User text, commentScore int, Sentiment float)",sub)


sentiment_list = []
count = 0
neg_count = 0;
pos_count = 0;
neutral_count = 0;
count_limit = 0
# user input subreddit
#sub = raw_input("Enter a subreddit: ")
# for submission in reddit.subreddit(sub).new(limit=50)
for submission in reddit.subreddit("LoseIt").new(limit=50):
    #subreddit = reddit.subreddit(sub)
    #name = str(subreddit.display_name)
    comments = submission.comments.list()
    count_limit +=1
    print count_limit

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

        # add to database
        # can recycle language style
        # and freq posters and posters vs types of comments (pos/neg)
        # overall sentiment per subreddit

        params = (comment, username, score, sentiment)



print "%s new submissions have been posted to this sub so far today" % count

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
        #submission.comment_sort = "/top"



        #text = submission.selftext

        # this gets the comments


    #submissions[submission.title] = text
'''
