import praw
from textblob import TextBlob
import math
from praw.models import MoreComments
from sentiment_analysis import *
import io

reddit = praw.Reddit(client_id='SYLwpLpvyBQ4Xw', client_secret='QuyR-qIccKFkhKzdwr_ZMFdrkdw',
                      user_agent='subSentiment')

# add in comments from new as well as hot

subreddit_sentiment = 0
num_comments = 0
conn = sqlite3.connect("./reddit_comments.db")
conn.text_factory = str
cur = conn.cursor()
dropStmt = "DROP TABLE moreData"
cur.execute(dropStmt)


cur.execute('''CREATE TABLE moreData
            (commentBody text, Subreddit text, User text, commentScore int, Sentiment float, created_at int)''')

filename = "moresubs.txt"
with io.open(filename, encoding='utf-8') as f:
    for line in f:
        string1 = '"' + line.strip() + '"'
        sub = line.strip()
        print sub


        ''' TO DO:
            Repopulate database with own sentiment analysis algorithma
            Find out a way to add loads of data from subs without hitting LIMIT
            or unexpected end of file
        '''
        # top 5 hot submissions & their comments 
        for submission in reddit.subreddit(sub).hot(limit=5):
            subreddit = reddit.subreddit(line.strip())
            name = str(subreddit.display_name)
            comments = submission.comments.list()

            for c in comments:
                if isinstance(c, MoreComments):
                    continue
                #print c.body
                author = c.author
                score = c.score
                created_at = c.created_utc
                upvotes = c.ups
                #print c.score
                comment_sentiment = getSentiment(c.body)

                #comment = (str(c.body.encode('utf-8',errors='ignore')))
                comment = c.body
                sentiment = str(comment_sentiment)
                #print comment
                #print sentiment


                if author is None:
                    username = "Null"
                else:
                    username = author.name

                params = (comment, name, username, score, sentiment, created_at)
                cur.execute("INSERT INTO moreData VALUES(?,?,?,?,?,?)",params)
                conn.commit()






'''
        #submission.comment_sort = "/top"



        #text = submission.selftext

        # this gets the comments


    #submissions[submission.title] = text
'''
