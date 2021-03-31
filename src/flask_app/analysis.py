import pandas as pd
from textblob import TextBlob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import Counter



def getSentiment(df):
   # get a basic sentiment analysis per each entry
   # in the racist_comments dataframe
   for entry in df:
       blob = TextBlob(entry)
       print entry
       print blob.sentiment.polarity, blob.sentiment.subjectivity
       print " "


def authorFrequency(df):
    # want to see users who frequently comments, and see if their comments are neg/pos

    # commented more than once
    print "==== USERS WHO COMMENTED MORE THAN ONCE ===="
    frq_users = pd.concat(g for _, g in df.groupby("Author") if len(g) > 1)
    df = frq_users[['Author','Comment', 'Score']]
    score = frq_users['Score']

    # average score is 3.6
    # want to find comments with < & > than the average score
    # can then get a feel of the frequent posters attitude
    less_than_df = df.loc[df['Score'] < 3.6]
    greater_than = df.loc[df['Score'] > 3.6]
    print "==== USERS WHOS COMMENTED SCORED < AVG (3.6) ===="
    print less_than_df
    print " "
    print "==== USERS WHO COMMENTED SCORE > AVG (3.6) ===="
    print greater_than
    print " "

    # commented more than twice -- all of them scored < than average score
    more_fr = pd.concat(g for _, g in df.groupby("Author") if len(g) > 2)
    more_df = more_fr[['Author', 'Comment','Score']]
    print "==== USERS WHO COMMENTED MORE THAN TWICE ===="
    print more_df
    print "==== Conclusion: All of them scored < avg score"
    print " "






def main():

    rel_df = pd.read_csv('./cleaned.csv', sep=",", encoding='utf-8')
    count = 0

    authorFrequency(rel_df)

    #comments = rel_df['Comment']
    #comments = comments.dropna()

    #racist_df = rel_df[rel_df['Comment'].str.contains('racis')]
    #racist_comments = racist_df['Comment']
    #print racist_comments[:10]


    # there are 13 comments that contain "racist"
    '''for i in racist_comments:
        count = count + 1
    print count

    comment_list = list(comments)
    comment_blob = TextBlob(''.join(comments))'''

    #print (len(comment_blob.words))
    # comment_blob.words tokenizes each comment
    # counter returns the count of how frequent each word is
    '''counter = Counter(comment_blob.words)
    counter.most_common()[60:100]'''


    #print rel_df[:500]

    #getSentiment(comments)








if __name__ == '__main__':
    main()
