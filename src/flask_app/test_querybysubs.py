# -*- coding: utf-8 -*-
import unittest
from app import *
import praw
import sqlite3
import pandas as pd
import re

reddit = praw.Reddit(client_id='SYLwpLpvyBQ4Xw', client_secret='QuyR-qIccKFkhKzdwr_ZMFdrkdw',
                      user_agent='subSentiment')
conn = sqlite3.connect("reddit_comments.db")
c = conn.cursor()


class querybysub_test(unittest.TestCase):


    # getNewestPostsinSub
    '''def test_getNewestPostsinSub(self):
        subreddit2 = " "
        result2 = getNewestPostsinSub(subreddit2, conn)
        self.assertEqual(result2, "Invalid")
        subreddit1 = "LoseIt"
        new_submissions, new_comments, top_neg, top_pos = getNewestPostsinSub(subreddit1, conn)
        isInteger = False
        if isinstance(new_submissions, int):
            isInteger = True
        self.assertEqual(isInteger, True )'''


    def test_frequentWordsinSub(self):
        result1 = frequentWordsinSub(c)
        is_df = False
        if isinstance(result1, pd.DataFrame):
            is_df = True

        self.assertEqual(is_df, True)

    # overall_sentiment
    def test_overallSentiment(self):
        result1 = overall_sentiment(c)
        is_float = False


        if isinstance(result1, float):
            is_float = True
        self.assertEqual(is_float, True)


    # activePostersinSub
    def test_activePostersinSub(self):
        result1 = activePostersinSub(c)
        is_dict = False
        if isinstance(result1, dict):
            is_dict = True
        self.assertEqual(is_dict, True)

    # language_style_subreddit
    def test_language_style_subreddit(self):
        result1, result2, result3 = language_style_subreddit(conn)
        isSeries = False
        length = len(result1)
        if isinstance(result1, pd.Series):
            isSeries = True
        self.assertEqual(isSeries, True )
        self.assertEqual(length,5)
        self.assertEqual(len(result2), 5)
        self.assertEqual(len(result3), 5)






if __name__ == '__main__':
    unittest.main()
