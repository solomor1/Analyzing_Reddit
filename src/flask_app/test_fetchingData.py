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


class Fetching_Data_Test(unittest.TestCase):

    # getNewestPostsinSub
    def test_getNewestPostsinSub(self):
        subreddit2 = " "
        result2 = getNewestPostsinSub(subreddit2, conn)
        self.assertEqual(result2, "Invalid")
        subreddit1 = "LoseIt"
        new_submissions, new_comments, top_neg, top_pos = getNewestPostsinSub(subreddit1, conn)
        isInteger = False
        if isinstance(new_submissions, int):
            isInteger = True
        self.assertEqual(isInteger, True )


if __name__ == '__main__':
    unittest.main()
