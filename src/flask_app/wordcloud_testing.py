import unittest
import sqlite3
import pandas
from sentiment_analysis import *
from app import *


conn = sqlite3.connect("reddit_comments.db")
c = conn.cursor()


class WordCloud_Test_Case(unittest.TestCase):
    def test_wordcloudLength(self):
        result = wordcloud_words(c)
        result_length = len(result)
        self.assertEqual(result_length, 50)

    def test_wordcloudType(self):
        result = wordcloud_words(c)
        result_type = type(result)
        isDataFrame = False
        if isinstance(result, pd.DataFrame):
            isDataFrame = True
        self.assertEqual(isDataFrame, True)

    def test_subFreqWords(self):
        test_subreddit = "programming"
        string = "Subreddit"
        c.execute("SELECT commentBody FROM allData WHERE "+string+"=?",(test_subreddit,))
        sql_query = c.fetchall()
        conn.commit()
        sql_query = list(sql_query)
        result = subFreqWords(sql_query)
        isList = False
        print type(result)
        if isinstance(result, list):
            isList = True
        self.assertEqual(isList, True)



if __name__ == '__main__':
    unittest.main()
