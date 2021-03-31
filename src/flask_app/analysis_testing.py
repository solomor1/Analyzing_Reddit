import unittest
import sqlite3
import pandas
from sentiment_analysis import *
from app import *



conn = sqlite3.connect("reddit_comments.db")
c = conn.cursor()

class Analysis_Test_Case(unittest.TestCase):

    def test_detectSaracsm(self):
        result = detectSarcasm(c)
        self.assertEqual(result, 0.0236)

    def test_freqPosters(self):
        result = frequentPosters(c)
        expected_result = [u'KeepingDankMemesDank', u'kurzjacob', u'coololly', u'ohohlook', u'vtalii', u'FantasticFootball10', u'abedtime', u'LeKing_James', u'TheGenitalman', u'Optimuskck']
        self.assertEqual(result,expected_result)

    def test_freqWords(self):
        # discovered that I had to expand the stopwords lists
        result = frequentWords(c)
        length = len(result)
        self.assertEqual(length, 10)

        stopwords = nltk.corpus.stopwords.words('english')
        stoplist = """ something good great right player going way could though want make even better well go https http there this let every looks people feel put would getting ca us things said think around since many point say said still time said used need play made must looks really go get got also text next see try lot first much use think know into like sure one two three four five six back take"""
        stopwords += stoplist.split()
        isStopWord = False
        for word in result['Word']:
            if word in stopwords:
                isStopWord = True
        self.assertEqual(isStopWord, False)

    def test_top5neg(self):
        result = get_top_neg_5(c)
        expected_result = [u'cringepics', u'HistoryPorn', u'TrendingReddits', u'4chan', u'dankmemes']
        self.assertEqual(result, expected_result)

    def test_top5pos(self):
        result = get_top_pos_5(c)
        expected_result = [u'tattoos', u'travel', u'pokemon', u'malefashionadvice', u'sex']
        self.assertEqual(result, expected_result)

    def test_createSubSeries(self):
        result = createSubSeries()
        isDf = False
        if isinstance(result, pd.DataFrame):
            isDf = True
        self.assertEqual(isDf, True)

    def test_activeSubs(self):
        subreddit = createSubSeries()
        sarc_percentage, negative_percentage, pos_percentage, other = activeSubs(subreddit)
        self.assertEqual(sarc_percentage, 60.0)
        self.assertEqual(negative_percentage, 20.0)
        self.assertEqual(pos_percentage, 0.0)
        self.assertEqual(other, 20.0)

        isFloat = False
        if isinstance(sarc_percentage, float):
            isFloat = True
        if isinstance(negative_percentage, float):
            isFloat = True
        if isinstance(pos_percentage, float):
            isFloat = True
        if isinstance(other, float):
            isFloat = True
        self.assertEqual(isFloat, True)

    def test_timeConversion(self):
        time1 = str("2018:04:15:11:10")
        time2 = str("2014:05:18:17:16")
        time3 = str("2019:13:33:25:76")
        time4 = str(" ")
        alpha_time = "abcdef"
        result1 = convertTime(time1)
        result2 = convertTime(time2)
        result3 = convertTime(time3)
        result4 = convertTime(time4)
        result5 = convertTime(alpha_time)
        self.assertEqual(result1,'1523787000')
        self.assertEqual(result2,'1400429760')
        self.assertEqual(result3, "Invalid")
        self.assertEqual(result4, "Invalid")
        self.assertEqual(result5, "Invalid format")


if __name__ == '__main__':
    unittest.main()
