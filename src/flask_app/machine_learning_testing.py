import unittest
import sqlite3
import pandas
from sentiment_analysis import *
from app import *

conn = sqlite3.connect("reddit_comments.db")
c = conn.cursor()
# sql query to set up data for the machine learning algorithms, they require a dataframe to be passed in 
c.execute('''SELECT commentBody, Subreddit, commentScore, Sentiment, created_at FROM allData WHERE LENGTH(commentBody) < 100 and LENGTH(commentBody) > 10 and commentScore < 5 ORDER BY Random() LIMIT 50000; ''')
data = c.fetchall()
data_df = pd.DataFrame(data)
data_df.columns = ['comment', 'subreddit', 'score', 'sentiment', 'created_utc']

# add categories
data_df['category_id'] = data_df['subreddit'].factorize()[0]
category_id_df = data_df[['subreddit', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','subreddit']].values)


class Analysis_Test_Case(unittest.TestCase):
    def test_predictSentence(self):
        result1 = predict_sentence("My favourite sport is basketball")
        result2 = predict_sentence("38")
        result3 = predict_sentence(" ")
        result4 = predict_sentence("I really like to travel")

        self.assertEqual(result1, [u'nba'])
        # for result2 and result3, it should return either /r/nba or /r/soccor as they are the largest subreddits
        self.assertEqual(result2, [u'nba'])
        self.assertEqual(result3, [u'soccer'])
        self.assertEqual(result4, [u'travel'])

    def test_predictedSubreddit(self):
        nb_result, svm_result = get_predicted_subreddit(c, data_df)
        self.assertGreater(nb_result, 30)
        self.assertGreater(svm_result, 65)

    def test_predictedScore(self):
        ab_result, svc_result = get_predicted_score(c,data_df)
        self.assertGreater(ab_result, 30)
        self.assertGreater(svc_result, 30)



if __name__ == '__main__':
    unittest.main()
