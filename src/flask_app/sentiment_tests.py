import unittest
from sentiment_analysis import *

class Sentiment_Test_Case(unittest.TestCase):

    def test_getSentiment(self):
        result1 = getSentiment("I don't like that shit")
        result2 = getSentiment("Who cares?")
        result3 = getSentiment("Thank you so much")
        result4 = getSentiment("People who are still alive who were in the camps aren't running around telling people they were victims of the holocaust, and demanding more be done to teach kids about the holocaust. Cherry picking stories of the victims and throwing them at schoolkids adds nothing to their learning, the same way cherry picking the stories of some Syrian refugee adds nothing to understanding the situation in Syria. All this is is more idiots crying about how the holocaust is the only thing that matters, because horrible shit hasn't happened in the 70 years since it ended. It's bandwagoners that can't get over it, they're the problem.")
        result5 = getSentiment("The Tick is good and theyre also working on a Dark Tower series as well as a Wheel of Time series. They have some great stuff lined up, just need to get it out there.")
        result6 = getSentiment("12")
        result7 = getSentiment(" ")

        self.assertEqual(result1, -1.0)
        self.assertEqual(result2, 0.0)
        self.assertEqual(result3, 1.0)
        self.assertEqual(result4, -4.0)
        self.assertEqual(result5, 3.0)
        self.assertEqual(result6, 0.0)
        self.assertEqual(result7,0)


    def test_splitString(self):
        result1 = splitString("I love that movie. My favourite was the third one though")
        result2 = splitString(" ")
        result3 = splitString("thanks")
        result4 = splitString("I agree.")
        result5 = splitString("17:38,24/7 ALL week")

        correct_result = [['I', 'love', 'that', 'movie', '.'], ['My', 'favourite', 'was', 'the', 'third', 'one', 'though']]
        self.assertEqual(result1, correct_result)
        self.assertEqual(result2,[])
        self.assertEqual(result3, [['thanks']])
        self.assertEqual(result4, [['I', 'agree','.']])
        self.assertEqual(result5, [['17:38,24/7','ALL','week']])


    def test_tagValue(self):
        result1 = tag_value("positive")
        result2 = tag_value("negative")
        result3 = tag_value("random")
        result4 = tag_value("42")
        result5 = tag_value(" ")

        self.assertEqual(result1, 1)
        self.assertEqual(result2, -1)
        self.assertEqual(result3, 0)
        self.assertEqual(result4, 0)
        self.assertEqual(result5, 0)

    def test_makeTag(self):

        max_key_size, dictionary = combineDictionaries()

        result1 = splitString("I hate that")
        tag_result1 = addPosTag(result1)
        makeTagResult1 = makeTag(tag_result1,max_key_size,dictionary)
        expected_result = [[('I', 'I', ['PRP']), (u'hate', u'hate', ['negative', 'VBP']), ('that', 'that', ['IN'])]]
        self.assertEqual(makeTagResult1,expected_result)

        result2 = splitString("Oh okay")
        tag_result2 = addPosTag(result2)
        makeTagResult2 = makeTag(tag_result2,max_key_size,dictionary)
        expected_result = [[('Oh', 'Oh', ['UH']), ('okay', 'okay', ['NN'])]]
        self.assertEqual(makeTagResult2,expected_result)

        result3 = splitString("38")
        tag_result3 = addPosTag(result3)
        makeTagResult3 = makeTag(tag_result3, max_key_size, dictionary)
        expected_result = [[('38', '38', ['CD'])]]
        self.assertEqual(makeTagResult3, expected_result)

        result4 = splitString(" ")
        tag_result4 = addPosTag(result4)
        makeTagResult4 = makeTag(tag_result4, max_key_size, dictionary)
        self.assertEqual(makeTagResult4, [])

    def test_sentiScore(self):
        '''###############################################################
            sentiment_score() uses sentence_total_score() so tests both
           ###############################################################
         '''
        max_key_size, dictionary = combineDictionaries()
        text1 = "Ugh I hate that"
        text1 = splitString(text1)
        tagged = addPosTag(text1)
        tagged_full = makeTag(tagged, max_key_size, dictionary)
        result1 = sentiment_score(tagged_full)
        

        text2 = "That is awful"
        text2 = splitString(text2)
        tagged = addPosTag(text2)
        tagged_full = makeTag(tagged, max_key_size, dictionary)
        result2 = sentiment_score(tagged_full)

        text3 = "He seems to have a great singing voice"
        text3 = splitString(text3)
        tagged = addPosTag(text3)
        tagged_full = makeTag(tagged, max_key_size, dictionary)
        result3 = sentiment_score(tagged_full)


        text4 = "Wow that is really really awesome!"
        text4 = splitString(text4)
        tagged = addPosTag(text4)
        tagged_full = makeTag(tagged, max_key_size, dictionary)
        result4 = sentiment_score(tagged_full)


        text5 = "418"
        text5 = splitString(text5)
        tagged = addPosTag(text5)
        tagged_full = makeTag(tagged, max_key_size, dictionary)
        result5 = sentiment_score(tagged_full)


        text6 = ""
        text6 = splitString(text6)
        tagged = addPosTag(text6)
        tagged_full = makeTag(tagged, max_key_size, dictionary)
        result6 = sentiment_score(tagged_full)


        self.assertEqual(result1, -2.0)
        self.assertEqual(result2, -1.0)
        self.assertEqual(result3, 1.0)
        self.assertEqual(result4, 3.0)
        self.assertEqual(result5, 0.0)
        self.assertEqual(result6, 0)






if __name__ == '__main__':
    unittest.main()
