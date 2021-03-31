from textblob import TextBlob
from sentiment_analysis import *

def main():
    sentences = []
    sentence1 = "I hate that movie, it's so bad"
    sentence2 = "wow, dude I think you should calm down"
    sentence3 = "words can't express how much I love that!"
    sentence4 = "to be honest, I don't really care too much about it"
    sentence5 = "Mods must be asleep. They would've taken my shit down quick with some bs. "
    sentence6 = "there is a lack of integrity"
    sentence7 = "I love dogs so much, they are so cute"
    sentence8 = ",.[)@]+"
    sentence9 = "6789998212"
    sentence10 = " "

    sentences.append(sentence1)
    sentences.append(sentence2)
    sentences.append(sentence3)
    sentences.append(sentence4)
    sentences.append(sentence5)
    sentences.append(sentence6)
    sentences.append(sentence7)
    sentences.append(sentence8)
    sentences.append(sentence9)
    sentences.append(sentence10)

    for sentence in sentences:
        print sentence
        print "Own Sentiment: ", getSentiment(sentence)
        blob = TextBlob(sentence)
        print "TextBlob's Sentiment: ", blob.sentiment.polarity
        print " "



if __name__ == '__main__':
    main()
