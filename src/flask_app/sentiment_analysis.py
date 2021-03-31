# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sqlite3
import pandas as pd
import nltk
import yaml
import sys
import os
import re
from pprint import pprint
from textblob import TextBlob


# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

# split comment into sentences
# split sentences into tokens & add part of speech tag
# tag these tokens based off of positive, negative, incremental, decremental, inverted dictionaries
# calculate score based of occurence of tags

def splitString(text):
    nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
    nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
    sentences = nltk_splitter.tokenize(text)
    tokenized_sentences = [nltk_tokenizer.tokenize(sentence) for sentence in sentences]
    return tokenized_sentences


def addPosTag(sentences):
    pos = [nltk.pos_tag(sentence) for sentence in sentences]
    # make format pretty
    pos = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
    return pos

def tag_value(sentiment):
    if sentiment == 'positive':
        return 1
    if sentiment == 'negative':
        return -1
    return 0

def sentence_total_score(tokens, prev_token, total_score):
    if not tokens:
        return total_score
    else:
        current_token = tokens[0]
        tags = current_token[2]
        token_score = sum([tag_value(tag) for tag in tags])
        if prev_token is not None:
            prev_tag = prev_token[2]
            if 'inc' in prev_tag:
                token_score *= 2.0
            elif 'dec' in prev_tag:
                token_score /= 2.0
            elif 'inv' in prev_tag:
                token_score *= -1.0

    return sentence_total_score(tokens[1:], current_token, total_score + token_score)

def sentiment_score(comment):
    return sum([sentence_total_score(sentence, None, 0.0) for sentence in comment])


def combineDictionaries():
    dictionary_paths = ['./positive.yml', './negative.yml','./inc.yml', './dec.yml','./inv.yml']
    files = [open(path,'r') for path in dictionary_paths]
    dictionaries = [yaml.load(dict) for dict in files]
    map(lambda x:x.close(), files)

    dictionary = {}
    max_key_size = 0

    # combine dictionaries
    for current_dictionary in dictionaries:
        for key in current_dictionary:
            if key in dictionary:
                dictionary[key].extend(current_dictionary[key])
            else:
                dictionary[key] = current_dictionary[key]
                max_key_size = max(max_key_size, len(key))

    return max_key_size, dictionary


def getSentiment(text):
    #text = text.encode('utf-8'
    split_text = splitString(text)
    pos_tag_text = addPosTag(split_text)

    max_key_size, dictionary = combineDictionaries()
    tagged_sentences = makeTag(pos_tag_text, max_key_size, dictionary)

    #print "Sentiment: ", sentiment_score(tagged_sentences)
    #print "Type: ", type(sentiment_score(tagged_sentences))
    senti_score = sentiment_score(tagged_sentences)
    #print " "
    # compare to TextBlob
    #blob = TextBlob(text)
    #print "TextBlob Sentiment: ", blob.sentiment.polarity
    return senti_score


def makeTag(postagged_sentences,max_key_size, dictionary):
    return [addTag(sentence, max_key_size, dictionary) for sentence in postagged_sentences]

def addTag(sentence,max_key_size, dictionary, tag_stem=False):
    # Tag all possible sentences
    tagged_sentence = []
    length = len(sentence)
    if max_key_size == 0:
        max_key_size = length
    i = 0
    while (i < length):
        j = min(i + max_key_size, length)
        tagged = False
        while (j > i):
            expression_word = ' '.join([word[0] for word in sentence[i:j]]).lower()
            expression_stem = ' '.join([word[1] for word in sentence[i:j]]).lower()
            if tag_stem == True:
                word = expression_word
            else:
                word = expression_word
            if word in dictionary:
                is_one_word = j - i == 1
                original_pos = i
                i = j
                tags = [tag for tag in dictionary[word]]
                tagged_word = (expression_word, expression_stem, tags)

                if is_one_word:
                    original_tag = sentence[original_pos][2]
                    tagged_word[2].extend(original_tag)
                tagged_sentence.append(tagged_word)
                tagged = True
            else:
                j = j - 1
        if not tagged:
            tagged_sentence.append(sentence[i])
            i += 1
    return tagged_sentence


def main():
    conn = sqlite3.connect('reddit_comments.db')
    c = conn.cursor()
    c.execute(''' SELECT commentBody FROM allData LIMIT 10;''')
    sql = c.fetchall()

    df = pd.DataFrame(sql)
    text = ""

    for i in df[0]:
        text+=i
        print text
        print getSentiment(text)
        text = ""

'''
    test = raw_input("Enter your sentence: ")
    score = getSentiment(test)'''

'''or s in score:
        print s'''


if __name__ == '__main__':
    main()
