
# encoding=utf8
from __future__ import unicode_literals
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import sqlite3
import pandas as pd
import nltk
import yaml
import sys
import os
import re
from pprint import pprint
from textblob import TextBlob








class StringSplitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        # input: paragraph
        # output: ngrams

        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class POSTagger(object):
    def __init__(self):
         pass

    # add part of speech tag for adverbs, nouns etc
    def pos_tag(self, sentences):
        # input: list of ngrams
        # output: list of lists of tagged tokens (word, lemma, tag)

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        # adapt format
        pos = [[(word,word,[postag]) for(word, postag) in sentence] for sentence in pos]
        return pos



class DictionaryTagger(object):
    def __init__(self, dict_paths):
        files = [open(path, 'r') for path in dict_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x:x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0

        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        # tagging all possible ones depending on:
        # longest matches = higher priority
        # search is L->R

        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while(i<N):
            j = min(i+self.max_key_size,N) # avoid overflow
            tagged = False
            while(j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_form
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    is_single_token = j - i == 1
                    original_pos = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expr = (expression_form, expression_lemma, taggings)

                    if is_single_token:
                        original_token_tagging = sentence[original_pos][2]
                        tagged_expr[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expr)
                    tagged = True
                else:
                    j = j -1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence


def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0
def sentence_score(sentence_tokens, previous_token, acum_score):
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0

        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0) for sentence in review])





conn = sqlite3.connect('reddit_comments.db')
c = conn.cursor()

c.execute(''' SELECT commentBody FROM allData LIMIT 10;''')
sql = c.fetchall()

df = pd.DataFrame(sql)
text = ""

for i in df[0]:
    text+=str(i)

    print text
    print " "
    text = text.encode('utf-8')
    splitter = StringSplitter()
    postagger = POSTagger()

    splitted_sentences = splitter.split(text)
    #print splitted_sentences

    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
    #print pos_tagged_sentences

    dicttagger = DictionaryTagger(['./positive.yml', './negative.yml','./inc.yml', './dec.yml','./inv.yml'])
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)

    #pprint(dict_tagged_sentences)
    print "Sentiment:",sentiment_score(dict_tagged_sentences)
    sent_score = sentiment_score(dict_tagged_sentences)
    print " "
    blob = TextBlob(text)
    print "TextBlob Sentiment: ", blob.sentiment.polarity
    blob_sent = blob.sentiment.polarity
    if sent_score > blob_sent:
        difference = sent_score - blob_sent
    else:
        difference = blob_sent - sent_score
    print "Difference: ", difference
    text = ""
