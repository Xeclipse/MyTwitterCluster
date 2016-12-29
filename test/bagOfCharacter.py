#coding: utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Layer
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.text as prep
import sklearn.cluster as sc
import nltk as nlt
import keras.preprocessing.sequence as seq
from keras.preprocessing.text import one_hot
import string


corpusFile="../resource/fsd/pure_tweets_fsd"
labelsFile='../resource/fsd/labels'
processedFile='../resource/fsd/processed_tweets_fsd'


def bagofword(word):
    v=[0]*256
    for i in word:
        if ord(i)<256:
            v[ord(i)]+=1
    return v

#print bagofword('hello')

def process(line):
    ret=''
    for i in line.strip().split():
        if i.find('http')>=0 or i.find('...')>=0 or i.find('&')>=0:
            continue
        else:
            ret+=i+' '
    return ret+'\n'

file = open(corpusFile)
text = [process(i) for i in file.readlines()]
print text
file.close()
toknizer = prep.Tokenizer()
toknizer.fit_on_texts(texts=text)
file = open(processedFile,'w')
for i in text:
    file.write(i)
file.close()
words= toknizer.word_index.keys()
wordsv=[bagofword(i) for i in words]
print wordsv
