#coding: utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Layer
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.text as prep
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.cluster as sc
import nltk as nlt
import keras.preprocessing.sequence as seq
from keras.preprocessing.text import one_hot
import string


corpusFile="../resource/fsd/processed_tweets_fsd"
labelsFile='../resource/fsd/labels'
#processedFile='../resource/fsd/processed_tweets_fsd'


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

def mostSimilar(w,dicW):
    sim=0
    wv=dicW[w]
    word=''
    for i in dicW:
        dis=0
        if i!=w:
            dis=cosine_similarity(wv,dicW[i])
        if dis>sim:
            sim=dis
            word=i
    return sim, word


file = open(corpusFile)
text = [process(i) for i in file.readlines()]
print text
file.close()
toknizer = prep.Tokenizer()
toknizer.fit_on_texts(texts=text)

words= toknizer.word_index.keys()
wordsv=[bagofword(i) for i in words]
dicW={}
for i in range(words.__len__()):
    dicW[words[i]]=wordsv[i]

dis, w=mostSimilar(words[3],dicW)
print words[3]
print dis
print w