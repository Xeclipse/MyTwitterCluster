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
            v[ord(i)]+=10
    return v

#print bagofword('hello')


#filter useless words
def process(line):
    ret=''
    for i in line.strip().split():
        if i.find('http')>=0 or i.find('...')>=0 or i.find('&')>=0:
            continue
        else:
            ret+=i+' '
    return ret+'\n'

def cmpFloatTuple(x,y):
    return int(x[1]<y[1])

def mostSimilar(w,dicW):
    sim=[]
    wv=dicW[w]
    for i in dicW:
        sim.append((i,cosine_similarity(wv,dicW[i])[0][0]))
    sim= sorted(sim,key=lambda x:x[1], reverse=True)
    return sim


file = open(corpusFile)
text = file.readlines()
file.close()
toknizer = prep.Tokenizer()
toknizer.fit_on_texts(texts=text)

words= toknizer.word_index.keys()
wordsv=[bagofword(i) for i in words]
dicW={}
for i in range(words.__len__()):
    dicW[words[i]]=wordsv[i]
k= mostSimilar(words[0],dicW)
print k
