#coding: utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Layer
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.text as prep
import keras.preprocessing.sequence as seq
from keras.callbacks import EarlyStopping
from keras import backend as K


corpusFile="../resource/fsd/pure_tweets_fsd"
labelsFile='../resource/fsd/labels'


def dataText2Seq(fileName="../resource/fsd/pure_tweets_fsd", nb=1900):
    file = open(fileName)
    text = [i.strip() for i in file.readlines()]
    toknizer = prep.Tokenizer(nb_words=nb)
    toknizer.fit_on_texts(texts=text)
    data = toknizer.texts_to_sequences(texts=text)
    data = np.asanyarray(data)
    data = seq.pad_sequences(sequences=data, padding='pre')

    maxlen = [i.__len__() for i in data]
    maxlen = maxlen[np.argmax(maxlen)]
    file.close()
    return data,toknizer,maxlen


#trans label from 1 dim to k dim, bag of label :)
def prepLabel(tmp,LabelNum):
    labels=[]
    for i in tmp:
        k=[0]*LabelNum
        k[i]=1
        labels.append(k)
    return labels