#coding: utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.text as prep
import keras.preprocessing.sequence as seq
from keras import backend as K
import  sklearn.cluster as clu
from matplotlib import pyplot as plt



'''
step 1: init initial label
    step 1.1: assign a tweet with a random label
step 2: for all samples: randomly choose a label with probability genrated by each cluster neural net work
step 3: repeat step 2 until converged
'''


def createNewModel(wordEmbeddingVocab, vocabSize, maxlen):
    model = Sequential()
    model.add(Embedding(input_dim=vocabSize + 1, output_dim=256,weights=wordEmbeddingVocab))
    model.add(LSTM(output_dim=500, input_length=maxlen, activation='tanh', inner_activation='hard_sigmoid',
                   return_sequences=False, name='lstm'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mse'])
    return model


def trainModel(model, data, labels):
    model.fit(data,labels)

'''
change all index in labels with 1 and others' with 0
'''
def processLabels(labels, val):
    tmplabel=labels[:]
    for i,v in enumerate(tmplabel):
        if v==val: tmplabel[i]=1
        else: tmplabel[i]=0
    return tmplabel

def savePredictedText(file,pred,text):
    saveFile=file
    w=open(saveFile,'w')
    tmp=[]
    for i,v in enumerate(text):
        tmp.append(str(pred[i])+'\t'+v)
    tmp=sorted(tmp)
    for i in tmp:
        w.write(i)
    w.close()

def getWordEmbedding(model):
    return False


def updateNeuralNets(models, data, labels,batch=5,epoch=10):
    for i in models:
        tmpLabel=processLabels(labels=labels,val=i)
        print 'Model',
        print i
        models[i].fit(data,labels,batch_size=batch,nb_epoch=epoch)

def dataText2Seq(fileName="../resource/pure_tweets_fsd"):
    file = open(fileName)
    text = file.readlines()
    toknizer = prep.Tokenizer()
    toknizer.fit_on_texts(texts=text)
    data = toknizer.texts_to_sequences(texts=text)
    data = np.asanyarray(data)
    data = seq.pad_sequences(sequences=data, padding='pre')

    maxlen = [i.__len__() for i in data]
    maxlen = maxlen[np.argmax(maxlen)]
    return data,toknizer,maxlen
def saveLabels(labels,file):
    o = open(file,'w')
    for i in labels:
        o.write(str(i) + '\n')
    o.close()

def sampling(file='',cNumber=30):
    #prepare data from texts
    print 'prepare data & labels ...'
    data, toknizer, maxlen = dataText2Seq()
    vocabSize = toknizer.word_index.__len__()+1
    nSample = data.__len__()
    '''
    #tokenizer information
    print toknizer.word_index
    print toknizer.word_counts
    print maxlen
    '''


    #initial labels, randomly assignment
    labels = [int(i) for i in np.random.uniform(1, cNumber, data.__len__())]

    print 'initialize models:'
    #initial models
    models={}
    for i in range(cNumber):
        print 'model',
        print i
        models[i]=createNewModel(wordEmbeddingVocab=None,vocabSize=vocabSize,maxlen=maxlen)
    updateNeuralNets(models, data, labels,batch=5,epoch=10)
    coverage=100
    batch = 2499


    print 'start sampling'
    change=10000
    count=0
    while change>coverage:
        #sampling 1 iter
        change=0
        count+=1
        print 'change\t',
        print change
        start = 0
        while start<nSample:
            end=min(nSample,start+batch)
            for i in range(start,end):
                # Randomly Assign Tweet i
                params=[]
                for i in models:
                    params.extend(models[i].predict(data[i])[0])
                l=np.argmax(params)
                if l!=labels[i]:
                    change+=1
                    labels[i]=l
            saveLabels(labels, '../output/labels' + str(count))
            print '------C-H-A-N-G-E---------',
            print change
            if change<coverage: break
            updateNeuralNets(models,data,labels,batch=5, epoch=15)
            start=end

    for i in models:
        models[i].save(filepath=('../output/model'+str(i)),overwrite=True)
    return labels



labels=sampling(cNumber=30)
saveLabels(labels,'../output/predict_labels')