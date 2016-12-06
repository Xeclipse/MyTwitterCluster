#coding: utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Layer, Merge
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.text as prep
import keras.preprocessing.sequence as seq
import  random
from keras.constraints import maxnorm, nonneg
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers.core import Lambda
import preprocess.DataPre as dp


def readEmbedding(file='../resource/fsd/FSD_LSTM_embedding'):
    f=open(file)
    embedding=[]
    for i in f .readlines():
        tmp=[float(k) for k in i.strip().split()]
        embedding.append(tmp)
    return np.asanyarray([embedding])

def readImportance(file='../resource/fsd/FSD_Importance'):
    f=open(file)
    embedding=[]
    for i in f .readlines():
        tmp=[float(k) for k in i.strip().replace('[','').replace(']','').split()]
        embedding.append(tmp)
    return np.asanyarray([embedding])


data,toknizer,maxlen=dp.dataText2Seq()
vocabSize = toknizer.word_index.__len__()+2

labelsFile='../resource/fsd/labels'
file=open(labelsFile)
tmp=[int(i)-1 for i in file.readlines()]
labelLen=max(tmp)+1
print labelLen

tmpLabel=[random.randint(0,labelLen-1) for i in range(0,2499)]
X_train=data
Y_train=dp.prepLabel(tmpLabel,labelLen)


weights=readEmbedding()
importance=readImportance()

leftmodel = Sequential(name='Embedding')
leftmodel.add(Embedding(input_length=maxlen,input_dim=vocabSize, output_dim=50, mask_zero= True, name='embedding',weights=weights, trainable=False))
rightmodel=Sequential(name='Importance')
rightmodel.add(Embedding(input_length=maxlen,input_dim=vocabSize, output_dim=1, mask_zero= True, name='importance', weights=importance, trainable=False))


model=Sequential()
model.add(Merge([leftmodel,rightmodel], mode=(lambda x: x[0]*K.repeat_elements(x[1],50,2)) , output_shape=(maxlen,50), name='merge'))
model.add(LSTM(output_dim=100, input_length=maxlen, activation='tanh', inner_activation='hard_sigmoid',return_sequences=False,name='lstm'))
model.add(Dense(labelLen))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])
#early_stopping = EarlyStopping(monitor='val_loss', patience=0.000002)



changeFile=open('../output/changes_importance','w')
labelsChangeFile=open('../output/labelChanges_importance','w')
changes=10000
while changes>1:
    model.fit(x=[X_train, X_train], y=Y_train, batch_size=10, nb_epoch=10)
    tmpAns=model.predict([X_train, X_train])
    ttmpLabel=[np.argmax(i) for i in tmpAns]
    labelsChangeFile.write(str(ttmpLabel)+'\n')
    changes=0
    for i in range(ttmpLabel.__len__()):
        if ttmpLabel[i]!=tmpLabel[i]:
            changes+=1
    st=set(ttmpLabel)
    print '========================================'
    print st
    print st.__len__()
    print '========================================'
    changeFile.write(str(changes)+'\n')
    changeFile.flush()
    tmpLabel=ttmpLabel
    Y_train=dp.prepLabel(tmpLabel,labelLen)
changeFile.close()