#coding: utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Layer, Merge
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.text as prep
import keras.preprocessing.sequence as seq
from keras.constraints import maxnorm, nonneg
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers.core import Lambda


def readEmbedding(file='../resource/fsd/FSD_LSTM_embedding'):
    f=open(file)
    embedding=[]
    for i in f .readlines():
        tmp=[float(k) for k in i.strip().split()]
        embedding.append(tmp)
    return np.asanyarray([embedding])


corpusFile="../resource/fsd/processed_tweets_fsd"
labelsFile='../resource/fsd/labels'

file=open(corpusFile)
text= [i.strip() for i in file.readlines()]
toknizer=prep.Tokenizer(nb_words=1900)#
toknizer.fit_on_texts(texts=text)
data=toknizer.texts_to_sequences(text)
data=np.asanyarray(data)
maxlen=[i.__len__() for i in data]
maxlen=maxlen[np.argmax(maxlen)]
worddic=toknizer.word_index
wordcount= toknizer.word_counts
vocabSize = toknizer.word_index.__len__()+2
data=seq.pad_sequences(sequences=data,padding='pre')
file.close()



file=open(labelsFile)
tmp=[int(i)-1 for i in file.readlines()]
labelLen=max(tmp)+1
labels=[]
for i in tmp:
    k=[0]*labelLen
    k[i]=1
    labels.append(k)

X_train=data
Y_train=labels


weights=readEmbedding()

leftmodel = Sequential(name='Embedding')
leftmodel.add(Embedding(input_length=maxlen,input_dim=vocabSize, output_dim=50, mask_zero= True, name='embedding',weights=weights, trainable=False))
rightmodel=Sequential(name='Importance')
rightmodel.add(Embedding(input_length=maxlen,input_dim=vocabSize, output_dim=1, mask_zero= True, name='importance', trainable=True, W_constraint=nonneg()))


model=Sequential()
model.add(Merge([leftmodel,rightmodel], mode=(lambda x: x[0]*K.repeat_elements(x[1],50,2)) , output_shape=(maxlen,50), name='merge'))
model.add(LSTM(output_dim=100, input_length=maxlen, activation='tanh', inner_activation='hard_sigmoid',return_sequences=False,name='lstm'))
model.add(Dense(labels[0].__len__()))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])
#early_stopping = EarlyStopping(monitor='val_loss', patience=0.000002)

model.fit(x=[X_train,X_train], y=Y_train, batch_size=5, nb_epoch=50)



importance = rightmodel.get_layer(name='importance').get_weights()[0]
file=open('../output/FSD_Importance_Dic','w')
for i in worddic:
    tmp=tmp=str(i)+':'+str(importance[worddic[i]]).replace('\n','').replace('[','').replace(']','').strip()
    file.write(tmp+'\n')
file.close()

file=open('../resource/FSD_Importance','w')
for i in importance:
    tmp=str(i)
    file.write(tmp+'\n')
file.close()

'''
embeddings = model.get_layer(name='embedding').get_weights()[0]
file=open('../resource/FSD_LSTM_embedding','w')
for i in embeddings:
    tmp=str(i).replace('\n','').replace('[','').replace(']','').strip()
    file.write(tmp+'\n')
file.close()
'''
'''
file=open('../resource/fsd/FSD_Word_Vocab','w')
for i in worddic:
    tmp=str(i)+':'+str(worddic[i])
    file.write(tmp+'\n')
file.close()
'''
