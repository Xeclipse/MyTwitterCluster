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


corpusFile="../resource/pure_tweets_fsd"
labelsFile='../resource/labels'

file=open(corpusFile)
text=file.readlines()
toknizer=prep.Tokenizer()#nb_words=2000
toknizer.fit_on_texts(texts=text)
data=toknizer.texts_to_sequences(text)
data=np.asanyarray(data)
maxlen=[i.__len__() for i in data]
maxlen=maxlen[np.argmax(maxlen)]
#print toknizer.word_index
#print toknizer.word_counts
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


model = Sequential()
model.add(Embedding(input_dim=vocabSize, output_dim=10, mask_zero= True, name='embedding'))#from keras.callbacks import EarlyStopping
model.add(LSTM(output_dim=100, input_length=maxlen,activation='tanh', inner_activation='hard_sigmoid',return_sequences=False,name='lstm'))
model.add(Dense(labels[0].__len__()))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['mse'])
early_stopping = EarlyStopping(monitor='val_loss', patience=0.000002)

model.fit(X_train, Y_train, batch_size=100, nb_epoch=2)

embeddings = model.get_layer(name='embedding').get_weights()[0]

file=open('../resource/FSD_LSTM_embedding','w')
for i in embeddings:
    tmp=str(i).replace('\n','').replace('[','').replace(']','').strip()
    file.write(tmp+'\n')
file.close()
