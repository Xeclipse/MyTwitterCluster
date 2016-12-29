#coding: utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.text as prep
import keras.preprocessing.sequence as seq
from keras import backend as K
'''
root='../../sinanewsprocess/'
corpusFile=root+"fenci_feature.txt"
labelsFile=root+'label.txt'
'''

root='../resource/song/'
corpusFile=root+"jiebaqqsong.txt"
labelsFile=root+'label83_train.txt'

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
labels=[[float(k) for k in i.strip().split(' ')] for i in file.readlines()]

X_train=data[0:450]
Y_train=labels
test=data[450:500]

model = Sequential()
model.add(Embedding(input_dim=vocabSize, output_dim=100, mask_zero= True))
model.add(LSTM(output_dim=100, input_length=maxlen,activation='tanh', inner_activation='hard_sigmoid',return_sequences=False,name='lstm'))
model.add(Dense(labels[0].__len__()))
model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['mse'])

model.fit(X_train, Y_train, batch_size=20, nb_epoch=150)
get_lstm_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(name='lstm').output])
lstmout=get_lstm_layer_output([X_train])[0]
file=open('feature_qqsong_train','w')
for i in lstmout:
    tmp=str(i).replace('\n','').replace('[','').replace(']','').strip()
    file.write(tmp+'\n')
file.close()


get_lstm_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(name='lstm').output])
lstmout=get_lstm_layer_output([test])[0]
file=open('feature_qqsong_test','w')
for i in lstmout:
    tmp=str(i).replace('\n','').replace('[','').replace(']','').strip()
    file.write(tmp+'\n')
file.close()



