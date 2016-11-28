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
from keras.utils.visualize_util import plot



'''
step 1: assign each sample with a single label
step 2: train the neural net
step 3: extract last but two layer's output as features of tweets
step 4: clustering with kmeans
result: very bad
'''
file=open("../preprocess/pure_tweets_fsd")
text=file.readlines()
toknizer=prep.Tokenizer()
toknizer.fit_on_texts(texts=text)
data=toknizer.texts_to_sequences(texts=text)
data=np.asanyarray(data)
maxlen=[i.__len__() for i in data]
maxlen=maxlen[np.argmax(maxlen)]

print toknizer.word_index
print toknizer.word_counts
vocabSize=toknizer.word_index.__len__()
print maxlen


data=seq.pad_sequences(sequences=data,padding='pre')



X_train=data
Y_train=np.eye(data.__len__())

model = Sequential()
model.add(Embedding(input_dim=vocabSize+1, output_dim=256))
model.add(LSTM(output_dim=500, input_length=maxlen,activation='tanh', inner_activation='hard_sigmoid',return_sequences=False,name='lstm'))
model.add(Dense(data.__len__(),activation='sigmoid'))
model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

model.fit(X_train, Y_train, batch_size=100, nb_epoch=60)

get_lstm_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(name='lstm').output])


y=model.predict(x=X_train)
labels=[np.argmax(i) for i in y]
print labels
lstmout=get_lstm_layer_output([X_train])[0]
print lstmout
'''
#对句子向量降维
redim=Sequential()
redim.add(Dense(2,activation='linear',name='encoder',input_dim=lstmout[0].__len__()))
redim.add(Dense(lstmout[0].__len__(),activation='linear'))
redim.compile(optimizer='RMSprop',
              loss='mse',
              metrics=['mean_squared_error'])

redim.fit(x=lstmout,y=lstmout,nb_epoch=200,batch_size=1)

#reduce dimension
get_encode_layer_output= K.function([redim.layers[0].input],
                                  [redim.get_layer(name='encoder').output])
coordinate=([lstmout])[0]


#cluster
kmeans=clu.KMeans(n_clusters=20,max_iter=1000)
y_pred=kmeans.fit_predict(coordinate)
saveFile="output"
w=open(saveFile,'w')
tmp=[]
for i,v in enumerate(text):
    tmp.append(str(y_pred[i])+'\t'+v)
tmp=sorted(tmp)
for i in tmp:
    w.write(i)
w.close()

#paint
plt.scatter(coordinate[:,0],coordinate[:,1],c=y_pred)
plt.hold()
plt.show()
plot(model, to_file='model.png')
'''

kmeans=clu.KMeans(n_clusters=24,max_iter=500)
y_pred=kmeans.fit_predict(lstmout)
print y_pred

saveFile="output"
w=open(saveFile,'w')
tmp=[]
for i,v in enumerate(text):
    tmp.append(str(y_pred[i])+'\t'+v)
tmp=sorted(tmp)
for i in tmp:
    w.write(i)
