#coding: utf-8
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Input, Merge, GRU
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.sequence as seq
from preprocess import DataPre
import tensorflow as tf
import keras.backend as K
data,toknizer,maxlen = DataPre.dataText2Seq("../resource/fsd/processed_tweets_fsd",None)
nword=toknizer.word_index.__len__()



data=seq.pad_sequences(sequences=data,padding='pre')
X_train=data
y_train1=np.asanyarray([[[0]*100]*maxlen]*data.__len__())
y_train2=np.asanyarray([[0]*30]*data.__len__())

def Loss1sen(y_true, y_pred):
    mean, variance= tf.nn.moments(y_pred, axes=[1], shift=None, name=None, keep_dims=False)
    return tf.nn.l2_loss(tf.reduce_mean(variance))
def LossAllsen(y_true, y_pred):
    mean, variance= tf.nn.moments(y_pred, axes=[0], shift=None, name=None, keep_dims=False)
    return tf.scalar_mul(-1.0,tf.nn.l2_loss(variance))



sentenceInput=Input(shape=(maxlen,), name="sentence_sequence")
x=Embedding(output_dim=100,input_dim=nword+2, input_length=maxlen, mask_zero=True,name='embeddingOneSen')(sentenceInput)
sentence=GRU(output_dim=30, name='AllSen', activation='sigmoid')(x)
model=Model(input=[sentenceInput],output=[x,sentence])
model.compile(loss={'embeddingOneSen':Loss1sen, 'AllSen':LossAllsen},
              optimizer='rmsprop')
model.fit(x=X_train, y=[y_train1,y_train2], batch_size=2499, nb_epoch=500)
get_lstm_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(name='AllSen').output])
lstmout=get_lstm_layer_output([X_train])[0]
file=open("../resource/fsd/sentence_vector",'w')
for i in lstmout:
    file.write(str(i).replace('[','').replace(']','').replace(',',' ').replace('\n','')+'\n')
file.close()






'''
y=model.predict(x=X_train)
labels=[np.argmax(i) for i in y]
print labels
lstmout=get_lstm_layer_output([X_train])[0]
print lstmout
'''