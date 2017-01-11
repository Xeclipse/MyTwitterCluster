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
from keras.constraints import maxnorm
from keras.regularizers import l2
data,toknizer,maxlen = DataPre.dataText2Seq("../resource/toySet/sample",None)
nword=toknizer.word_index.__len__()

#一句话中的向量尽可能相等
#不同的话的向量尽可能不等

data=seq.pad_sequences(sequences=data,padding='pre')
X_train=data
y_train1=np.asanyarray([[[0]*100]*maxlen]*data.__len__())
y_train2=np.asanyarray([[0]*30]*data.__len__())

def LossSenWord(y_true, y_pred):
    mean, variance= tf.nn.moments(y_pred, axes=[1],  name=None, keep_dims=False)
    #return tf.nn.l2_loss(variance)
    return tf.scalar_mul(-1.0, tf.nn.l2_loss(variance))
def LossAllsen1(y_true, y_pred):
    mean, variance= tf.nn.moments(y_pred, axes=[0],  name=None, keep_dims=False)
    #return tf.scalar_mul(tf.nn.l2_loss(variance))
    return tf.nn.l2_loss(variance)
def LossAllsen(y_true, y_pred):
    mean1, variance1= tf.nn.moments(y_pred, axes=[1],  name=None, keep_dims=False)
    mean2, variance2 = tf.nn.moments(mean1, axes=[0], name=None, keep_dims=False)
    # x1 = tf.nn.l2_loss(variance1)
    #x2=tf.scalar_mul(-1.0,tf.nn.l2_loss(variance2))
    x1 = tf.scalar_mul(-1.0, tf.nn.l2_loss(variance1))
    x2 = tf.nn.l2_loss(variance2)
    return tf.add(x1,x2)


sentenceInput=Input(shape=(maxlen,), name="sentence_sequence")
x=Embedding(output_dim=10,input_dim=nword+2, input_length=maxlen, mask_zero=True,name='embeddingOneSen',W_regularizer=l2(1.0))(sentenceInput)
sentence=GRU(output_dim=20, name='AllSen', activation='sigmoid')(x)
model=Model(input=[sentenceInput],output=[x,sentence])
model.compile(loss={'embeddingOneSen':LossSenWord, 'AllSen':LossAllsen1},
              loss_weights={'embeddingOneSen':1.0, 'AllSen':1.0},
              optimizer='rmsprop')
model.fit(x=X_train, y=[y_train1,y_train2], batch_size=1000, nb_epoch=1000)
get_lstm_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(name='AllSen').output])
lstmout=get_lstm_layer_output([X_train])[0]
file=open("../resource/toySet/sentence_vector",'w')
for i in lstmout:
    file.write(str(i).replace('[','').replace(']','').replace(',',' ').replace('\n','')+'\n')
file.close()


'''
sentenceInput=Input(shape=(maxlen,), name="sentence_sequence")
x=Embedding(output_dim=10,input_dim=nword+2, input_length=maxlen, mask_zero=True,name='embeddingOneSen',W_regularizer=l2(1.0))(sentenceInput)
model=Model(input=[sentenceInput],output=[x])
model.compile(loss=LossAllsen,
              optimizer='rmsprop')
model.fit(x=X_train, y=[y_train1], batch_size=2499, nb_epoch=1)
get_lstm_layer_output = K.function([model.layers[0].input],
                                  [model.get_layer(name='embeddingOneSen').output])
lstmout=get_lstm_layer_output([X_train])[0]
lstmout=np.mean(lstmout,axis=1)
file=open("../resource/toySet/sentence_vector",'w')
for i in lstmout:
    file.write(str(i).replace('[','').replace(']','').replace(',',' ').replace('\n','')+'\n')
file.close()
'''