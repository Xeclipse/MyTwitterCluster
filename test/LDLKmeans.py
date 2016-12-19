#coding: utf-8
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Layer, Merge, Lambda,merge
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import preprocess.DataPre as dp
import random
import keras.constraints as kc
import keras.backend as K
import tensorflow as tf


#Loss= sum_i^N( sum_k^K (|| x_i - y_ik ＊ mu_k  ||_2/ || x_i ||2) )
#mu_k= sum_i^N( x_i * y_ik )/N

dim=2#sample维数
clusterNum=20#每个cluster的sample数量
clusters=4#cluster的数量
sampleSize=clusterNum*clusters
alpha=clusters

#generate data centers
dataCentre=np.random.multivariate_normal([5]*dim , np.eye(dim),clusters)
dataCentre= dataCentre.__mul__([10]*dim)



#generate data
labels=[]
data=None
for i in range(clusters):
    if data is None:
        data =np.random.multivariate_normal(dataCentre[i],np.eye(dim),clusterNum)
    else:
        data=np.concatenate( (data, np.random.multivariate_normal(dataCentre[i],np.eye(dim)*2,clusterNum)), axis=0 )
    labels.extend([[0]*i+[1]+[0]*(clusters-1-i)]*clusterNum)
X_train=data

#init random labels
tmpLabel=[random.randint(0,clusters-1) for i in range(clusters*clusterNum)]
Y_train=dp.prepLabel(tmpLabel,clusters)


#paint data
'''
colors = [i.index(1) for i in labels]
plt.scatter(data[:,0],data[:,1])
plt.hold()
plt.show()
plt.scatter(data[:,0],data[:,1],c=colors)
plt.hold()
plt.show()
'''

#Loss= sum_i^N( sum_k^K (|| x_i - y_ik ＊ mu_k  ||_2/ || x_i ||2) )
#mu_k= sum_i^N( x_i * y_ik )/N

def LossFunction(x):
    '''
    dat=tf.reshape(x[0],[sampleSize,dim])
    label=tf.reshape(x[1],[sampleSize,alpha])
    mu=tf.transpose(tf.scalar_mul(1.0/sampleSize,K.dot(tf.transpose(dat),label)))
    loss=(dat-(tf.matmul(label,mu)))
    loss=tf.trace(tf.matmul(loss,tf.transpose(loss)))
    return loss
    '''
    dat = tf.reshape(x[0], [sampleSize, dim])
    label = tf.reshape(x[1], [sampleSize, alpha])
    loss=tf.matmul(tf.matmul(label,tf.transpose(label)),dat)-dat
    loss = tf.matmul(loss, tf.transpose(loss))
    loss=tf.trace(loss)
    loss=tf.reshape(loss,(1,1))
    return loss


#cluster network

labelID=Input(batch_shape=(1,sampleSize))
labelEmbedding=Embedding(input_dim=sampleSize,  input_length=sampleSize, output_dim=alpha, W_constraint=kc.unitnorm())(labelID)
rawInput=Input(batch_shape=(1,sampleSize*dim))
#rawInput=Dense(input_dim=sampleSize*dim , output_dim=sampleSize* dim, weights=np.eye(sampleSize*dim),activation='linear', trainable=False)(rawInput)
out=merge(inputs=[rawInput,labelEmbedding], mode= LossFunction, output_shape=(1,1))

model=Model(input=[rawInput,labelID],output=out)

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['mse'])

X=X_train.reshape((sampleSize*dim))
X=np.asanyarray([X,X])
X2=np.asanyarray(range(X_train.__len__())).reshape(sampleSize)
X2=np.asanyarray([X2,X2])
target=np.asanyarray([[[0.0]],[[0.0]]])
model.fit(x=[X,X2], y=target ,batch_size=1)

'''
plt.scatter(data[:,0],data[:,1],c=colors)
plt.hold()
plt.show()
'''