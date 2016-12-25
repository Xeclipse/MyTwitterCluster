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
from keras.regularizers import l1

#Loss= sum_i^N( sum_k^K (|| x_i - y_ik ＊ mu_k  ||_2/ || x_i ||2) )
#mu_k= sum_i^N( x_i * y_ik )/N

dim=2#sample维数
clusterNum=20#每个cluster的sample数量
clusters=4#cluster的数量
sampleSize=clusterNum*clusters
alpha=clusters

#generate data centers
dataCentre=np.random.multivariate_normal([0]*dim , np.eye(dim),clusters)
dataCentre= dataCentre.__mul__([10]*dim)



#generate data
labels=[]
data=None
for i in range(clusters):
    if data is None:
        data =np.random.multivariate_normal(dataCentre[i],np.eye(dim),clusterNum)
    else:
        data=np.concatenate( (data, np.random.multivariate_normal(dataCentre[i],np.eye(dim)*4,clusterNum)), axis=0 )
    labels.extend([[0]*i+[1]+[0]*(clusters-1-i)]*clusterNum)
X_train=data
np.random.shuffle(X_train)
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
    dat=tf.reshape(x[0],[sampleSize,dim])
    label=tf.reshape(x[1],[sampleSize,alpha])
    minus1=tf.reshape(x[2],[sampleSize,1])
    allOne=tf.div(dat,dat)


    a=tf.matmul(label,dat,transpose_a=True)
    b=tf.matmul(label,allOne,transpose_a=True)
    mu=tf.div(a,b)
    loss=dat-(tf.matmul(label,mu))
    loss=tf.trace(tf.matmul(tf.scalar_mul(1.0/sampleSize,loss),loss,transpose_b=True))


    norm = tf.reduce_sum(label, 1, keep_dims=True)
    norm = tf.scalar_mul( 10.0, tf.add(norm, minus1))
    norm = tf.matmul(norm,norm,transpose_a=True)
    loss= tf.add(loss,norm)

    loss=tf.reshape(loss,(1,1))
    return loss


#cluster network
labelID=Input(batch_shape=(1,sampleSize))
labelEmbedding=Embedding(input_dim=sampleSize,  input_length=sampleSize, output_dim=alpha,W_constraint=kc.nonneg() ,name='embedding')(labelID)
rawInput=Input(batch_shape=(1,sampleSize*dim))
minus1Input=Input(batch_shape=(1,sampleSize))
#rawInput=Dense(input_dim=sampleSize*dim , output_dim=sampleSize* dim, weights=np.eye(sampleSize*dim),activation='linear', trainable=False)(rawInput)
out=merge(inputs=[rawInput,labelEmbedding,minus1Input], mode= LossFunction, output_shape=(1,1))
model=Model(input=[rawInput,labelID,minus1Input],output=out)
model.compile(optimizer='RMSprop',
              loss='mse',
              metrics=['mse'])



X=X_train.reshape((1,sampleSize*dim))
X2=np.asanyarray(range(X_train.__len__())).reshape(1,sampleSize)
X3=np.asanyarray([-1.0]*sampleSize).reshape(1,sampleSize)
target=np.asanyarray([[[0.0]]])
model.fit(x=[X,X2,X3], y=target ,batch_size=1,nb_epoch=10000,verbose=1)


embeddings = model.get_layer(name='embedding').get_weights()[0]

#center
a=np.dot(np.transpose(embeddings),X_train)
b=np.dot(np.transpose(embeddings),np.asanyarray([[1.0]*dim]*sampleSize))
center=np.divide(a,b)
print center
col=[np.argmax(i) for i in embeddings]
print col
plt.scatter(data[:,0],data[:,1],c=col)
plt.scatter(center[:,0],center[:,1],c=['r','r','r','r'])
plt.hold()
plt.show()
