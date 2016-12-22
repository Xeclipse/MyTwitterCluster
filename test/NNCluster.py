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

all1_K1 =tf.constant([1.0]*alpha)#K*1
simple_11=tf.constant([1.0])

def LossFunction(x):
    dat=tf.reshape(x[0],[1,dim])#1*D
    mu=x[1]#K*D
    a1=tf.reshape(all1_K1,[alpha,1])
    s1=tf.reshape(simple_11,[1,1])
    E=tf.matmul(a1,dat)
    S=tf.sub(dat,E)
    loss=tf.trace(tf.matmul(S,S,transpose_b=True))
    loss=tf.add(loss,simple_11)
    loss=tf.sub(s1,loss)
    return loss


#cluster network
center=Input(shape=(None,alpha))
center=Embedding(input_dim=alpha,  input_length=alpha, output_dim=dim,name='Center')(center)
rawInput=Input(shape=(1,dim))
#rawInput=Dense(input_dim=sampleSize*dim , output_dim=sampleSize* dim, weights=np.eye(sampleSize*dim),activation='linear', trainable=False)(rawInput)
out=merge(inputs=[rawInput,center], mode= LossFunction, output_shape=(1,1))
model=Model(input=[rawInput,center],output=out)
model.compile(optimizer='RMSprop',
              loss='mse',
              metrics=['mse'])


X2=np.asanyarray(range(alpha)*sampleSize).reshape(sampleSize,alpha)
target=np.asanyarray([[[0.0]]])
model.fit(x=[X_train,X2], y=target ,batch_size=1,nb_epoch=10000,verbose=1)


'''
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
'''