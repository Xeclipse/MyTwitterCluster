#coding: utf-8
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Layer, Merge
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import preprocess.DataPre as dp
import random
#神经网络根本不能分辨什么样的数据是相似的,它只会不停的把数据分到同一个类中
#但是如果数据的中心点是对的呢? 关键是数据的中心点要找对,如果数据中心点是对的话,那么神经网络就能很好的进行聚类,从某种角度来说,神经网络可以用来聚类

dim=2#sample维数
clusterNum=20#每个cluster的sample数量
clusters=4#cluster的数量


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
colors = [i.index(1) for i in labels]
plt.scatter(data[:,0],data[:,1])
plt.hold()
plt.show()
plt.scatter(data[:,0],data[:,1],c=colors)
plt.hold()
plt.show()


#cluster network
model = Sequential()
model.add(Dense(input_dim=dim, output_dim=500,activation='sigmoid'))
model.add(Dense(output_dim=clusters,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

#want to update neural network parameters in small sampling size, but it's not work
'''
col=[]
batch=50
nCount=0
def oneIteration(model,batch,X_train,Y_train):
    tmpAns=Y_train
    nIndex = 0
    sampleSize=X_train.__len__()
    labesTMP=[-1]*sampleSize
    while nIndex<sampleSize:
        tmpAns = model.predict(X_train)
        for i in range(batch):
            maxPro=np.max(tmpAns[nIndex])
            if maxPro>0.6:
                k = [0] * clusters
                k[np.argmax(tmpAns[nIndex])] = 1
                tmpAns[nIndex]=k
                labesTMP[nIndex]=np.argmax(tmpAns[nIndex])
            nIndex+=1
            if(nIndex>= sampleSize): break
        model.fit(x=X_train, y=tmpAns, batch_size=10, nb_epoch=1)
        if nIndex%100==0:
            print '-----------------------------',
            print nIndex
        if (nIndex >= sampleSize): break
    print labesTMP
    return tmpAns
#model.fit(x=X_train, y=Y_train, batch_size=10, nb_epoch=10)
#tmpAns=model.predict(X_train)
predict=oneIteration(model,batch,X_train, Y_train)
col=[ np.argmax(i)  for i in predict]
print col
plt.scatter(data[:,0],data[:,1],c=col)
plt.hold()
plt.show()
'''

#want to stream data, but it's not work

eachIterRound=20
tmpTrain=[]
tmpLabelM=[]
tmpLabels=[]
tmpLabels.extend([0,1,2,3])
tmpTrain.append(X_train[0])
tmpTrain.append(X_train[20])
tmpTrain.append(X_train[40])
tmpTrain.append(X_train[60])
TrainData=np.asanyarray(tmpTrain)
tmpLabelM=dp.prepLabel(tmpLabels,clusters)
model.fit(TrainData, tmpLabelM, batch_size=1, nb_epoch=500,verbose=1)
#threhold=0.8
count=1
#batch=5 # each time i will add batch number data into dataset
for i in range(0,X_train.__len__()):
    print i,
    tmpLabelM = dp.prepLabel(tmpLabels, clusters)
    model.fit(TrainData, tmpLabelM, batch_size=(tmpTrain.__len__() / eachIterRound) + 1, nb_epoch=30,verbose=0)
    tmpTrain.append(X_train[i])
    TrainData = np.asanyarray(tmpTrain)
    y=model.predict(TrainData)
    pval=y[-1]#new instance class probability
    #if p[i]>threhold, assign it, else chose a new
    flagBiggerThre=False
    print np.max(pval),
    print np.argmax(pval)

    #for p in range(count):
    #    if pval[p]>=threhold:
    #        flagBiggerThre=True
    #        break
    #if flagBiggerThre:
    tmpLabels.append(np.argmax(pval))
    #else:
    #    if count<clusters:
    #        tmpLabels.append(count)
    #        count += 1
    #        print 'new Count',
    #        print count
    #    else:
    #        tmpLabels.append(count-1)
    #        print 'run out of labels'

predict=model.predict(X_train)
col=[ np.argmax(i)  for i in predict]


#iterative
'''
changes=1000
while changes>1:
    model.fit(x=X_train, y=Y_train, batch_size=10, nb_epoch=10)
    tmpAns=model.predict(X_train)
    ttmpLabel=[np.argmax(i) for i in tmpAns]
    changes=0
    for i in range(ttmpLabel.__len__()):
        if ttmpLabel[i]!=tmpLabel[i]:
            changes+=1
    st=set(ttmpLabel)
    print '========================================'
    print st
    print st.__len__()
    print '========================================'
    tmpLabel=ttmpLabel
    col=tmpLabel
    Y_train=dp.prepLabel(tmpLabel,clusters)
'''
plt.scatter(data[:,0],data[:,1],c=col)
plt.hold()
plt.show()
