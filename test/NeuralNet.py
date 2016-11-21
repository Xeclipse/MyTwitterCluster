#coding: utf-8
from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt
from keras.utils.visualize_util import plot
import numpy as np
from keras import backend as K
from sklearn.cluster import KMeans
import preprocess





#generate data
filename='test'
data,vocabW2I,vocabI2W=preprocess.RawToVocabulary(filename)



#paint data
colors = [i.index(1) for i in labels]
plt.scatter(data[:,0],data[:,1],c=colors)
plt.hold()
plt.show()


kmeans=KMeans(n_clusters=10,max_iter=1000)
kmeans.fit(data)
ans=kmeans.predict(data)


#network
'''
inputs=Input(shape=(dim,))
x=Dense(50,activation='sigmoid')(inputs)
x=Dense(20,activation='sigmoid')(x)
predictions = Dense(10,activation='sigmoid')(x)
model=Model(input=inputs, output=predictions)
model.compile(optimizer='RMSprop',
              loss='mse',
              metrics=['categorical_accuracy'])
model.fit(data,labels,batch_size=15,nb_epoch=500)

#paint pred
pred=model.predict(data)
maxIndex=np.ndarray.argmax(pred,axis=1)

pred=[]

for i,v in enumerate(maxIndex):
    k=[0]*10
    k[v]=1
    pred.append(k)

colors = [i.index(1) for i in pred]
plt.scatter(data[:,0],data[:,1],c=colors)
plt.hold()
plt.show()
plot(model, to_file='model.png')
'''