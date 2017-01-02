import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Input
from keras.models import Model, Sequential
from scipy import stats

'''
can a neural network provide a line that new points are far away from it
'''
x=np.linspace(start=-10,stop=10,num=500,endpoint=True)
#y1=stats.multivariate_normal.pdf(x,mean=-2.5,cov=1)
#y2=stats.multivariate_normal.pdf(x,mean=2.5,cov=1)
#y=y1+y2
y=x[:]
train=np.transpose([x,y])
target=[1.0]*500
plt.plot(x,y,color='green')

model=Sequential()
model.add(Dense(input_dim=2,output_dim=2,activation='sigmoid'))
model.add(Dense(output_dim=1,activation='sigmoid'))

model.compile(loss='mse',
              optimizer='rmsprop',
              )
model.fit(x=train, y=target, batch_size=5, nb_epoch=60)

x=np.linspace(start=-5,stop=5,num=10,endpoint=True)
y=x[:]*2
pre=np.transpose([x,y])
distance=model.predict(x=pre)
print distance
plt.scatter(x,y,color='blue')
plt.show()

