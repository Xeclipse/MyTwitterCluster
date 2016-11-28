import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Input
from keras.models import Model, Sequential
from scipy import stats

'''
try to use neural net work to simulate a distribution
'''
x=np.linspace(start=-10,stop=10,num=500,endpoint=True)
y1=stats.multivariate_normal.pdf(x,mean=-2.5,cov=1)
y2=stats.multivariate_normal.pdf(x,mean=2.5,cov=1)
y=y1+y2
plt.plot(x,y,color='green')


model=Sequential()
model.add(Dense(input_dim=1,output_dim=100,activation='linear'))
model.add(Dense(output_dim=50,activation='sigmoid'))
model.add(Dense(output_dim=25,activation='sigmoid'))
model.add(Dense(output_dim=10,activation='sigmoid'))
model.add(Dense(output_dim=1,activation='sigmoid'))

model.compile(loss='mse',
              optimizer='rmsprop',
              )
model.fit(x=x, y=y, batch_size=1, nb_epoch=60)

x=np.linspace(start=-15,stop=15,num=1000,endpoint=True)
y=model.predict(x=x)
plt.plot(x,y,color='blue')
plt.show()