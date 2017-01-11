

import numpy as np
import  random
import preprocess.DataPre as dp


nsample=1000
samplelen=[random.randint(10,15) for i in range(nsample)]

width=12
repeat=3
start=width-repeat
nmods=4
mods=[(i*start,i*start+width) for i in range(1,nmods+1)]

sentences=[]
labels=[]
for len in samplelen:
    m=random.randint(0,nmods-1)
    labels.append(m)
    (a,b)=mods[m]
    sentences.append([random.randint(a,b) for i in range(len)])

dp.saveList('../resource/toySet/sample',sentences)
dp.saveList('../resource/toySet/label',labels)