from matplotlib import pyplot as plt
from collections import Counter
import numpy as np



#to analysis the cluster result
labelsFile='../resource/toySet/label'
predictLabelsFile='../resource/toySet/kmeans_ans'
file=open(labelsFile)
labels=[int(i)-1 for i in file.readlines()]
file.close()


file=open(predictLabelsFile)
plabels=[int(i) for i in file.readlines()]
file.close()
cluster=[]
for i in range(30):
    cluster.append([])
for i in range(labels.__len__()):
    cluster[plabels[i]].append(labels[i])

sta=[]
for i in cluster:
    if i.__len__()==0: continue
    c=Counter(i)
    x=c.keys()
    y=c.values()
    sta.append(c)
    #plt.pie(y)
    #plt.show()

maxNumId=[]
purity=[]
for c in sta:
    x = c.keys()
    y = c.values()
    bigId=np.argmax(y)
    maxNumId.append(x[bigId])
    purity.append(float(y[bigId])/sum(y))


print maxNumId
print purity
print np.mean(purity)
plt.bar(left=[0.1+i*(0.5+0.1) for i in range(purity.__len__())],width=0.5, height=purity)
plt.show()