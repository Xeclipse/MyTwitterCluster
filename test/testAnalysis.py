from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
labelsFile='../resource/fsd/labels'
predictLabelsFile='../output/predict_labels'
file=open(labelsFile)
labels=[int(i)-1 for i in file.readlines()]
file.close()


file=open(predictLabelsFile)
plabels=[int(i) for i in file.readline().strip().split(',')]
file.close()
cluster=[]
for i in range(27):
    cluster.append([])
for i in range(labels.__len__()):
    cluster[plabels[i]].append(labels[i])



t=[]
for i in cluster:
    if i.__len__()==0: continue
    c=Counter(i)
    x=c.keys()
    y=c.values()

    t.append( x[np.argmax(y)])
    #plt.pie(y)
    #plt.show()
print set(t).__len__()