from matplotlib import pyplot as plt
from collections import Counter
import numpy as np

#to analysis the cluster result
def readLabels(file):
    file = open(file)
    labels = [int(i) for i in file.readlines()]
    file.close()
    return labels
def PurityAnalysis(y_pred,y_true,n_cluster):
    cluster = []
    for i in range(n_cluster+1):
        cluster.append([])
    for i in range(y_true.__len__()):
        cluster[y_pred[i]].append(y_true[i])
    sta = []
    for i in cluster:
        if i.__len__() == 0: continue
        c = Counter(i)
        sta.append(c)
    maxNumId = []
    purity = []
    for c in sta:
        x = c.keys()
        y = c.values()
        bigId = np.argmax(y)
        maxNumId.append(x[bigId])
        purity.append(float(y[bigId]) / sum(y))
    plt.bar(left=[0.1 + i * (0.5 + 0.1) for i in range(purity.__len__())], width=0.5, height=purity)
    plt.show()
    return maxNumId, purity

