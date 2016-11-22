import numpy as np
from scipy import stats
from collections import Counter
from matplotlib import pyplot as plt



'''
k = [int(i) for i in np.random.uniform(1,25,100000)]
print k
c=Counter(k)
print c
plt.bar(left=[0.1+i*1.1 for i in range(c.values().__len__())],height=c.values())
plt.show()
'''

'''
def processLabels(labels, val):
    tmplabel=labels[:]
    for i,v in enumerate(tmplabel):
        if v==val: tmplabel[i]=1
        else: tmplabel[i]=0
    return tmplabel

labels=[1,2,2,2,3,4,5,6,1]
print processLabels(labels,2)
print labels
'''


a=[0.48,0.56,0.22,0.33,0.14]
print np.random.multinomial(1,pvals=[i/sum(a) for i in a])