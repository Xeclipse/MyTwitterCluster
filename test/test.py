'''
from collections import Counter

a=[1,2,3,4,5,6,7,8,9]
cou=Counter(a)
print cou[1]
'''

file=open('../output/FSD_Importance')
tmp=[]
for i in file.readlines():
    k=i.strip().split(':')
    tmp.append([k[0],float(k[1])])
k= sorted(tmp, cmp=lambda x,y: x[0]<y[0] , reverse=True)
print k