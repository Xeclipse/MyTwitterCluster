'''
from collections import Counter

a=[1,2,3,4,5,6,7,8,9]
cou=Counter(a)
print cou[1]
'''

file=open('../output/FSD_Importance')
tmp=[(i.strip().split(':')) for i in file.readlines()]
file.close()
for i in tmp:
    i[1]=float(i[1])
k= sorted(tmp,cmp=lambda x,y : cmp(x[1],y[1]),reverse=True)
file=open('../output/FSD_Importance_sorted','w')
for i in k:
    file.write(i[0]+':'+str(i[1])+'\n')
