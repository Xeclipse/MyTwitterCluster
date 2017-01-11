from sklearn.cluster import KMeans



file=open("../resource/toySet/sentence_vector")
ls=file.readlines()
file.close()
features=[]
for i in ls:
    i=i.strip()
    i=i.replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ')
    features.append( [float(k) for k in i.split(' ')])

m=KMeans(n_clusters=4,max_iter=500)
v= m.fit_predict(features)
file=open("../resource/toySet/kmeans_ans",'w')
for i in v:
    file.write(str(i)+'\n')
file.close()