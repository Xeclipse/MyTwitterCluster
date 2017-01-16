#coding: utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import keras.preprocessing.text as prep
import keras.preprocessing.sequence as seq
from keras import backend as K
import  sklearn.cluster as clu
import  postprocess.AnsAnalysis as AA
from matplotlib import pyplot as plt
from keras.utils.visualize_util import plot

import preprocess.DataPre as dpp

print 'preparing feature----->',
features = dpp.BoW(fileName='../resource/fsd/processed_tweets_fsd',nb_words=1900)
print 'finish'


print 'Clustering feature----->',
cluster=clu.KMeans(n_clusters=30)
pred=cluster.fit_predict(features)
print 'finish'

print 'Clustering feature----->',
dpp.saveList('../output/fsd/b1900cKn30',l=pred)
print 'finish'

#pred=AA.readLabels('../output/fsd/bow_kmeans_ans')



print 'Analize answer----->',
y_true=AA.readLabels('../resource/fsd/labels')
maxNumId, purity = AA.PurityAnalysis(pred,y_true,30)
print 'finish'
print 'maxNumId',
print maxNumId
print 'purity'
print purity
print set(purity).__len__()