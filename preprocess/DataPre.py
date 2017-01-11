#coding: utf-8
import numpy as np
import keras.preprocessing.text as prep
import keras.preprocessing.sequence as seq
import sqlite3 as sql
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


corpusFile="../resource/fsd/pure_tweets_fsd"
labelsFile='../resource/fsd/labels'

def readSQL(file='../resource/fsd/fsd_relevent.db', savefile='../resource/fsd/raw_fsd_tweets'):
    conn=sql.connect(file)
    print 'sucessfunlly open database'
    cursor=conn.execute('select text from tweets')
    conn.close()
    file=open(savefile,'w')
    for i in cursor:
        file.write(str(i[0]).replace('\n','').replace('\r\n','')+'\n')
    file.close()

def dataText2Seq(fileName="../resource/fsd/pure_tweets_fsd", nb=1900):
    file = open(fileName)
    text = [i.strip() for i in file.readlines()]
    toknizer = prep.Tokenizer(nb_words=nb)
    toknizer.fit_on_texts(texts=text)
    data = toknizer.texts_to_sequences(texts=text)
    data = np.asanyarray(data)
    data = seq.pad_sequences(sequences=data, padding='pre')
    maxlen = [i.__len__() for i in data]
    maxlen = maxlen[np.argmax(maxlen)]
    file.close()
    return data,toknizer,maxlen


#trans label from 1 dim to k dim, bag of label :)
def prepLabel(tmp,LabelNum):
    labels=[]
    for i in tmp:
        k=[0]*LabelNum
        k[i]=1
        labels.append(k)
    return labels

def isTagged(l):
    if l.find('#')>=0 and l[l.find('#')+1]!=' ':
        return True
    return False

def saveList(file,l, spt=' '):
    f=open(file,'w')
    for i in l:
        f.write(str(i).replace('[','').replace(']','').replace(',',spt).replace('\n','')+'\n')
    f.close()

def extractHashTag(f='../resource/fsd/raw_fsd_tweets', savefile='../resource/fsd/fsd_hashTag'):
    file=open(f)
    allHashTag=[]
    for l in file.readlines():
        words=l.strip().split(' ')
        t=[]
        for w in words:
            if w.__len__()>1 and w[0]=='#':
                t.append(w)
        allHashTag.append(t)
    file.close()
    saveList(savefile,allHashTag)