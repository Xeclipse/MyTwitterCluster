#coding: utf-8
import numpy as np
import keras.preprocessing.text as prep
import keras.preprocessing.sequence as seq
import sqlite3 as sql
import sys
import nltk as nk
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

def dataText2Seq(fileName="../resource/fsd/pure_tweets_fsd", nb=None, padding=None):
    file = open(fileName)
    text = [i.strip() for i in file.readlines()]
    file.close()
    toknizer = prep.Tokenizer(nb_words=nb,lower=False)
    toknizer.fit_on_texts(texts=text)
    data = toknizer.texts_to_sequences(texts=text)
    if padding is not None:
        data = np.asanyarray(data)
        data = seq.pad_sequences(sequences=data, padding=padding)
        maxlen =data[0].__len__()
    else:
        maxlen = [i.__len__() for i in data]
        maxlen = maxlen[np.argmax(maxlen)]
    return data,toknizer,maxlen


#trans label from 1 dim to k dim, bag of label :)
def prepLabel(tmp,LabelNum):
    labels=[]
    for i in tmp:
        k=[0]*LabelNum
        k[i]=1
        labels.append(k)
    return labels

#whether a word is a hash tag
def isTagged(l):
    if l.find('#')>=0 and l[l.find('#')+1]!=' ':
        return True
    return False

#whether a word is a name
def isName(l):
    if l.find('@')>=0 and l[l.find('@')+1]!=' ':
        return True
    return False


#save a list
def saveList(file,l, spt=' '):
    f=open(file,'w')
    for i in l:
        f.write(str(i).replace('[','').replace(']','').replace(',',spt).replace('\n','')+'\n')
    f.close()


#extract hashtag from file
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


#extract name from file
def extractName(f='../resource/fsd/raw_fsd_tweets', savefile='../resource/fsd/fsd_name'):
    file=open(f)
    allHashTag=[]
    for l in file.readlines():
        words=l.strip().split(' ')
        t=[]
        for w in words:
            if w.__len__()>1 and w[0]=='@':
                t.append(w)
        allHashTag.append(t)
    file.close()
    saveList(savefile,allHashTag)



#transform a seqlist to a BOW
def sparseSeq(seq,dim):
    ret = [0]*dim
    for i in seq:
        ret[i] += 1
    return ret

#generate BOW feature
def BoW(fileName="../resource/fsd/pure_tweets_fsd", nb_words=None):
    data, toknizer, maxlen, nb=dataText2Seq(fileName,nb=nb_words)
    dim=toknizer.word_index.__len__()+1
    return [sparseSeq(i,dim) for i in data]


#tokenize sentences
def tokSentence(filename):
    file = open(filename)
    text = file.readlines()
    file.close()
    tok = nk.TweetTokenizer(reduce_len=True, strip_handles=False)
    print 'tokenize tweets---->',
    tokSentences = [tok.tokenize(i) for i in text]
    print 'finish'
    return tokSentences

#posTagSentences
def PosSentence(tokSentences ,posSaveFile = '../resource/fsd/fsd_pos', save=False):
    print 'posTag tweets----->',
    posSentences = [nk.pos_tag(i) for i in tokSentences]
    print 'finish'
    if save:
        print 'Save POS tweets----->',
        saveList(posSaveFile, posSentences)
        print 'finish'
    return posSentences

#NerSentences
def NerSentence(posSentences,nerSaveFile = '../resource/fsd/fsd_ner',save=False):
    print 'NER tweets----->',
    nerSentences = [nk.ne_chunk(i) for i in posSentences]
    print 'finish'
    if save:
        print 'Save NER tweets----->',
        nerSave = open(nerSaveFile, 'w')
        for l in nerSentences:
            s = ''
            for i in l:
                if type(i) is not tuple:
                    s += str(i._label) + ':'
                    for p in i:
                        s += str(p[0]) + '_'
                    s += '\t'
                    nerSave.write(s + '\n')
            nerSave.close()
        print 'finish'
    return nerSentences