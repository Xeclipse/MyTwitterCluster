#coding:utf-8
import nltk as nk

#nk.download()

'''

Sample:
    input:'i love you'
    output:{
        encodeText=[[0,1,2]]
        vocab={I:0, love:1, you:2}
        }
function: trans texts to code
param:{
    file:input file name
    readSize: each turn read how many
    tokenize: a tokenizer, cannot be none
    stem:a stemer, can be none
    }
'''
def TransToVocabulary(file, readSize=100000, tokenize=nk.TweetTokenizer(), stem=None):
    encodeText=[]
    vocabW2I={}
    vocabsize=0
    f=open(file)
    while 1:
        lines=f.readlines(readSize)
        if not lines: break;
        for l in lines:
            words=tokenize.tokenize(l)
            encodeLine = []
            for w in words:
                #stem word part
                if stem:
                    w=stem.stem(w)
                #construct vocab and code line
                if w not in vocabW2I:
                    vocabW2I[w] = vocabsize
                    vocabsize += 1
                encodeLine.append(vocabW2I[w])
                #finish

            encodeText.append(encodeLine)
    vocabI2W={}
    for i in vocabW2I.items():
        vocabI2W[i[1]]=i[0]
    return encodeText , vocabW2I, vocabI2W