#coding: utf-8
import nltk as nk
import preprocess.DataPre as dpp


file=open('../resource/fsd/raw_fsd_tweets')
text=file.readlines()
file.close()


posSaveFile='../resource/fsd/fsd_pos'
nerSaveFile=open('../resource/fsd/fsd_ner','w')
tok=nk.TweetTokenizer(reduce_len=True, strip_handles=False)

'''
sentence='Blast at French Nuclear Site Is Said to Kill 1 Person: An explosion shook a French nuclear waste site in southe... http://t.co/o5f469M'

posSentences=nk.pos_tag(tok.tokenize(sentence))
nerSentences= nk.ne_chunk(posSentences)
print nerSentences
for i in nerSentences:
    if type(i) is not tuple:
        print i._label,
        for p in i:
            print p[0],

'''
print 'tokenize tweets---->',
tokSentences=[tok.tokenize(i) for i in text]
print 'finish'
print 'posTag tweets----->',
posSentences=[nk.pos_tag(i) for i in tokSentences]
print 'finish'

print 'Save POS tweets----->',
dpp.saveList(posSaveFile,posSentences)
print 'finish'

print 'NER tweets----->',
nerSentences= [nk.ne_chunk(i) for i in posSentences]
print 'finish'


print 'Save NER tweets----->',
for l in nerSentences:
    s = ''
    for i in l:
        if type(i) is not tuple:
            s+= str(i._label)+':'
            for p in i:
                s+=str(p[0])+'_'
            s+='\t'
    nerSaveFile.write(s+'\n')
nerSaveFile.close()
print 'finish'