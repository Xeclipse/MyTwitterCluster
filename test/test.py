#coding: utf-8
import nltk as nk
import keras.preprocessing.text as prep
import preprocess.DataPre as dpp

'''
file=open('../resource/fsd/fsd_ner')
text=file.readlines()
tok=prep.Tokenizer(lower=False)
tok.fit_on_texts(text)
print tok.word_counts
l=sorted([(i,tok.word_counts[i]) for i in tok.word_counts],key=lambda x:x[1], reverse=True)
'''

'''
tok=prep.Tokenizer(nb_words=None)
tok.fit_on_texts(text)
data=tok.texts_to_sequences(text)
l=sorted([(i,tok.word_index[i]) for i in tok.word_index],key=lambda x:x[1], reverse=False)
print l
for d in data:
    s=''
    for i in d:
        if tok.word_counts[l[i][0]] > 2:
            s+=l[i-1][0]+' '
    s+='\n'
    print s
'''

dpp.extractName()

