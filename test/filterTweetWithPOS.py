#coding: utf-8
import nltk as nt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
tk=nt.TweetTokenizer(strip_handles=True)
s1='Betty Ford Dead At 93 - Local News - Spokane , WA - msnbc.com via '
words = tk.tokenize(s1)
print words
tagged=nt.pos_tag(words)
print tagged
print nt.ne_chunk(tagged)

reserve=['JJ', 'NNP','NN','CD', 'VB', 'VBP']
file=open('../resource/fsd/raw_fsd_tweets')
t=file.readlines()
file.close()
file=open('../resource/fsd/processed_fsd_tweets_filter','w')
for l in t:
    words = tk.tokenize(l)
    tagged = nt.pos_tag(words)
    #nt.ne_chunk(tagged)
    '''
    s=''
    for w in tagged:
        if w.find('http')>=0:
            s+=w+' '
    s+='\n'
    '''
    s = ''
    for w in tagged:
        if w[1] in reserve and not w[0].find('http')>=0:
            s += str(w[0]) + ' '
    s += '\n'
    file.write(s)
file.close()
