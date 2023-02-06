import pickle
import numpy as np
import random
import math
random.seed(1999)
from scipy.spatial.distance import cdist
from scipy import spatial
fname1='output/linkresult_test_1/embed_ent.pkl'
fname2='../file/linkresult_test/self.ent2id'
fname3='../file/linkresult_test/self.id2ent'

fname4='data/myexp/numbertrueDict.txt'
def sigmoid(num):
    return 1/(1+math.exp(-num))
k=1
def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_theta = 0.5 + 0.5 * cos_theta
    return cos_theta

ent2embed = pickle.load(open(fname1, 'rb'))
ent2id = pickle.load(open(fname2, 'rb'))
id2ent = pickle.load(open(fname3, 'rb'))

resdict=dict()
file4=open(fname4,'r',encoding='utf-8')
for line in file4:
    dict1=eval(line)
    resdict.update(dict1)

ambicount=0

truecount=0
totalcount=0

result=dict()
totalcandi=[str(i) for i in range(21856,1045274)]


confidence=dict()
errcount=0
#simmat=1-cdist(Lvec,Rvec,metric='cosine').astype(np.float32)
#sim=spatial.distance.cdist(Lvec,Rvec,metric='cosine')

for i in range(21856):
    print(i)
    queryid=ent2id[str(i)]
    queryemb=ent2embed[queryid]
    if str(i) in resdict.keys():
        candidates=resdict[str(i)]
        scoredict=dict()
        for ent in candidates.keys():
            emb=ent2embed[ent2id[ent]]
            sim=cos_sim(queryemb,emb)
            scoredict[ent]=sim
        mostcloset=max(scoredict,key=scoredict.get)
        result[str(i)]=str(mostcloset)
        if len(scoredict)==1:
            confidence[str(i)]=1
        else:
            x1=scoredict[sorted(scoredict,key=scoredict.get,reverse=True)[0]]
            x2=scoredict[sorted(scoredict,key=scoredict.get,reverse=True)[1]]
            score=sigmoid(k*(x1-x2)/x1)
            confidence[str(i)]=score
            print(scoredict)
    else:
        errcount+=1
        scoredict=dict()
        for ent in totalcandi:
            emb = ent2embed[ent2id[ent]]
            sim = cos_sim(queryemb, emb)
            scoredict[ent] = sim
        mostcloset = max(scoredict, key=scoredict.get)
        result[str(i)] = str(mostcloset)

        x1=scoredict[sorted(scoredict,key=scoredict.get,reverse=True)[0]]
        x2=scoredict[sorted(scoredict,key=scoredict.get,reverse=True)[1]]
        score=sigmoid(k*(x1-x2)/x1)
        confidence[str(i)]=score
        print(x1,x2)
        print(score)

pickle.dump(result, open('linkresult', 'wb'))

pickle.dump(confidence, open('linkresultconfidence', 'wb'))
