import pickle
import itertools
import math

result = pickle.load(open('linkresult', 'rb'))
confidence = pickle.load(open('linkresultconfidence', 'rb'))

inv_map={}
print(len(result))

k=10

for k, v in result.items():
    if math.log(1+math.exp(confidence[k]),math.e)>0.95+0.05*math.exp(-k/10):#threshold
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)


dst=open('myexp/softcanseed.txt', "w", encoding='utf-8')


for ent in inv_map:
    if len(inv_map[ent])>1:
        cc=list(itertools.combinations(inv_map[ent],2))
        for tuple in cc:
            a=tuple[0]
            b=tuple[1]
            #confid=confidence[a]*confidence[b]
            confid = (confidence[a] + confidence[b])/2
            dst.writelines(a+'#####'+b+'#####'+str(confid)+'\n')

