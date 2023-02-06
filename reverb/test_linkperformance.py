import pickle

truecount=0
totalcount=0
samenamecount=0
reducecount=0
result1=pickle.load(open('linkresult', 'rb'))


dictent=set()
file4=open('data/myexp/numbertrueDict.txt','r',encoding='utf-8')
for line in file4:
    dict1=eval(line)
    for key in dict1.keys():
        dictent.add(key)

ckbname = dict()
with open('data/myexp/ckbname2id.txt', "r", encoding='utf-8') as f:
    for line in f:
        res = line.strip('\n').split('#####')
        name = res[0]
        entnumber = res[1]
        ckbname[entnumber] = name

npname = dict()
with open('data/myexp/NPid.txt', "r", encoding='utf-8') as f:
    for line in f:
        res = line.strip('\n').split('#####')
        name = res[0]
        entnumber = res[1]
        npname[entnumber] = name


ckbdict = dict()
with open('data/myexp/ckbentid.txt', "r", encoding='utf-8') as f:
    for line in f:
        res = line.strip('\n').split('#####')
        qid = res[0]
        entnumber = res[1]
        ckbdict[qid] = entnumber

with open('data/myexp/cleantesttriples.txt', "r", encoding='utf-8') as f:
    for line in f:
        res = line.strip('\n').split('#####')
        num = res[0]
        query1 = res[1]
        query2 = res[3]
        subent = res[4]
        objent = res[5]



        totalcount += 1
        predict1=result1[query1]
        answer1 = ckbdict[subent]
        if answer1 == predict1:
            truecount += 1

print(truecount)
print(totalcount)


print(truecount/totalcount)
