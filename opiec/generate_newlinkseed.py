from helper import *
from utils import *
from metrics import evaluate  # Evaluation metrics
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import pickle
import math
from tqdm import tqdm

ave = True
# ave = False

resdict = dict()
fname4 = 'data/myexp/numbertrueDict.txt'
file4 = open(fname4, 'r', encoding='utf-8')
for line in file4:
    dict1 = eval(line)
    resdict.update(dict1)

ent2embed = pickle.load(open('embed_ent.pkl', 'rb'))
dst = open('data/myexp/softlinkseed.txt', "w", encoding='utf-8')


def HAC_getClusters(params, embed, cluster_threshold_real, threshold_or_cluster):
    embed_dim = 300
    dist = pdist(embed, metric=params.metric)

    clust_res = linkage(dist, method=params.linkage)
    if threshold_or_cluster == 'threshold':
        labels = fcluster(clust_res, t=cluster_threshold_real, criterion='distance') - 1
    else:
        labels = fcluster(clust_res, t=cluster_threshold_real, criterion='maxclust') - 1

    clusters = [[] for i in range(max(labels) + 1)]
    for i in range(len(labels)):
        clusters[labels[i]].append(i)

    clusters_center = np.zeros((len(clusters), embed_dim), np.float32)
    for i in range(len(clusters)):
        cluster = clusters[i]
        if ave:
            clusters_center_embed = np.zeros(embed_dim, np.float32)
            for j in cluster:
                embed_ = embed[j]
                clusters_center_embed += embed_
            clusters_center_embed_ = clusters_center_embed / len(cluster)
            clusters_center[i, :] = clusters_center_embed_
        else:
            sim_matrix = np.empty((len(cluster), len(cluster)), np.float32)
            for i in range(len(cluster)):
                for j in range(len(cluster)):
                    if i == j:
                        sim_matrix[i, j] = 1
                    else:
                        if params.metric == 'cosine':
                            sim = cos_sim(embed[i], embed[j])
                        else:
                            sim = np.linalg.norm(embed[i] - embed[j])
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
            sim_sum = sim_matrix.sum(axis=1)
            max_num = cluster[int(np.argmax(sim_sum))]
            clusters_center[i, :] = embed[max_num]
    # print('clusters_center:', type(clusters_center), clusters_center.shape)
    return labels, clusters_center


def entropy(dic):
    sum = 0
    for ent, count in dic.items():
        sum += count
    result = 0
    for ent, count in dic.items():
        p = count / sum
        result -= p * math.log(p)
    return result


def compactness(dic, ent2id, id2ent):
    # method 1 average distance to cluster center
    clusters_center_embed = np.zeros(300, np.float32)
    for noun in dic:
        clusters_center_embed += ent2embed[ent2id[noun]]
    clusters_center_embed_ = clusters_center_embed / len(dic)
    distance = 0
    for noun in dic:
        distance += cosine_distance(clusters_center_embed_, ent2embed[ent2id[noun]])
    meandistance = distance / len(dic)

    return meandistance


def negexp(num):
    return math.exp(-num)


def check(myclust2ent, ent2id, id2ent):
    result = pickle.load(open('linkresult', 'rb'))
    # result = pickle.load(open('firstcan0.53', 'rb'))
    confidence = pickle.load(open('linkresultconfidence', 'rb'))
    # confidence = pickle.load(open('firstcan0.53confidence', 'rb'))
    count = dict(dict())

    threshold = 0
    clusterentropy = {}
    clustcompactness = {}
    maxcount = {}
    clusterconfidence = {}
    errorcount = 0
    for label in myclust2ent.keys():
        count[label] = {}
        clust = myclust2ent[label]
        for np in clust:
            if np in result:
                link = result[np]
                if link in count[label]:
                    count[label][link] += confidence[np]
                else:
                    count[label][link] = confidence[np]
            else:
                errorcount += 1
        if len(count[label]) > 0:
            clusterentropy[label] = entropy(count[label])
            clustcompactness[label] = compactness(myclust2ent[label], ent2id, id2ent)
            maxcount[label] = max(count[label], key=lambda k: count[label][k])
            clusterconfidence[label] = negexp(entropy(count[label]))

    print(count)
    print(maxcount)
    print(errorcount)

    print(clustcompactness)
    print(clusterentropy)
    print(clusterconfidence)
    labels = []
    error = 0
    for label, v in clusterentropy.items():
         if math.log(1+math.exp(-v),math.e)>0.3+0.1*math.exp(-10/10) :
            labels.append(label)
    print(len(labels))
    for label in labels:
        print(label, myclust2ent[label])
    for label in labels:
        for np in myclust2ent[label]:
            if np not in resdict.keys():
                dst.writelines(np + '#####' + maxcount[label] + '#####' + str(clusterconfidence[label]) + '\n')
            else:
                if maxcount[label] in resdict[np].keys():
                    dst.writelines(np + '#####' + maxcount[label] + '#####' + str(clusterconfidence[label]) + '\n')
                else:
                    error += 1
    print(error)



def cluster_test(params, side_info, cluster_predict_list, true_ent2clust, true_clust2ent, print_or_not=False):
    sub_cluster_predict_list = []
    clust2ent = {}
    isSub = side_info.isSub
    triples = side_info.triples
    ent2id = side_info.ent2id
    id2ent = side_info.id2ent

    for eid in isSub.keys():
        sub_cluster_predict_list.append(cluster_predict_list[eid])
    # print(sub_cluster_predict_list)#2195 Sub NP->561 Cluster

    for sub_id, cluster_id in enumerate(sub_cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(sub_id)
        else:
            clust2ent[cluster_id] = [sub_id]
    cesi_clust2ent = {}

    for rep, cluster in clust2ent.items():
        # cesi_clust2ent[rep] = list(cluster)
        cesi_clust2ent[rep] = set(cluster)
    cesi_ent2clust = invertDic(cesi_clust2ent, 'm2os')
    clust2ent = {}
    for sub_id, cluster_id in enumerate(sub_cluster_predict_list):
        if cluster_id in clust2ent.keys():
            clust2ent[cluster_id].append(id2ent[sub_id])
        else:
            clust2ent[cluster_id] = [id2ent[sub_id]]
    myclust2ent = {}
    for rep, cluster in clust2ent.items():
        # cesi_clust2ent[rep] = list(cluster)
        myclust2ent[rep] = set(cluster)

    print(myclust2ent)
    check(myclust2ent, ent2id, id2ent)
    cesi_ent2clust_u = {}
    if params.use_assume:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]
    else:
        for trp in triples:
            sub_u, sub = trp['triple_unique'][0], trp['triple_unique'][0]
            cesi_ent2clust_u[sub_u] = cesi_ent2clust[ent2id[sub]]

    cesi_clust2ent_u = invertDic(cesi_ent2clust_u, 'm2os')
    eval_results = evaluate(cesi_ent2clust_u, cesi_clust2ent_u, true_ent2clust, true_clust2ent)
    macro_prec, micro_prec, pair_prec = eval_results['macro_prec'], eval_results['micro_prec'], eval_results[
        'pair_prec']
    macro_recall, micro_recall, pair_recall = eval_results['macro_recall'], eval_results['micro_recall'], eval_results[
        'pair_recall']
    macro_f1, micro_f1, pair_f1 = eval_results['macro_f1'], eval_results['micro_f1'], eval_results['pair_f1']
    ave_prec = (macro_prec + micro_prec + pair_prec) / 3
    ave_recall = (macro_recall + micro_recall + pair_recall) / 3
    ave_f1 = (macro_f1 + micro_f1 + pair_f1) / 3
    model_clusters = len(cesi_clust2ent_u)
    model_Singletons = len([1 for _, clust in cesi_clust2ent_u.items() if len(clust) == 1])
    gold_clusters = len(true_clust2ent)
    gold_Singletons = len([1 for _, clust in true_clust2ent.items() if len(clust) == 1])
    if print_or_not:
        print('Ave-prec=', ave_prec, 'macro_prec=', macro_prec, 'micro_prec=', micro_prec,
              'pair_prec=', pair_prec)
        print('Ave-recall=', ave_recall, 'macro_recall=', macro_recall, 'micro_recall=', micro_recall,
              'pair_recall=', pair_recall)
        print('Ave-F1=', ave_f1, 'macro_f1=', macro_f1, 'micro_f1=', micro_f1, 'pair_f1=', pair_f1)
        print('Model: #Clusters: %d, #Singletons %d' % (model_clusters, model_Singletons))
        print('Gold: #Clusters: %d, #Singletons %d' % (gold_clusters, gold_Singletons))
        print()

    return ave_prec, ave_recall, ave_f1, macro_prec, micro_prec, pair_prec, macro_recall, micro_recall, pair_recall, \
           macro_f1, micro_f1, pair_f1, model_clusters, model_Singletons, gold_clusters, gold_Singletons