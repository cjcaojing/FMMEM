import community  # Make sure you have the python-louvain library installed

import igraph as ig
from itertools import permutations
# import cython_function as cfunc
import networkx as nx
# import geatpy as ea # import geatpy
import pandas as pd
import time
import find_motifs as fm
import numpy as np
import copy
from igraph.clustering import VertexClustering
from community import community_louvain
# import fai_m as fai_m_func
def motif_xuanze(motif):
    M_ins = nx.Graph()####模体按照Benson论文中设置
    if motif == "motif_1":
        # motif_7 = nx.DiGraph()
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2, 3), (1, 3)])  # 连通
    elif motif == "motif_2":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2, 3)])  # 连通
    elif motif == "motif_3":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 2), (2, 3), (2, 4)])  # 连通
    elif motif == "motif_4":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 2), (2, 3), (3, 4)])  # 连通
    elif motif == "motif_5":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 4)])  # 连通
    elif motif == "motif_6":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 4), (3, 2)])  # 连通
    elif motif == "motif_7":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])  # 连通
    elif motif == "motif_8":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 4), (3, 2), (3, 4)])  # 连通
    return M_ins

def motif_jiaquan(motif):

    '''
    #模体在网络中的数量
    motif_number = fm.total_motif_num(G, motif_1, directed=True, weighted=False)#总模体数
    print('总摸体数量', motif_number)
    '''
    
    # motif = motif_7
    #模体在网络中的节点集合和边集合
    motif_node_set, motif_edge_set = fm.total_motif_list(G, motif, directed=False, weighted=False) # 获得所有模体的节点和边
    # print('模体点集合', motif_node_set)
    # print('模体边集合',motif_edge_set)
    
    
    edge_all = Gi.get_edgelist()
   
    df = pd.DataFrame({ 'edge': edge_all})
    # index_edge = list(range(M))
    # a = df[df.edge==(7,8)].index.tolist()[0]
    # print(a)
    #网络中每条边的模体加权
    original_weights=[0] * edge_num
    
    for i in motif_edge_set:         
        for j in i:
            temp = df[df.edge==j]
            if temp.empty == False:
                ind = df[df.edge==j].index.tolist()[0]
            else:
            
                u,v = j
                x = (v,u)
                ind = df[df.edge==x].index.tolist()[0]
         
            original_weights[ind] += 1
    # weights = original_weights 
    weights=[]
    for each_weight in original_weights:
        each_weight += 1e-08
        weights.append(each_weight) 
    return weights

#######网络和社团数######
bianhao='31'
net='PGP'
# G = nx.read_edgelist(r"../data/LFRnetwork/LFR1000/network"+bianhao+".txt", create_using=nx.Graph(), nodetype=int, encoding='utf-8''')
# Gi = ig.Graph.Read_Edgelist(r"../data/LFRnetwork/LFR1000/network"+bianhao+".txt", directed=False)
G = nx.read_edgelist(r"../data/realnetwork/"+net+".txt", create_using=nx.Graph(), nodetype=int, encoding='utf-8''')
Gi = ig.Graph.Read_Edgelist(r"../data/realnetwork/"+net+".txt", directed=False)
Gi = Gi.subgraph(map(int, G.nodes()))  # G.nodes()获取一个Graph对象中的node数组,
Gi = Gi.as_undirected()
edge_num = Gi.ecount()
node_num = Gi.vcount()
edges = copy.deepcopy(Gi.get_edgelist())



motif_ins = motif_xuanze('motif_1')


wei = motif_jiaquan(motif_ins)
# nmi_list = []
#####fai-m参数######
# faim_flag = 1
M_N = nx.Graph.number_of_nodes(motif_ins)#几阶模体

for i, (u, v) in enumerate(G.edges):
    G[u][v]['weight'] = wei[i]


communities=community_louvain.best_partition(G,weight='weight')

# communities = nx.community.louvain_communities(G,weight='weight')
# print(communities)



sorted_dict = sorted(communities.items(), key=lambda x: x[0])
solute1 = [value for key, value in sorted_dict]    


# result = {value: index for index, subset in enumerate(communities) for value in subset}
# full_dict = {i: i for i in range(10)}
# full_dict.update(result)
# solute1=list(full_dict.values())
solute1=np.array(solute1)
# print(solute1)
# qlr = cfunc.getQl(Lij,google,solute1,node_num)
qw = ig.GraphBase.modularity(Gi,solute1,directed = False,weights = wei)
# print(solute1)
# qd = ig.GraphBase.modularity(Gi,solute1,directed = True)
##nmi###
# df_news = pd.read_table(r"./LFR325/community_"+bianhao+".txt", header=None)
# df_news = pd.read_table(r"../data/realnetwork/real/"+net+"_groundtruth.txt", header=None)
# df_news = pd.read_table(r"../data/LFR1000/real/network"+bianhao+"_groundtruth.txt", header=None)
# real_membership = []
# for i in df_news[0]:
#     real_membership.append(i)
# real_membership = np.array(real_membership)
# nmi = ig.compare_communities(real_membership,solute1 , method='nmi', remove_none=False)
# ari = ig.compare_communities(real_membership,solute1 , method='adjusted_rand', remove_none=False)
# print('nmi',nmi)
# print(nmi)
all_node_set = set(list(range(node_num)))####所有节点的集合

membershipSet = list(set(solute1))

membership_arr = np.asarray(solute1, dtype=int)
membershipSet_arr = np.asarray(membershipSet, dtype=int)
membershipSet_len = len(membershipSet)


print('#######')


print(net)
print(qw)





