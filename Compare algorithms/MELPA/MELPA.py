# -*- coding: utf-8 -*-
"""
@author: wyw
"""
import igraph as ig
import networkx as nx
import pandas as pd
import time
import numpy as np
from numpy import log
import random as rd
import motif_network_construct as net_stru_func
from tqdm import tqdm
  
def p_uv(u,v,me_adj):
    ku = me_adj[u].sum()
    kv = me_adj[v].sum()
    return log((kv+1)) / (log((ku+1))+log((kv+1)))

def svot_u_label_vlist(u,v_label_count,label_vlist,me_adj,aerfa):
    tmp_value = 0
    for v in label_vlist:
        wuv = me_adj[u,v]
        puv = p_uv(u,v,me_adj)
        tmp_value = tmp_value + wuv*puv
    return (aerfa*v_label_count) + ((1-aerfa)*tmp_value)

network_name = "test12"
Mlist = {1:"M1",2:"M2",3:"M3",4:"M4",5:"M5",6:"M6",7:"M7",8:"M8"} #模体选择列表
M_flag = Mlist[1]
nmi_flag = 1 #0:关，1：开
aerfa_list = [i/10 for i in range(11)]
# 获取网络数据中的边列表，并根据其使用igraph创建网络
G1 = nx.read_edgelist("data/realnetwork/"+ network_name + ".txt", create_using=nx.Graph())
G1 = G1.to_undirected()
Gi = ig.Graph.Read_Edgelist("data/realnetwork/"+ network_name +".txt")
Gi = Gi.subgraph(map(int,G1.nodes()))          
Gi = Gi.as_undirected()
edge_all = Gi.get_edgelist()
node_list = list(G1.nodes)
n=G1.number_of_nodes()
# st1 = time.process_time()
G2 = nx.Graph() 
G2.add_nodes_from([i for i in range(n)])
G2.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G2)
adj=adj.todense() 

G,ij_participate_motif_number_list = net_stru_func.construct_weighted_network(G2,edge_all,n,M_flag) #构建出基于M_flag模体加权的网络
# 构建基于模体的加权网络邻接矩阵motif_adj
motif_adj = nx.adjacency_matrix(G)
motif_adj = motif_adj.todense() 
me_adj = motif_adj + adj #总权重矩阵=模体权重+边权重

if nmi_flag==1:
    #真实社区划分
    fo = pd.read_table(r"data/realnetwork/real/" + network_name+"_groundtruth.txt",header = None)
    real=[]
    for i in fo[0]:
        real.append(i) 

aerfa_Qw = [0]*len(aerfa_list)
aerfa_NMI = [0]*len(aerfa_list)
runs = 10 #独立运行次数
run,aerfa_index,run_flag = 1,0,0
#初始化数据
Qws,nmis = [],[]
while run <= runs:
    aerfa = aerfa_list[aerfa_index] # aerfa参数
    #节点对应标签
    node_label = [i for i in range(n)]
    tmax = 200
    # =============================================================================
    # 标签传播算法 MELPA
    # =============================================================================
    for gen in tqdm(range(tmax)):
        use_node_index = np.random.permutation(np.arange(n))
        for u in use_node_index:
            # 获得节点 i 的所有邻居节点 j_nodes
            vlist = np.nonzero(me_adj[u,:])[1] # u的邻居节点集合vlist
            vlist_label_set = set([node_label[v] for v in vlist]) #u的邻居节点的标签集合
            # vlist_label_set.add(node_label[u])
            vlist_label_svot = []
            for label in vlist_label_set:
                label_vlist = [vi for vi in vlist if node_label[vi]==label] #vlist中具有与label相同标签的节点集合
                label_count = len(label_vlist) #u邻域中label标签的数量
                label_svot = svot_u_label_vlist(u,label_count,label_vlist,me_adj,aerfa)
                vlist_label_svot.append(label_svot)
    
            u_maximal_score = max(vlist_label_svot) #最高分数
            maximal_score_labels = []   #获得最高分数的标签集合
            for index, v_label in enumerate(vlist_label_set):
                if vlist_label_svot[index] == u_maximal_score:
                    maximal_score_labels.append(v_label)
            if len(maximal_score_labels) == 1:
                node_label[u] = maximal_score_labels[0]    #更新节点u的标签
            else:
                # print("\nmaximal_score_labels=",maximal_score_labels)
                node_label[u] = rd.choice(maximal_score_labels) #更新节点u的标签
    # =============================================================================
    # 计算结果
    # =============================================================================
    best_x_Qw=ig.GraphBase.modularity(Gi,node_label,weights=ij_participate_motif_number_list) #计算QW
    if nmi_flag==1:
        best_fit_x_nmi = ig.compare_communities(real,node_label,method='nmi',remove_none=False)  #计算NMI
        aerfa_NMI[aerfa_index] = best_fit_x_nmi
    # print("aerfa_index=",aerfa_index)
    if run_flag ==0: 
        aerfa_index+=1
        aerfa_Qw[aerfa_index] = best_x_Qw
    if run_flag==0 and aerfa_index == len(aerfa_list)-1:
        run_flag, aerfa_index = 1, np.argmax(aerfa_Qw)
    if run_flag == 2: 
        run +=1
        Qws.append(best_x_Qw)
        if nmi_flag==1:
            nmis.append(best_fit_x_nmi)
    elif run_flag==1:
        Qws.append(aerfa_Qw[aerfa_index])  
        nmis.append(aerfa_NMI[aerfa_index])  
        run,run_flag = 1,2
        
# =============================================================================
# 输出结果 
# =============================================================================
print("\n## MLEPA #### {} ### {} ####".format(network_name,M_flag))        
print('QW_mean={},std={}, max={}'.format(round(np.mean(Qws),3), np.std(Qws), max(Qws)))
if nmi_flag==1:
    best_fit_x_nmi = ig.compare_communities(real,node_label,method='nmi',remove_none=False)  #计算NMI
    print('NMI_mean={},std={}, max={}'.format(round(np.mean(nmis),3), np.std(nmis), max(nmis)))
print("max_aerfa=", aerfa_list[aerfa_index])
print("\nmembership_c=",node_label)



