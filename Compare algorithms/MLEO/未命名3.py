# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:19:58 2022

@author: bozite
"""
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

def directed_motif_xuanze(motif):
    M_ins = nx.DiGraph()####模体按照Benson论文中设置

    if motif == "motif_1":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(2,1),(1,3),(3,2)])  # 连通
    elif motif == "motif_2":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(2,1),(1,3),(3,2),(2,3)])  # 连通
    elif motif == "motif_7":
        # motif_7 = nx.DiGraph()
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2, 3), (1, 3),(2, 1)])  # 连通
    elif motif == "motif_6":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2,1),(3, 1), (3, 2)])  # 连通
    elif motif == "motif_5":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3), (1,2),(3,2)])  # 连通
    elif motif == "motif_4":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2,1),(1, 3), (3, 1),(2,3),(3,2)])  # 连通
    elif motif == "motif_3":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2,1),(1, 3), (3, 2),(2,3)])  # 连通
    elif motif == "motif_8":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(3, 1),(3,2)])  # 连通
    elif motif == "motif_9":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(3, 1),(2,3)])  # 连通
    elif motif == "motif_10":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3),(2, 3)])  # 连通
    elif motif == "motif_11":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3),(3,1),(3,2)])  # 连通 
    elif motif == "motif_12":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3),(3,1),(2,3)])  # 连通 
    elif motif == "motif_13":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3),(3,1),(2,3),(3,2)])  # 连通 
    elif motif == "motif_square":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 2), (2, 3), (3, 4),(4, 1)])  # 连通
    elif motif == "motif_bifan":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 3), (1,4),(2, 3), (2, 4)])  # 连通
    return  M_ins

def undirected_motif_xuanze(motif):
    M_ins = nx.Graph()####模体按照Benson论文中设置
    if motif == "motif_1":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2,3),(3, 1)])  # 连通
    elif motif == "motif_2":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3),(3,2)])  # 连通
    return  M_ins

def RM_C(shequ,node_set):
    
    jiediancanyumoti = sum(node_set[shequ], [])#社区内节点参与的模体result_info = sum(info, [])
    # for i in shequ:
    #     jiediancanyumoti+=node_set[i]
    # print(jiediancanyumoti)
    jiediancanyumoti = [*{*[tuple(sorted(i)) for i in jiediancanyumoti]}] #删除重复模体
    shequ = set(shequ)
    fenmu = len(jiediancanyumoti)#分母，即为社区所有节点参与的模体数
    # print(fenmu)
    fenzi = 0#分子 社区内的模体数
    if len(shequ) >= M_N:
        for mt in jiediancanyumoti:
            if set(mt) <= shequ:
                fenzi += 1
    # print(fenzi)
    return fenzi,fenmu

def RM_C_Ai(shequ,Ai,fenzi,fenmu,all_node_set,node_set):
    shequ = set(shequ)
    shequfei = all_node_set-shequ
    Ai_moti = sum(node_set[Ai], [])
    # print(Ai_moti)
    Ai_moti = [*{*[tuple(sorted(i)) for i in Ai_moti]}]
    new_shequ = shequ | set(Ai)
    
    E_in_Ai = 0
    E_out_Ai = 0
    for mt in Ai_moti:
        # print(mt)
        if set(mt) <= new_shequ:
            E_in_Ai += 1
        if set(mt) <= shequfei:
            E_out_Ai += 1    

    return (fenzi+E_in_Ai)/(fenmu+E_out_Ai)**aerfa

def RM_C_v0(shequ,v0,fenzi,fenmu,all_node_set,node_set):
    shequ = set(shequ)
    shequfei = all_node_set-shequ | {v0}
    v0_moti = node_set[v0]
    # print('v0_moti',v0_moti)
    # print(Ai_moti)
    # v0_moti = [*{*[tuple(sorted(i)) for i in v0_moti]}]
    # new_shequ = shequ - {v0}
    # print(new_shequ)
    E_in_v0 = 0
    E_out_v0 = 0
    for mt in v0_moti:
        # print(mt)
        if set(mt) <= shequ:
            E_in_v0 += 1
        if set(mt) <= shequfei:
            E_out_v0 += 1    
    # print('E_in_Ai,E_out_Ai',E_in_v0,E_out_v0)
    return (fenzi-E_in_v0)/(fenmu-E_out_v0)**aerfa

def checkPhase(shequ22,test_rmc_fenzi,test_rmc_fenmu,all_node_set,node_set):
    result = False
    shequ1 = copy.deepcopy(shequ22)
    for vertex in shequ1:
        if len(shequ22) == 1:
            return result
        value1 = RM_C_v0(shequ22,vertex,test_rmc_fenzi, test_rmc_fenmu,all_node_set,node_set)
        R_v = value1 - (test_rmc_fenzi/test_rmc_fenmu**aerfa)
        if R_v > 0:
            shequ22.remove(vertex)
            test_rmc_fenzi, test_rmc_fenmu = RM_C(shequ22,node_set)
            result = True
    return result

def checkPhase2(shequ22,test_rmc_fenzi,test_rmc_fenmu,all_node_set,node_set):
    result = False
    shequ1 = copy.deepcopy(shequ22)
    shequ2 = copy.deepcopy(shequ22)
    for vertex in shequ1:
        # print()
        # print('test_rmc_fenzi, test_rmc_fenmu',test_rmc_fenzi, test_rmc_fenmu)
        # print('vertex',vertex)
        if len(shequ2) == 1:
            return result
        value1 = RM_C_v0(shequ2,vertex,test_rmc_fenzi, test_rmc_fenmu,all_node_set,node_set)
        # print('value1',value1)
        R_v = value1 - (test_rmc_fenzi/test_rmc_fenmu**aerfa)
        # print('R_v',R_v)
        if R_v > 0:
            shequ2.remove(vertex)
            test_rmc_fenzi, test_rmc_fenmu = RM_C(shequ2,node_set)
            result = True
    return result

def shequlinju(shequ,all_nei):
    c_nei = []
    for i in shequ:
        c_nei += all_nei[i]
    c_nei = list(set(c_nei))
    temp1 = copy.deepcopy(c_nei)
    
    for j in temp1:
        if j in shequ:
            c_nei.remove(j)
    return c_nei

G = nx.read_edgelist(r'MLEO_testgraph2.txt', create_using=nx.Graph(), nodetype=int, encoding='utf-8')
Gi = ig.Graph.Read_Edgelist(r'MLEO_testgraph2.txt', directed=False)
Gi = Gi.subgraph(map(int, G.nodes()))  # G.nodes()获取一个Graph对象中的node数组,
Gi = Gi.as_undirected()
edge_num = Gi.ecount()
node_num = Gi.vcount()
edges = Gi.get_edgelist()

out_nei = []
for i in range(node_num):
    out_nei.append(list(Gi.successors(i)))
in_nei = []
for i in range(node_num):
    in_nei.append(list(Gi.predecessors(i)))
all_nei = []
for i in range(node_num):
    all_nei.append(list(Gi.neighbors(i)))
all_node_set = set(list(range(node_num)))####所有节点的集合
# n = nx.DiGraph.number_of_nodes(G)
motif_ins = undirected_motif_xuanze('motif_1')
total_motif_nums = fm.total_motif_num(G, motif_ins, directed=False, weighted=False)
total_motif_lists_1 = fm.total_motif_list(G, motif_ins, directed=False, weighted=False)
total_motif_lists = total_motif_lists_1[0]######网络中所有模体
total_motif_lists = np.array(total_motif_lists)


node_motif_num_list=[]#####点模体度列表#####
for i in range(node_num):
    node_motif_num_list.append(fm.node_in_motif(G,motif_ins,i,directed=False,weighted=False))
node_motif_num_list = np.array(node_motif_num_list)

node_set= []#####每个节点参与的模体#####
for node in range(node_num):
    Node_set,edge_set=fm.node_in_motif_list(G, motif_ins, node, directed=False, weighted=False)
    node_set.append(Node_set)
node_set = np.array(node_set,dtype = object)
#核心种子社区获取
yuanshi_node = np.argmax(node_motif_num_list)
seed = set(np.array(node_set[yuanshi_node]).reshape(1,-1).tolist()[0])

#扩展
M_N = nx.Graph.number_of_nodes(motif_ins)#几阶模体

shequ = [0,7,6,8]

test_rmc_fenzi, test_rmc_fenmu= RM_C(shequ,node_set)
print(test_rmc_fenzi, test_rmc_fenmu)

# value = RM_C_Ai(shequ,[2],test_rmc_fenzi, test_rmc_fenmu,all_node_set,node_set)
# xxx
# Ai = [10,14]
# value = RM_C_Ai(shequ,Ai,test_rmc_fenzi, test_rmc_fenmu,all_node_set,node_set)
# print(value)
# xxx

# RM_C(shequ,node_set)
# shequ2 = [2,4,5]
# vertex = 4
# value1 = RM_C_v0(shequ2,vertex,test_rmc_fenzi, test_rmc_fenmu,all_node_set,node_set)
# RM_C([2,4,5,6],node_set)

ii = 1
jj = 0
aerfa = 1.1
#社区的邻居节点
c_nei = shequlinju(shequ,all_nei)
# xxx
# 社区扩张
belta = 1
# print(shequ)
flag = checkPhase2(shequ,test_rmc_fenzi,test_rmc_fenmu,all_node_set,node_set)
# print(shequ)
print(flag)
# xxx
# print(shequ)
# flag = checkPhase(shequ,test_rmc_fenzi,test_rmc_fenmu,all_node_set,node_set)

# xxx
# xxx
while checkPhase(shequ,test_rmc_fenzi,test_rmc_fenmu,all_node_set,node_set):
    # print(checkPhase(shequ,test_rmc_fenzi,test_rmc_fenmu,all_node_set,node_set))
    while ii < M_N and len(c_nei)!= 0:
        # print('shequ',shequ)
        # print('c_nei',c_nei)
        R_list = []
        Ai_list = []
        for vr in c_nei:
            # print('vr',vr)
            if ii == 1: 
                Ai = [vr]
            else:
                temp = copy.deepcopy(all_nei[vr])
                
                for t in all_nei[vr]:
                    if t in shequ:
                        temp.remove(t)
                # print('temp',temp)
                # xxx
                if len(temp) > ii:
                    temp1 = set(np.random.choice(temp,ii,replace=False))
                else:
                    temp1 = set(temp)
                Ai = list({vr}|temp1)
            value = RM_C_Ai(shequ,Ai,test_rmc_fenzi, test_rmc_fenmu,all_node_set,node_set)
            R = value - (test_rmc_fenzi/test_rmc_fenmu**aerfa)
            R_list.append(R)
            Ai_list.append(list(Ai))
            # print('shequ',shequ)
            # print(R)
        Rx = np.max(R_list)
        # print(R_list)
        Rx_index = np.argmax(R_list)
        
        Max_A = Ai_list[Rx_index]

        if Rx<=0:
            ii += 1
        else:
            ii = 1
            jj += 1
            shequ.extend(Max_A)
            shequ = list(set(shequ))
            test_rmc_fenzi, test_rmc_fenmu = RM_C(shequ,node_set)
            c_nei = shequlinju(shequ,all_nei)
            if jj % belta == 0:                # x
                flag = checkPhase(shequ,test_rmc_fenzi,test_rmc_fenmu,all_node_set,node_set)

# shequ = list(range(node_num))
# c_nei = shequlinju(shequ,all_nei)




# # shequ = set([1,2,3])

# # new_shequ = shequ - {2}
# shequ = [2,5]
# Ai = [4]
# test_rmc_fenzi, test_rmc_fenmu= RM_C(shequ,node_set)
# RM_C_Ai(shequ,Ai,test_rmc_fenzi, test_rmc_fenmu,all_node_set,node_set)
