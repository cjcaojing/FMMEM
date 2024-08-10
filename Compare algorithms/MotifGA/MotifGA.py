# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:44:33 2022

@author: bozite
"""

import networkx as nx
import find_motifs as fm
import pandas as pd
import igraph as ig
import numpy as np
import random  
from tqdm import tqdm
import time
import geatpy as ea
from time import process_time
import copy
import community as community_louvain
import fai_m as fai_m_func
# import cython_function as cfunc
# from operator import add
# from functools import reduce
import motif_network_construct as net_stru_func
from igraph.clustering import VertexClustering

def fai_m(Gi, partition, node_motif_num_list, all_node_set, total_motif_lists,n, M_N):####计算模体电导，但是是单个社区的模体电导
    # partition = list(filter(None,VertexClustering(Gi,partition)))
    partition_len = len(partition)
    # print("partition=",partition)
    # list4 = []
    if partition_len != n and partition_len != 1:###所有节点一个社区 和 每个节点一个社区的情况为1.0
        list1 = []######计算单个社区内点模体
        for cur_C_node in partition:
            jishu = 0
            for node in cur_C_node:
                jishu += node_motif_num_list[node]
            list1.append(jishu)
        
        # list1 = np.array(list1)
        temp_sum = sum(list1)######更小的单个社区  与  其余节点的点模体作为分母
        for i in range(len(list1)):
            temp = list1[i]
            # print("temp=",temp)
            list1[i] = min(temp,temp_sum-temp)

        
        list2 = []#########计算社区与其余节点  间 的模体
        app = list2.append
        for indde,cur_C_node in enumerate(partition):
            l1 = list1[indde]
            if l1 != 0 and len(cur_C_node) >= M_N:
                cur_C_node = set(cur_C_node)
                # print('cur_C_node',cur_C_node)
                S_fei = all_node_set-cur_C_node
                # print('S_fei',S_fei)
                sqnei_s = 0
                sqnei_sfei = 0
                for mt_index, mt in enumerate(total_motif_lists):
                    if set(mt) <= cur_C_node:#模体是否存在于社区中,既mt是否是cur_C_node的子集
                        sqnei_s += 1
                    if set(mt) <= S_fei:
                        sqnei_sfei += 1
                shjian_mt = len(total_motif_lists)-sqnei_sfei-sqnei_s#所有模体数减去 社区内的模体 为社区间模体
                if shjian_mt != len(total_motif_lists):
                    app(shjian_mt/l1)

            else:
                app(1.0)
        # print(list2)
        return min(list2)#sum(list2)/len(list2)#
    else:
        return 1.0

def fai_m2(Gi, partition, node_motif_num_list, all_node_set, total_motif_lists,n,M_N):####计算所有社区的模体电导并进行平均
    # partition = list(filter(None,VertexClustering(Gi,membership)))#将标签型改为社区型
    if len(partition) != n and len(partition) != 1:###所有节点一个社区 和 每个节点一个社区的情况为1.0
        list1 = []######计算单个社区内节点的点模体度
        app1 = list1.append
        for cur_C_node in partition:
            jishu = 0
            for node in cur_C_node:
                jishu += node_motif_num_list[node]
            app1(jishu)
        
        # list1 = np.array(list1)
        temp_sum = sum(list1)######更小的  单个社区内点模体度  与  其余节点的点模体度作为分母
        for i in range(len(list1)):
            temp = list1[i]
            list1[i] = min(temp,temp_sum-temp)
        
        list2 = []#########计算社区与其余节点  间 的模体  并除以对应社区的点模体度
        app2 = list2.append
        for indde,cur_C_node in enumerate(partition):#indde是索引，cur_C_node是社区内节点
            l1 = list1[indde]
            if l1 != 0:
                if len(cur_C_node) >= M_N:#先计算社区内的模体数
                    cur_C_node = set(cur_C_node)#单个社区节点
                    # print('cur_C_node',cur_C_node)
                    S_fei = all_node_set-cur_C_node#除社区外的其余节点
                    # print('S_fei',S_fei)
                    sqnei_s = 0
                    sqnei_sfei = 0
                    for mt_index, mt in enumerate(total_motif_lists):
                        if set(mt) <= cur_C_node:#模体是否存在于社区中,既mt是否是cur_C_node的子集
                            sqnei_s += 1
                            # print(mt_index)
                        if set(mt) <= S_fei:
                            sqnei_sfei += 1
                    
                    shjian_mt = len(total_motif_lists)-sqnei_sfei-sqnei_s#所有模体数减去 社区内的模体 为社区间模体
                    if shjian_mt != len(total_motif_lists):
                        app2(shjian_mt/l1)
                    else:
                        app2(1.0)
                else:
                    app2(1.0)
            else:
                app2(1.0)
        # print('list1',len(list1))
        # print('list2',list2)
        return sum(list2)/len(list2)#min(list3)#
    else:
        return 1.0

def edge_mutu(ofsp,pm,n,all_nei):
    # Xa,Xb = ofsp
    X_rand = np.random.rand(NP, n)
    for ro in range(NP):
        for i in range(n):
            if X_rand[ro][i] < pm:
                ofsp[ro][i] = ofsp[ro][np.random.choice(all_nei[i], 1, replace=False)]

    return ofsp  

def merge_comm(Gi,yyyy,n):
    bol = False
    while not bol:
        member_0 = [-1]*n
        for cno in range(len(yyyy)):
            for j in yyyy[cno]:
                member_0[j]=cno
        a = VertexClustering(Gi,member_0)
        b = a.cluster_graph(combine_edges = False)
        Cm = b.get_adjacency().data
        listdui = copy.deepcopy(np.diagonal(Cm))
        lists = []
        listzhi = []
        for c_ind in range(len(Cm)):
            Cm[c_ind][c_ind] = 0
            s = np.argmax(Cm[c_ind])
            zhi = np.max(Cm[c_ind])
            lists.append(s)
            listzhi.append(zhi)
        bol = np.all((np.array(listdui) - np.array(listzhi)) > 0)
        if bol == False:
            uuu = np.array(listdui) - np.array(listzhi)
            # print(uuu)
            ke = np.argmin(uuu)
            # print("ke=",ke)
            # print("lists_ke=",lists[ke])
            # print("yyy_len=",len(yyyy))
            yyyy[ke] += yyyy[lists[ke]]
            yyyy.pop(lists[ke])
    return yyyy


network_name = "football"
nmi_flag = 0 #0:关，1：开
faim_flag = 3
M ="M1"
runs = 1 #独立运行次数

time_start_1 = time.process_time() #时间蹉1
G = nx.read_edgelist("data/realnetwork/"+ network_name + ".txt", create_using=nx.Graph())
G = G.to_undirected()
Gi = ig.Graph.Read_Edgelist("data/realnetwork/"+ network_name + ".txt")
Gi = Gi.subgraph(map(int,G.nodes()))          
Gi = Gi.as_undirected()
edge_all = Gi.get_edgelist()
node_list = list(G.nodes)
n = G.number_of_nodes()

all_nei = []
for i in range(n):
    all_nei.append(list(Gi.neighbors(i)))

# 获得无权网络邻接矩阵

G1 = nx.Graph() 
G1.add_nodes_from([i for i in range(n)])
G1.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G1)
adj=adj.todense()
GW,g,ij_participate_motif_number_list = net_stru_func.construct_weighted_network(G1,edge_all,node_list,M) #构建出基于M_flag模体加权的网络
motif_adj = net_stru_func.get_motifadd_adj(G1, edge_all, M)

total_motif_nums = fm.total_motif_num(GW, g, directed=False, weighted=False)
total_motif_lists_1 = fm.total_motif_list(GW, g, directed=False, weighted=False)
total_motif_lists = total_motif_lists_1[0]

if nmi_flag==1:
    # 获得真实社区划分
    fo = pd.read_table(r"data/realnetwork/real/" + network_name + "_groundtruth.txt",header = None)
    real_membership =[]
    for i in fo[0]:
        real_membership.append(i) 

node_set= []#####每个节点参与的模体#####
for node in range(n):
    Node_set,edge_set=fm.node_in_motif_list(GW, g, node, directed=False, weighted=False)
    node_set.append(Node_set)

node_motif_num_list=[]#####点模体度集合#####
for i in range(n):
    node_motif_num_list.append(fm.node_in_motif(GW,g,i,directed=False,weighted=False))
node_motif_num_list = np.array(node_motif_num_list)

time_end_1 = time.process_time() #时间蹉1
time_spend_1 = time_end_1 - time_start_1

#####fai-m参数######
M_N = nx.DiGraph.number_of_nodes(g)#几阶模体
node_motif_num_arr = np.asarray(node_motif_num_list, dtype=int)
total_motif_len = len(total_motif_lists)
total_motif_arr = np.zeros((total_motif_len, M_N), dtype = int, order = 'C')
for imt_index, imt in enumerate(total_motif_lists):
    total_motif_arr[imt_index,:] = np.asarray(imt, dtype=int)
Qws,nmis,Cs_lens,faimeans = [],[],[],[]
for run in range(runs):
    ##############
    NP = 100 # The number of candidate solutions
    Gen = 3000  # The number of iterations
    
    all_node_set = set(list(range(n)))
    # 网络参数设置
    # Am = np.array(Gi.get_adjacency().data)
    # suijilist1 = list(range(NP))
    #编码+初始化
    pop = np.random.randint(n, size=(NP, n))
    for ro in range(NP):
        for i in range(n):
            pop[ro][i] = pop[ro][np.random.choice(all_nei[i], 1, replace=False)]
    
    time_start_fit1 = time.process_time()
    fit = []
    for i in range(NP):
        membershipSet = list(set(pop[i]))
        membership_arr = np.asarray(pop[i], dtype=int)
        membershipSet_arr = np.asarray(membershipSet, dtype=int)
        membershipSet_len = len(membershipSet)
        fai_m1_value2 = fai_m_func.fai_m(faim_flag, membership_arr, membershipSet_arr,membershipSet_len, node_motif_num_arr, 
                              total_motif_arr, total_motif_len,n, M_N)
        fit.append(fai_m1_value2)
    time_end_fit1 = time.process_time()
    fit_spend_time1 = time_end_fit1 - time_start_fit1
    
    fit = np.array(fit).reshape(NP,1)
    # print(fit)
    ##参数##
    selectStyle = "etour" # 精英锦标赛etour 轮盘赌rws 锦标赛tour 
    REC_F = 'xovud' #均匀分布交叉
    pm = 0.1
    pc = 0.8
    #######
    
    best_fit_history = []
    best_x_history = []
    
    FitnV = fit.copy()
    # offspring = pop[ea.selecting(selectStyle, FitnV, 2)]
    # print(offspring)
    # xxxx
    print("\n## {}网络循环开始第{}次实验 #####".format(network_name,run))
    fit_spend_times = []
    ga_spend_times = []
    for gen in tqdm(range(Gen)):
        time_start_ga1 = time.process_time()
        cross_off = ea.recombin(REC_F, pop, RecOpt = pc) #重组
        mutu_off = edge_mutu(cross_off,pm,n,all_nei) #变异
        time_end_ga1 = time.process_time() #GA时间计算
        mutu_fit = []
        time_start_fit = time.process_time()
        for ind in range(NP):
            membershipSet = list(set(mutu_off[ind]))
            membership_arr = np.asarray(mutu_off[ind], dtype=int)
            membershipSet_arr = np.asarray(membershipSet, dtype=int)
            membershipSet_len = len(membershipSet)
            fai_m1_value2 = fai_m_func.fai_m(faim_flag, membership_arr, membershipSet_arr,membershipSet_len, node_motif_num_arr, 
                                  total_motif_arr, total_motif_len,n, M_N)
            mutu_fit.append(fai_m1_value2)
        time_end_fit = time.process_time()
        fit_spend_time = time_end_fit - time_start_fit 
        fit_spend_times.append(fit_spend_time)
        time_start_ga2 = time.process_time()
        # 优秀个体选择
        for i in range(NP):
            if mutu_fit[i] <= FitnV[i]:
                pop[i] = mutu_off[i]
                # pop_decode[i] = mutu_pop_decode[i]
                FitnV[i] = mutu_fit[i]
                 
        best_fit = np.min(FitnV)
        best_fit_history.append(best_fit)
        # 当代种群最优个体
        best_x = pop[np.argmin(FitnV)]
        best_x_history.append(best_x)
        time_end_ga2 = time.process_time()
        ga_spend_times.append(time_end_ga2-time_start_ga2 + time_end_ga1 - time_start_ga1)
        if gen>=100 and (((abs(abs(best_fit_history[gen-100]) - best_fit) <= 1.0e-5) 
                          or (best_fit_history[gen-50] == best_fit))):
            print("最后迭代次数为",gen-50)
            break #跳出本次循环
        # if gen%100==0:
        #     print("\nfaim_sum=", best_fit)
        #     Qwd = ig.GraphBase.modularity(Gi, best_x ,directed = False, weights = ij_participate_motif_number_list)
        #     print("Qwd",Qwd)
        #     nmi = ig.compare_communities(real_membership, best_x, method='nmi', remove_none=False)
        #     print('nmi', nmi)
    # print("\nCs_len=", len(set(best_x)))
    ################################
    time_end_2 = time.process_time()
    Qwd = ig.GraphBase.modularity(Gi, best_x ,directed = False, weights = ij_participate_motif_number_list)
    print("GA_QW=",Qwd)
    
    time_start_3 = time.process_time()
    yyyy = list(filter(None,VertexClustering(Gi,copy.deepcopy(best_x))))
    best_x_1 = merge_comm(Gi,yyyy,n)
    # print()
    # 社区合并结果
    com_list_symbol=[0]*n
    for i in range(len(best_x_1)):
        for j in best_x_1[i]:
            com_list_symbol[j]=i
    com_list_symbol = np.array(com_list_symbol)
    membershipSet = list(set(com_list_symbol))
    membership_arr = np.asarray(com_list_symbol, dtype=int)
    membershipSet_arr = np.asarray(membershipSet, dtype=int)
    membershipSet_len = len(membershipSet)
    fai_m1_value2 
    time_end_3 = time.process_time()
    time_spend_3 = time_end_3 - time_start_3
    print("#######时间统计结果#######")
    print("init_time_spend={}s".format(time_spend_1))
    print("ga_spend_times={}s".format(np.sum(ga_spend_times)))
    print("fit_spend_time={}s".format(fit_spend_time1 + np.sum(fit_spend_times)))
    print("merge_time_spend={}s".format(time_spend_3))
    print("time_spend_sum={}s,".format(time_end_3 - time_end_1))
    Qwd = ig.GraphBase.modularity(Gi, list(com_list_symbol) ,directed = False, weights = ij_participate_motif_number_list)
    Qws.append(Qwd)
    print("Qwd",Qwd)
    
    if nmi_flag == 1:
        nmi = ig.compare_communities(real_membership, list(com_list_symbol), method='nmi', remove_none=False)
        nmis.append(nmi)
        # print('nmi', nmi)
    print("best_x=", list(com_list_symbol))
    Cs_lens.append(len(set(best_x)))
    # faisum = fai_m_func.fai_m(faim_flag, membership_arr, membershipSet_arr,membershipSet_len, node_motif_num_arr, 
    #                       total_motif_arr, total_motif_len,n, M_N)
    # faimeans.append(faisum)
    # print("faisum",faisum)

# =============================================================================
# 统计结果    
# =============================================================================
# min_value,max_value = min(Qws),max(Qws)
# min_index,max_index = Qws.index(min_value),Qws.index(max_value)
# Qws.remove(min_value)
# Qws.remove(max_value)
# nmis.pop(min_index)
# nmis.pop(max_index)
print("###### {} #######".format(network_name))       
print('QW_mean={},std={}, max={}'.format(np.mean(Qws), np.std(Qws), max(Qws)))
if nmi_flag == 1:
    print('NMI_mean={},std={}, max={}'.format(np.mean(nmis), np.std(nmis), max(nmis)))
print('Cs_len_mean={},faimeans_mean={}'.format(np.mean(Cs_lens), np.mean(faimeans)))
