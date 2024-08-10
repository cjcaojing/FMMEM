# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022

@author: WYW
"""
"""
    FFM_CD_main_v1_NMM
   使用各种优化算法，基于模体的加权网络的社区检测
"""

import numpy as np
import igraph as ig
import networkx as nx
import os
import time
import copy
from tqdm import tqdm

# 各模块函数
import motif_network_construct as net_stru_func
import algorithm_FCD_function as alg_func
import motif_FFM_CD_functionW as func

# 引入外部函数
import find_motifs as fm
import xlwt

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:05:52 2020

@author: Administrator
"""
import xlwt
#import xlsxwriter
 
import numpy as np
import igraph as ig
import random  
from numpy import random
import networkx as nx
import copy
import pandas as pd
import itertools
from pandas import DataFrame
from numpy import mean,std,median
import xlsxwriter
def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item] 
#计算模体结构3-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def three_one_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        return len(set(u_friends) & set(v_friends))   
    
def three_one_morphology(G,edge_all):
    ij_participate_motif_number_list=[]    
    for u,v in edge_all:    
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        u_friends=list(u_friends)
        v_friends=list(v_friends)
        if (u_friends == []) or (v_friends == []):
            ij_participate_motif_number_list.append(0)
        else:
            ij_participate_motif_number_list.append(len(set(u_friends) & set(v_friends)))
    return  ij_participate_motif_number_list

#计算模体结构3-2(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/2为模体数量
def three_two_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    u_friends=list(u_friends)
    v_friends=list(v_friends)
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    num = len(u_mor) + len(v_mor)
    return num
#计算模体结构4-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def four_one_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    v_friends = list(v_friends)
    u_friends = list(u_friends)
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    u_list = []
    v_list = []
    if len(u_mor) <= 1:
        deta1 = 0
    else:
        for i in itertools.combinations(u_mor,2):
            u_list.append(i)
        deta1 = int(len(u_list))
        for p,q in u_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta1 -= 1
            else:
                deta1 += 0
    if len(v_mor) <= 1:
        deta2 = 0
    else:
        for j in itertools.combinations(v_mor,2):
            v_list.append(j)
        deta2 = int(len(v_list))
        for p,q in v_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta2 -= 1
            else:
                deta2 += 0
    return deta1+deta2
#计算模体结构4-2（连边参与模体构造数量）,sum(ij_participate_motif_number_list)/3为模体数量
def four_two_morphology(G,edge_all):
    #求列表的长度
    n=len(edge_all)
    #生成一个由n个0组成的列表
    ij_participate_motif_number_list=[0 for x in range(n)]
    for u,v in edge_all: 
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if u in v_friends:
            v_friends.remove(u)
        if v in u_friends:
            u_friends.remove(v)
        u_mor = list(set(u_friends)-set(v_friends))
        v_mor = list(set(v_friends)-set(u_friends))
        mor_list0 = []
        if (u_mor == []) or (v_mor == []):
            index0=edge_all.index((u,v))
            ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+0
        else:
            for i in u_mor:
                for j in v_mor:
                    mor_list0.append((i,j))
            deta = int(len(mor_list0))
            mor_list=copy.deepcopy(mor_list0)
            for p,q in mor_list0:
                if (p,q) in edge_all or (q,p) in edge_all:
                    mor_list.remove((p,q))
                    deta -= 1
            index0=edge_all.index((u,v))
            ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+deta
            #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
            node_list=[]
            node_number_list=[]
            for i in range(len(mor_list)):
                for j in mor_list[i]:
                   node_list.append(j) 
            set_node_list=set(node_list)
            for i in set_node_list:
                node_number_list.append([i,node_list.count(i)])
            #判断每个节点与u、v哪个节点相连接，并更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
            for i in range(len(node_number_list)):
                if (node_number_list[i][0],u) in edge_all:
                    index1=edge_all.index((node_number_list[i][0],u))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                elif (u,node_number_list[i][0]) in edge_all:
                    index1=edge_all.index((u,node_number_list[i][0]))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]      
                elif (node_number_list[i][0],v) in edge_all:
                    index1=edge_all.index((node_number_list[i][0],v))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                elif (v,node_number_list[i][0]) in edge_all:
                    index1=edge_all.index((v,node_number_list[i][0]))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
    return ij_participate_motif_number_list
#计算模体结构4-3(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def four_three_morphology(G,edge_all):
    #求列表的长度
    n=len(edge_all)
    #生成一个由n个0组成的列表
    ij_participate_motif_number_list=[0 for x in range(n)]
    for u,v in edge_all:
        index_uv=edge_all.index((u,v))
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if u in v_friends:
            v_friends.remove(u)
        if v in u_friends:
            u_friends.remove(v)
        u_mor = list(set(u_friends)-set(v_friends))
        v_mor = list(set(v_friends)-set(u_friends))
        u_list0 = []
        v_list0 = []
        if len(u_mor) <= 1:
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            for i in itertools.combinations(u_mor,2):
                u_list0.append(i)
            deta1 = 0
            u_list=[]
            for p,q in u_list0:
                if (p,q) in edge_all:
                    deta1 += 1
                    u_list.append((p,q))
                    index0=edge_all.index((p,q))
                    ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
                elif (q,p) in edge_all:
                    deta1 += 1
                    u_list.append((p,q))
                    index0=edge_all.index((q,p))
                    ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
                else:
                    deta1 += 0
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta1
            #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
            node_list=[]
            node_number_list=[]
            for i in range(len(u_list)):
                for j in u_list[i]:
                   node_list.append(j) 
            set_node_list=set(node_list)
            for i in set_node_list:
                node_number_list.append([i,node_list.count(i)])
            #更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
            for i in range(len(node_number_list)):
                if (node_number_list[i][0],u) in edge_all:
                    index1=edge_all.index((node_number_list[i][0],u))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                elif (u,node_number_list[i][0]) in edge_all:
                    index1=edge_all.index((u,node_number_list[i][0]))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]   
        if len(v_mor) <= 1:
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            for i in itertools.combinations(v_mor,2):
                v_list0.append(i)
            deta2 = 0
            v_list=[]
            for p,q in v_list0:
                if (p,q) in edge_all:
                    deta2 += 1
                    v_list.append((p,q))
                    index0=edge_all.index((p,q))
                    ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
                elif (q,p) in edge_all:
                    deta2 += 1
                    v_list.append((p,q))
                    index0=edge_all.index((q,p))
                    ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
                else:
                    deta2 += 0
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta2
            #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
            node_list=[]
            node_number_list=[]
            for i in range(len(v_list)):
                for j in v_list[i]:
                   node_list.append(j) 
            set_node_list=set(node_list)
            for i in set_node_list:
                node_number_list.append([i,node_list.count(i)])
            #更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
            for i in range(len(node_number_list)):
                if (node_number_list[i][0],v) in edge_all:
                    index1=edge_all.index((node_number_list[i][0],v))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                elif (v,node_number_list[i][0]) in edge_all:
                    index1=edge_all.index((v,node_number_list[i][0]))
                    ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]      
    return ij_participate_motif_number_list
#计算模体结构4-4(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/5为模体数量
def four_four_morphology(G,edge_all):
    #求列表的长度
    n=len(edge_all)
    #生成一个由n个0组成的列表
    ij_participate_motif_number_list=[0 for x in range(n)]
    for u,v in edge_all: 
        index_uv=edge_all.index((u,v))
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
            else:
                cn_edge = []
                for i in itertools.combinations(cn,2):
                    cn_edge.append(i)
                d1 = 0
                #不相互连接的连边集合
                cn_edge0=copy.deepcopy(cn_edge)
                for p,q in cn_edge:
                   if (p,q) in edge_all or (q,p) in edge_all:
                       d1 += 1
                       cn_edge0.remove((p,q))
                   else:
                       d1 += 0
                deta = int(len(cn_edge)) - d1
                ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta
                #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
                node_list=[]
                node_number_list=[]
                for i in range(len(cn_edge0)):
                    for j in cn_edge0[i]:
                       node_list.append(j) 
                set_node_list=set(node_list)
                for i in set_node_list:
                    node_number_list.append([i,node_list.count(i)])
                #判断每个节点与u、v哪个节点相连接，并更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
                for i in range(len(node_number_list)):
                    if (node_number_list[i][0],u) in edge_all:
                        index1=edge_all.index((node_number_list[i][0],u))
                        ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                    else :
                        index1=edge_all.index((u,node_number_list[i][0]))
                        ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]      
                    if (node_number_list[i][0],v) in edge_all:
                        index1=edge_all.index((node_number_list[i][0],v))
                        ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
                    else :
                        index1=edge_all.index((v,node_number_list[i][0]))
                        ij_participate_motif_number_list[index1]=ij_participate_motif_number_list[index1]+node_number_list[i][1]
    return ij_participate_motif_number_list
#计算模体结构4-5(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def four_five_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    v_friends = list(v_friends)
    u_friends = list(u_friends)
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    mor_list = []
    if (u_mor == []) or (v_mor == []):
        return 0
    else:
        for i in u_mor:
            for j in v_mor:
                mor_list.append((i,j))
        deta = 0
        for p,q in mor_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta += 1
        return deta
#计算模体结构4-6(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/6为模体数量
def four_six_morphology(u,v,G):
    u_friends = G.neighbors(u)
    v_friends = G.neighbors(v)
    v_friends = list(v_friends)
    u_friends = list(u_friends)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = list(set(u_friends) & set(v_friends))
        if len(cn) <= 1:
            return 0
        else:
            cn_edge = []
            for i in itertools.combinations(cn,2):
                cn_edge.append(i)
            deta = 0
            for p,q in cn_edge:
               if (p,q) in edge_all or (q,p) in edge_all:
                    deta += 1
               else:
                   deta += 0
            return deta
def moti_num(all_edge_G):    
    data=DataFrame(list(all_edge_G),columns=['src','dst'])
    GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    
    #计算三阶模体 和M4(a,e,f)和S1和SS5
#    ij_participate_motif_number_list=[]
#    for i,j in all_edge_G:
#        ij_participate_motif_number=three_one_morphology(i,j,GG)        
#        ij_participate_motif_number_list.append(ij_participate_motif_number)
    
    #计算4-2,4-3,4-4 S2-S14 SS1-SS4模体 函数中遍历
    # ij_participate_motif_number_list=four_two_morphology(GG,all_edge_G) 
    ij_participate_motif_number_list=three_one_morphology(GG,all_edge_G)       
      
    return ij_participate_motif_number_list
def float_rand(a, b, size):
   return a + ((b - a) * np.random.random(size))

# 一行数组限定每个元素取值范围
def bound_SOSCD(l_bound, u_bound, x, n):
    for j in range(n):
        if x[j] < l_bound:
            x[j] = 2*l_bound - x[j]
            if x[j] > u_bound:
                x[j] = u_bound
        elif x[j] > u_bound:
            x[j] = 2*u_bound - x[j]
            if x[j] < l_bound:
                x[j] = l_bound            
    return x

def clean_up_random(X,n,NP,threshold_value):
    # 对每个种群个体进行操作
    for i in range(NP):
        # 在个体i中随机选择get_num个节点进行社区调整
        get_num = np.random.randint(1, n)
        # 保存在use_node_index
        use_node_index = np.random.permutation(np.arange(n))[0:get_num]      
        # 对use_node_index中的节点,进行社区标号纠错
        for rand_i in range(get_num):           
           # 针对use_node_index中的每一个节点进行社区标号纠错
            node=use_node_index[rand_i]
            # 确定节点node的所有邻域个体
            neigh_node=Gi.neighbors(node)            
            # 构建节点node自身及邻居集合列表
            all_adj_node = copy.deepcopy(neigh_node)           
            all_adj_node.append(node)                  
            # node的社区编号
            node_comm = copy.deepcopy(X[i][node])
            # node的邻域节点的社区编号
            node_neigh_comm = copy.deepcopy(X[i][neigh_node]) 
            # 计算节点node的CV
            # 节点node与邻域个体属于不同社区的数目
            different_comm_number=0
            for k in node_neigh_comm:
                if node_comm!=k:
                   different_comm_number+=1
            # 节点node的度
            degree_node=len(neigh_node)
            # 节点node的CV值
            CV_node=float(different_comm_number)/degree_node
            # 判断CV是否大于阈值
            # 若是，则说明节点node与邻域节点不在同一社区的概率较大
            # 节点社区标号错误,选择邻域节点中出现的社区标号
            if CV_node >= threshold_value:
               # 邻域节点社区候选列表，及其规模 
               comm_list = copy.deepcopy(pd.value_counts(node_neigh_comm).index)
               comm_list = copy.deepcopy(comm_list.tolist())
               #print(comm_list)
               comm_num =  copy.deepcopy(pd.value_counts(node_neigh_comm).values)
               comm_num =  copy.deepcopy(comm_num.tolist())
               #根据comm_num确定comm_list中的最大规模社区和次大规模社区
               # 最大规模社区集合 = max_comm_id             
               max_comm_num = max(comm_num)
               max_comm_id = []
               max_comm_index = []
               for k in range(len(comm_num)):
                   if comm_num[k] == max_comm_num:
                       max_comm_id.append(comm_list[k])
                       max_comm_index.append(k)                       
               #从comm_num和comm_list中清除最大社区信息
               del_comm_num = []  #清除最大社区信息后的社区规模集合
               del_comm_list = [] #清除最大社区信息后的社区标号集合
               for k in range(len(comm_list)):
                   if comm_num[k] != max_comm_num:
                       del_comm_list.append(comm_list[k])
                       del_comm_num.append(comm_num[k])
               #次大规模社区集合 = second_max_comm_id 
               if len(del_comm_list) > 0:
                   second_max_comm_num = max(del_comm_num)
                   second_max_comm_id = []
                   for k in range(len(del_comm_num)):
                       if del_comm_num[k] == second_max_comm_num:
                           second_max_comm_id.append(del_comm_list[k])
               else:
                   second_max_comm_num = 0
                   second_max_comm_id = []
               # 最大规模社区集合 + 次大规模社区集合 = 可选社区集合
               comm_id_for_choice = copy.deepcopy(max_comm_id + second_max_comm_id)
#                逐个遍历可选社区集合comm_id_for_choice中的每个社区 
#                以不同的概率对节点node的社区进行重置
               for comm_id in comm_id_for_choice:                   
                   if comm_id == comm_id_for_choice[0] and (comm_id in max_comm_id):
                       #第1个最大规模社区，100%概率重置
                       X[i][node] = comm_id
                   elif (comm_id != comm_id_for_choice[0]) and (comm_id in max_comm_id):                       
                       if random.random() < 0.5:
                           X[i][node] = comm_id
                   elif (comm_id != comm_id_for_choice[0]) and (comm_id in second_max_comm_id):
                       if random.random() < 0.2:
                           X[i][node] = comm_id 
                
#               probability_number=random.random()
#               if (probability_number < 0.2) and (len(second_max_comm_id)!=0):
#                   comm_id=random.choice(second_max_comm_id)
#                   X[i][node] = comm_id
#               elif (probability_number >=0.2) and (len(max_comm_id)!=0):
#                   comm_id=random.choice(max_comm_id)
#                   X[i][node] = comm_id
                        
    # 一个种群个体操作完毕                         
    # 返回纠错后的新种群
    return X                   

# def local_optimization(bestx,bestfit,n):
#     # 对单个最优解个体进行局部搜索
#     for i in range(n):       
#         # 节点i的社区标号
#         comm_id_i=bestx[i]
#         # 节点i的邻居节点
#         i_neighbors=Gi.neighbors(i)
#         # 邻居节点的社区标号
#         comm_id_i_neighbors=bestx[i_neighbors]
#         # 节点i的备选社区标号
#         comm_id_i_choice = copy.deepcopy(pd.value_counts(comm_id_i_neighbors).index)
#         comm_id_i_choice = copy.deepcopy(comm_id_i_choice.tolist())
#         if comm_id_i in comm_id_i_choice:
#             comm_id_i_choice.remove(comm_id_i)           
#         #逐个更换节点i的社区标号
#         for j in range(len(comm_id_i_choice)):
#             #print(j)
#             x_change = copy.deepcopy(bestx)
#             x_change[i] = comm_id_i_choice[j]
# #            x_change_fit = ig.GraphBase.modularity(Gi,x_change)
#             x_change_fit=ig.GraphBase.modularity(Gi,x_change,weights=ij_participate_motif_number_list)
#             # 比较新个体与原个体的优劣
#             if x_change_fit > bestfit:
#                 bestx = copy.deepcopy(x_change)
#                 bestfit = copy.deepcopy(x_change_fit)
#     return bestx
#def save(data,path):
#    f = xlwt.Workbook() # 创建工作簿
#    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True) # 创建sheet
#    [h, l] = Adj.shape # h为行数，l为列数
#    for i in range(h):
#        for j in range(l):
#            sheet1.write(i, j, str(data[i, j]))
#    f.save(path)
def save(data,path):
    f = xlsxwriter.Workbook(path) # 创建工作簿
    sheet1 = f.add_worksheet(u'sheet1') # 创建sheet
    [h, l] = data.shape # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(int(i), int(j), int(data[i, j]))
    f.close() 


# =============================================================================
# 网络信息 
# network
# =============================================================================
## 真实网络
path = r"data/经典数据集"
## 真实网络
karate_network = path + r'/karate.txt'
polbooks_network = path + r'/polbooks.txt'
dolphins_network = path + r'/dolphins.txt'
polbooks_network = path + r'/polbooks.txt'
football_network = path + r'/football.txt'
jazz_network = path + r'/jazz.txt'
metabolic_network = path + r'/metabolic.txt'
netscience_network = path + r'/netscience.txt'
email_network = path + r'/email.txt'
road_minnesota_network = path + r'/road_minnesota.txt'
cora_network = path + r'/cora.txt'
powergrid_network = path + r'/powergrid.txt'
pgp_network = path + r'/pgp.txt'

beican_9_network = path + r'/9_beican.txt'
zhang_network = path + r'/zhang.txt'
lesmis_network = path + r'/lesmis.txt'
## 功能网络
func_path = r"data/功能网络"
macaque_network = func_path + r'/macaque.txt'

## 人工合成网络
LFR_path = r"data/人工数据集/LFR_1000"
for i in range(1):
    lfr_no = i
    lfr_network = LFR_path + r'/network'+str(lfr_no)+'.txt'
    
    # 选择网络
    # network = lfr_network
    # network_name = "network" + str(lfr_no)
    network = karate_network
    network_name = 'karate'
    G1 = nx.read_edgelist(network,create_using=nx.Graph())
    G1 = G1.to_undirected()
    
    # 获取网络数据中的边列表，并根据其使用igraph创建网络
    Gi=ig.Graph.Read_Edgelist(network)
    Gi=Gi.subgraph(map(int,G1.nodes()))          
    Gi=Gi.as_undirected()
    
    edge_all = Gi.get_edgelist()
      
    # 各参数设置
    # =============================================================================
    n=G1.number_of_nodes()
    NP = 150
    c = 50 #社区的真实划分数
    Gen = 400  #进化代数
    threshold_value = 0.25  #阈值
    # 各标记列表
    Mlist = {1:"M1",2:"M2",3:"M3",4:"M4",5:"M5",6:"M6",7:"M7",8:"M8"} #模体选择列表
    Qlist = {1:"Q",2:"Qg",3:"Qc_FCD",4:"Qc_OCD",5:"Qov"} # 模块度函数列表
    nmmlist = {1:"NOMM",2:"NMM",3:"MNMM",4:"NWMM"} # nmm操作列表
    # 本次算法使用的标记
    M_flag = Mlist[1]
    Q_flag = Qlist[1] #模块度函数 Q
    # 独立运行运行次数
    Independent_Runs = 5 #本次实验独立运行次数
    NMIflag = 1 # 0:关闭NMI，1:开启NMI
     
    # =============================================================================
    # 构建基于模体M1的加权网络
    # =============================================================================
    
    # 获得无权网络邻接矩阵
    G2 = nx.Graph() 
    G2.add_nodes_from([i for i in range(n)])
    G2.add_edges_from(edge_all)
    adj= nx.adjacency_matrix(G2)
    adj=adj.todense() 

    G = net_stru_func.construct_weighted_network(G2,edge_all,n,M_flag) #构建出基于M_flag模体加权的网络        
    # 构建基于模体的加权网络邻接矩阵motif_adj
    motif_adj = nx.adjacency_matrix(G)
    motif_adj = motif_adj.todense() 
    me_adj = motif_adj + adj #总权重矩阵=模体权重+边权重
    save(motif_adj,network_name + '_motif_adj.xlsx')
