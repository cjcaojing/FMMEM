# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:48:34 2022

@author: bozite
"""
import xlwt
import xlsxwriter
import igraph as ig
import scipy
# import cython_function as cfunc
import networkx as nx
# import geatpy as ea # import geatpy
import pandas as pd
import time
import find_motifs as fm
import numpy as np
import copy
from pandas import DataFrame
from sklearn.cluster import KMeans
import fai_m as fai_m_func
# from heapq import nsmallest
import itertools
from itertools import combinations

def undirected_motif_xuanze(motif):
    M_ins = nx.Graph()####模体按照Benson论文中设置
    if motif == "motif_1":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2,3),(3, 1)])  # 连通
    elif motif == "motif_2":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3),(3,2)])  # 连通
    elif motif == "motif_3":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 4),(2,4),(3,4)])  # 连通
    elif motif == "motif_4":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 3),(3,4),(4,2)])  # 连通
    elif motif == "motif_5":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 4),(2,3),(2,4),(3,4)])  # 连通
    elif motif == "motif_6":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 3),(1,4),(2,4),(2,3),(3,4)])  # 连通
    elif motif == "motif_7":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 2),(2,4),(4,3),(3,1)])  # 连通
    elif motif == "motif_8":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 2),(1, 3),(1,4),(2,3),(2,4),(3,4)])  # 连通
    elif motif == "edge":
        M_ins.add_nodes_from([1, 2])
        M_ins.add_edges_from([[1, 2]])  # 连通
    return  M_ins

def motif_jiaquan_gai(G,Gi,edge_num,total_motif_lists_1):
    motif_node_set, motif_edge_set = total_motif_lists_1 # 获得所有模体的节点和边
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
    # weights = original_weights 
    weights=[]
    for each_weight in original_weights:
        each_weight += 0
        weights.append(each_weight) 
    return weights

def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1/np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

def benson_normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    # R_sqrt = np.where(R == 0, 0, 1/np.sqrt(R))
    R_sqrt = np.divide(1, np.sqrt(R),out=np.zeros_like(R), where=R!=0) #R中为0的地方保持不变
    # print(R_sqrt)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

def m1(G,edge_all):
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
def m1_motifadd(G,edge_all):
    n=len(G.nodes())
    motif_matrix=np.zeros((n,n))
    for u,v in edge_all:    
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        u_friends=list(u_friends)
        v_friends=list(v_friends)
        if (u_friends == []) or (v_friends == []):
            pass
        else:
            motif_matrix[u][v]+=len(set(u_friends) & set(v_friends))
            motif_matrix[v][u]+=len(set(u_friends) & set(v_friends))
    return  motif_matrix 
#计算模体结构3-2(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/2为模体数量
def m2(G,edge_all):
    ij_participate_motif_number_list=[]    
    for u,v in edge_all:      
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
        ij_participate_motif_number_list.append(len(u_mor) + len(v_mor))
    return ij_participate_motif_number_list
def m2_motifadd(G,edge_all):
    n=len(G.nodes())
    motif_matrix=np.zeros((n,n))    
    for u,v in edge_all:      
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
        for j in u_mor:
            motif_matrix[j][v]+=1
            motif_matrix[v][j]+=1
        for j in v_mor:
            motif_matrix[j][u]+=1
            motif_matrix[u][j]+=1        
        motif_matrix[u][v]+=len(u_mor) + len(v_mor)
        motif_matrix[v][u]+=len(u_mor) + len(v_mor)
    return motif_matrix
#计算模体结构4-1(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/3为模体数量
def m3(G,edge_all):    
    ij_participate_motif_number_list=[]    
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
        u_list = []
        v_list = []
        if len(u_mor) <= 1:
            deta1 = 0
        else:
            for i in itertools.combinations(u_mor,2):
                if (min(i[0],i[1]),max(i[0],i[1])) not in edge_all:
                    u_list.append((min(i[0],i[1]),max(i[0],i[1])))
            deta1 = int(len(u_list))               
        if len(v_mor) <= 1:
            deta2 = 0
        else:
            for i in itertools.combinations(v_mor,2):
                if (min(i[0],i[1]),max(i[0],i[1])) not in edge_all:
                    v_list.append((min(i[0],i[1]),max(i[0],i[1])))
            deta2 = int(len(v_list))
        ij_participate_motif_number_list.append(deta2+deta1)
    return ij_participate_motif_number_list
def m3_motifadd(G,edge_all):
    #求列表的长度
    #生成全0矩阵
    n=len(G.nodes())
    motif_matrix=np.zeros((n,n)) 
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
        u_list = []
        v_list = []
        if len(u_mor) <= 1:
            deta1 = 0
        else:
            for i in itertools.combinations(u_mor,2):
                if (min(i[0],i[1]),max(i[0],i[1])) not in edge_all:
                    u_list.append((min(i[0],i[1]),max(i[0],i[1])))
            deta1 = int(len(u_list)) 
              
        if len(v_mor) <= 1:
            deta2 = 0
        else:
            for i in itertools.combinations(v_mor,2):
                if (min(i[0],i[1]),max(i[0],i[1])) not in edge_all:
                    v_list.append((min(i[0],i[1]),max(i[0],i[1])))
            deta2 = int(len(v_list))
        motif_matrix[u][v]=deta2+deta1
        motif_matrix[v][u]=deta2+deta1
        for i in u_list:
            motif_matrix[i[0]][i[1]]+=1/3
            motif_matrix[i[1]][i[0]]+=1/3
            motif_matrix[v][i[0]]+=1/3
            motif_matrix[i[0]][v]+=1/3
            motif_matrix[v][i[1]]+=1/3
            motif_matrix[i[1]][v]+=1/3
        for i in v_list:
            motif_matrix[i[0]][i[1]]+=1/3
            motif_matrix[i[1]][i[0]]+=1/3
            motif_matrix[u][i[0]]+=1/3
            motif_matrix[i[0]][u]+=1/3
            motif_matrix[u][i[1]]+=1/3
            motif_matrix[i[1]][u]+=1/3        
    return motif_matrix
#计算模体结构4-2（连边参与模体构造数量）,sum(ij_participate_motif_number_list)/3为模体数量
def m4(G,edge_all):
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
#计算模体结构4-2（连边参与模体构造数量）,sum(ij_participate_motif_number_list)/3为模体数量
def m4_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n)) 
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
            motif_matrix[u][v]+=0
            motif_matrix[v][u]+=0
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
            motif_matrix[u][v]+=deta
            motif_matrix[v][u]+=deta
            for mor_list_i in mor_list:
                motif_matrix[mor_list_i[0]][mor_list_i[1]]+=1
                motif_matrix[mor_list_i[1]][mor_list_i[0]]+=1
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
                node_i=node_number_list[i][0]
                if (min(node_i,u),max(node_i,u)) in edge_all:
                    motif_matrix[node_i][u]+=node_number_list[i][1]
                    motif_matrix[u][node_i]+=node_number_list[i][1] 
                    motif_matrix[node_i][v]+=node_number_list[i][1]
                    motif_matrix[v][node_i]+=node_number_list[i][1]                     
                elif (min(node_i,v),max(node_i,v)) in edge_all:
                    motif_matrix[node_i][v]+=node_number_list[i][1]
                    motif_matrix[v][node_i]+=node_number_list[i][1]
                    motif_matrix[node_i][u]+=node_number_list[i][1]
                    motif_matrix[u][node_i]+=node_number_list[i][1]
    return motif_matrix
#计算模体结构4-3(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def m5(G,edge_all):
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
        #如果节点u的邻居节点除v外只有一个，那么无法构成一个三角形，因此无模体
        if len(u_mor) <= 1:
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            #如果节点u的邻居节点除v外有多个，那么判断多个节点能构成几个全封闭三角形
            for i in itertools.combinations(u_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    u_list0.append((min_pq,max_pq))            
            deta1 = 0
            for p,q in u_list0:
                ij_participate_motif_number_list[edge_all.index((min(u,p),max(u,p)))]=ij_participate_motif_number_list[edge_all.index((min(u,p),max(u,p)))]+1
                ij_participate_motif_number_list[edge_all.index((min(u,q),max(u,q)))]=ij_participate_motif_number_list[edge_all.index((min(u,q),max(u,q)))]+1                
                deta1 += 1
                index0=edge_all.index((p,q))
                ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta1
        if len(v_mor) <= 1:
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+0
        else:
            for i in itertools.combinations(v_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    v_list0.append((min_pq,max_pq)) 
            deta2 = 0
            for p,q in v_list0:
                ij_participate_motif_number_list[edge_all.index((min(v,p),max(v,p)))]=ij_participate_motif_number_list[edge_all.index((min(v,p),max(v,p)))]+1
                ij_participate_motif_number_list[edge_all.index((min(v,q),max(v,q)))]=ij_participate_motif_number_list[edge_all.index((min(v,q),max(v,q)))]+1                
                deta2 += 1
                index0=edge_all.index((p,q))
                ij_participate_motif_number_list[index0]=ij_participate_motif_number_list[index0]+1
            ij_participate_motif_number_list[index_uv]=ij_participate_motif_number_list[index_uv]+deta2 
    return ij_participate_motif_number_list
def m5_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n))    
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
        u_list0 = []
        v_list0 = []
        #如果节点u的邻居节点除v外只有一个，那么无法构成一个三角形，因此无模体
        if len(u_mor) <= 1:
            motif_matrix[u][v]=motif_matrix[u][v]+0
            motif_matrix[v][u]=motif_matrix[v][u]+0
        else:
            #如果节点u的邻居节点除v外有多个，那么判断多个节点能构成几个全封闭三角形
            for i in itertools.combinations(u_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    u_list0.append((min_pq,max_pq))            
            deta1 = 0
            for p,q in u_list0:
                motif_matrix[u][q]=motif_matrix[u][q]+1
                motif_matrix[q][u]=motif_matrix[q][u]+1
                motif_matrix[u][p]=motif_matrix[u][p]+1
                motif_matrix[p][u]=motif_matrix[p][u]+1
                
                motif_matrix[v][q]=motif_matrix[v][q]+1
                motif_matrix[q][v]=motif_matrix[q][v]+1
                motif_matrix[v][p]=motif_matrix[v][p]+1
                motif_matrix[p][v]=motif_matrix[p][v]+1               
                deta1 += 1
                motif_matrix[p][q]=motif_matrix[p][q]+1
                motif_matrix[q][p]=motif_matrix[q][p]+1   
            motif_matrix[u][v]=motif_matrix[u][v]+deta1
            motif_matrix[v][u]=motif_matrix[v][u]+deta1
        if len(v_mor) <= 1:
            motif_matrix[u][v]=motif_matrix[u][v]+0
            motif_matrix[v][u]=motif_matrix[v][u]+0
        else:
            #如果节点u的邻居节点除v外有多个，那么判断多个节点能构成几个全封闭三角形
            for i in itertools.combinations(v_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    v_list0.append((min_pq,max_pq))            
            deta1 = 0
            for p,q in v_list0:
                motif_matrix[v][q]=motif_matrix[v][q]+1
                motif_matrix[q][v]=motif_matrix[q][v]+1
                motif_matrix[v][p]=motif_matrix[v][p]+1
                motif_matrix[p][v]=motif_matrix[p][v]+1
                
                motif_matrix[u][q]=motif_matrix[u][q]+1
                motif_matrix[q][u]=motif_matrix[q][u]+1
                motif_matrix[u][p]=motif_matrix[u][p]+1
                motif_matrix[p][u]=motif_matrix[p][u]+1               
                deta1 += 1
                motif_matrix[p][q]=motif_matrix[p][q]+1
                motif_matrix[q][p]=motif_matrix[q][p]+1   
            motif_matrix[u][v]=motif_matrix[u][v]+deta1
            motif_matrix[v][u]=motif_matrix[v][u]+deta1
    return motif_matrix
#计算模体结构4-4(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/5为模体数量
def m6(G,edge_all):
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
                deta = int(len(cn_edge0))
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
def m6_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n)) 
    for u,v in edge_all: 
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            motif_matrix[u][v]+=0
            motif_matrix[v][u]+=0
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                motif_matrix[u][v]+=0
                motif_matrix[v][u]+=0
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
                for p0,q0 in  cn_edge0: 
                   motif_matrix[p0][q0]+=1
                   motif_matrix[q0][p0]+=1     
                deta = int(len(cn_edge))
                motif_matrix[u][v]+=deta
                motif_matrix[v][u]+=deta
                #统计mor_list中每个节点参与连边数量，生成一个包含（节点i，节点i数量）的列表
                node_list=[]
                node_number_list=[]
                for i in range(len(cn_edge0)):
                    for j in cn_edge0[i]:
                       node_list.append(j) 
                set_node_list=set(node_list)
                
                for i in set_node_list:
                    node_number_list.append([i,node_list.count(i)])
                # print(node_number_list)
                #判断每个节点与u、v哪个节点相连接，并更新ij_participate_motif_number_list(连边参与模体构造数量列表），在连边所在位置更新数值
                for i in range(len(node_number_list)):
                    node_i=node_number_list[i][0]
                    if (min(node_i,u),max(node_i,u)) in edge_all:
                        motif_matrix[u][node_i]+=node_number_list[i][1]
                        motif_matrix[node_i][u]+=node_number_list[i][1]    
                    if (min(node_i,v),max(node_i,v)) in edge_all:
                        motif_matrix[v][node_i]+=node_number_list[i][1]
                        motif_matrix[node_i][v]+=node_number_list[i][1]  
    return motif_matrix
#计算模体结构4-5(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/4为模体数量
def m7(G,edge_all):
    ij_participate_motif_number_list=[]
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
        mor_list = []
        if (u_mor == []) or (v_mor == []):
            ij_participate_motif_number_list.append(0)
        else:
            for i in u_mor:
                for j in v_mor:
                    mor_list.append((i,j))
            deta = 0
            for p,q in mor_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    deta += 1
            ij_participate_motif_number_list.append(deta)
    return ij_participate_motif_number_list
def m7_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n)) 
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
        mor_list = []
        if (u_mor == []) or (v_mor == []):
            pass
        else:
            for i in u_mor:
                for j in v_mor:
                    mor_list.append((i,j))
            deta = 0
            for p,q in mor_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    if p in u_friends:
                        motif_matrix[p][v]+=(1/4)
                        motif_matrix[v][p]+=(1/4)
                    else:
                        motif_matrix[p][u]+=(1/4)
                        motif_matrix[u][p]+=(1/4)   
                    if q in u_friends:
                        motif_matrix[q][v]+=(1/4)
                        motif_matrix[v][q]+=(1/4)
                    else:
                        motif_matrix[q][u]+=(1/4)
                        motif_matrix[u][q]+=(1/4)                     
                    deta += 1    
            motif_matrix[u][v]+=deta
            motif_matrix[v][u]+=deta
    return motif_matrix
##计算模体结构4-6(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/6为模体数量
def m8(G,edge_all):
    ij_participate_motif_number_list=[]
    for u,v in edge_all:    
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            ij_participate_motif_number_list.append(0)
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                ij_participate_motif_number_list.append(0)
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
                ij_participate_motif_number_list.append(deta)
    return ij_participate_motif_number_list   
#计算模体结构4-6(连边参与模体构造数量)，sum(ij_participate_motif_number_list)/6为模体数量
def m8_motifadd(G,edge_all):
    #求列表的长度
    n=len(G.nodes())
    #生成全0矩阵
    motif_matrix=np.zeros((n,n)) 
    for u,v in edge_all:    
        u_friends = G.neighbors(u)
        v_friends = G.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            pass
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                pass
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
                motif_matrix[u][v]+=deta
                motif_matrix[v][u]+=deta
    return motif_matrix 

def moti_num(motif_func_list, M, all_edge_G):    
    data=DataFrame(list(all_edge_G),columns=['src','dst'])
#    GG = nx.from_pandas_dataframe(data,'src','dst',create_using=nx.Graph())
    GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    ij_participate_motif_number_list=motif_func_list[M](GG,all_edge_G)       
    return ij_participate_motif_number_list 

def moti_num_motifadd(motifadd_func_list, M, all_edge_G):    
    data=DataFrame(list(all_edge_G),columns=['src','dst'])
#    GG = nx.from_pandas_dataframe(data,'src','dst',create_using=nx.Graph())
    GG = nx.from_pandas_edgelist(data,'src','dst',create_using=nx.Graph())
    matrix1=motifadd_func_list[M](GG,all_edge_G)       
    return matrix1  

def save(data,path):
    f = xlsxwriter.Workbook(path) # 创建工作簿
    sheet1 = f.add_worksheet(u'sheet1') # 创建sheet
    [h, l] = data.shape # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(int(i), int(j), int(data[i, j]))
    f.close()  

network_name = "9_beican" #网络
M = "M1"  #模体


# df_news = pd.read_table("data/real/"+network_name+"_groundtruth.txt", header=None)
# real_membership = []
# for i in df_news[0]:
#     real_membership.append(i)
# k = len(set(real_membership))#社区数
k = 2
# print("k=",k)
G = nx.read_edgelist("data/"+network_name + '.txt')
Gi = ig.Graph.Read_Edgelist("data/"+network_name + '.txt')
Gi = Gi.subgraph(map(int, G.nodes()))  # G.nodes()获取一个Graph对象中的node数组,
Gi = Gi.as_undirected()
edge_num = Gi.ecount()
node_num = Gi.vcount()
edges = copy.deepcopy(Gi.get_edgelist())
all_node_list = list(range(node_num))
all_node_set = set(all_node_list)
# motif_ins = undirected_motif_xuanze('motif_4')
edge_all = edges
# M_N = nx.Graph.number_of_nodes(motif_ins)#几阶模体
motif_func_list = {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "M6": m6, "M7": m7, "M8": m8}
motifadd_func_list = {"M1": m1_motifadd, "M2": m2_motifadd, "M3": m3_motifadd, "M4": m4_motifadd, "M5": m5_motifadd, "M6": m6_motifadd, "M7": m7_motifadd, "M8": m8_motifadd}

# total_motif_nums = fm.total_motif_num(G, motif_ins, directed=False, weighted=False)
# total_motif_lists_1 = fm.total_motif_list(G, motif_ins, directed=False, weighted=False)
# total_motif_lists = total_motif_lists_1[0]######网络中所有模体
# total_motif_lists = np.array(total_motif_lists)

ij_participate_motif_number_list = moti_num(motif_func_list, M, edge_all)
############## 网络的权重  ###################
wei = copy.deepcopy(ij_participate_motif_number_list)
# wei_2 = motif_jiaquan_gai(G,Gi,edge_num,total_motif_lists_1)
# xxxx
st1 = time.process_time()
# xxx
Motif_Matrix = moti_num_motifadd(motifadd_func_list, M, edge_all)    
# save(Motif_Matrix, "test.xlsx")
# G2 = nx.from_numpy_matrix(Motif_Matrix)
# Motif_Matrix = np.matrix(nx.adjacency_matrix(G2,nodelist=all_node_list).todense(),dtype = np.float32)


# G2 = nx.Graph()
# for inde,ele in enumerate(Gi.get_edgelist()):
#     u,v = ele
#     # print(ele)
#     G2.add_edge(u, v, weight=wei[inde])

# Motif_Matrix = np.matrix(nx.adjacency_matrix(G2,nodelist=all_node_list).todense(),dtype = np.float32)
# # print(nx.adjacency_matrix(G2))
# Motif_Matrix = Motif_Matrix.getA()

# Ln_g = nx.normalized_laplacian_matrix(G2, nodelist=all_node_list, weight="weight").toarray()#拉普拉斯矩阵
# xxx
# Ln_g = scipy.sparse.csgraph.laplacian(Motif_Matrix, normed=True, return_diag=True, use_out_degree=False)

###计算拉普拉斯矩阵###
# R = np.sum(Motif_Matrix, axis=1)
# # R_sqrt = np.where(R == 0, 0, 1/np.sqrt(R))
# R_sqrt = np.divide(1, np.sqrt(R), where=R!=0) 
# xxx
Ln_g = benson_normalized_laplacian(Motif_Matrix)


# xxx
# #计算特征向量与特征值，收敛容差=1e-6,Afun 矩阵有对称性，A为实数
eigenvalue, featurevector = scipy.sparse.linalg.eigsh(Ln_g,k=k, tol=1e-4, which='SM',return_eigenvectors=True)

uu = featurevector[:,:k]#对应matlab中的U

mat_u = np.abs(uu)

mat_u_fang_sqrt = np.sqrt(np.sum(mat_u ** 2, axis=1))
ttt = 1 / np.sqrt(np.sum(mat_u ** 2, axis=1))
# ttt = np.divide(1, mat_u_fang_sqrt,out=np.zeros_like(mat_u_fang_sqrt), where=mat_u_fang_sqrt!=0) 

# Dd = scipy.sparse.spdiags(np.eye(ttt.shape[0]), ttt, ttt.shape[0], ttt.shape[0], format=None)

Dd = scipy.sparse.diags(ttt, offsets=0, shape=None, format=None, dtype=None)#.toarray()#对应matlab中的D
D_arr  = Dd.toarray()
T = Dd * uu
# xxx

st2 = time.process_time()

rep = 1

Idx=KMeans(n_clusters=k, random_state=0,max_iter=rep).fit(T)#聚类

membership = Idx.labels_


en1 = time.process_time()
print('拉普拉斯时间为',st2-st1)
print('k-means时间为',en1-st2)
print('总时间为',en1-st1)
####指标计算#####

# node_set= []#####每个节点参与的模体#####
# for node in range(node_num):
#     Node_set,edge_set=fm.node_in_motif_list(G, motif_ins, node, directed=False, weighted=False)
#     node_set.append(Node_set)

# node_motif_num_list=[]#####点模体度集合#####
# for i in range(node_num):
#     node_motif_num_list.append(fm.node_in_motif(G,motif_ins,i,directed=False,weighted=False))
# node_motif_num_list = np.array(node_motif_num_list)

# # faim_flag = 1
# M_N = nx.Graph.number_of_nodes(motif_ins)#几阶模体
# node_motif_num_arr = np.asarray(node_motif_num_list, dtype=int)
# total_motif_len = len(total_motif_lists)
# total_motif_arr = np.zeros((total_motif_len, M_N), dtype = int, order = 'C')
# for imt_index, imt in enumerate(total_motif_lists):
#     total_motif_arr[imt_index,:] = np.asarray(imt, dtype=int)

# df_benson = pd.read_table(r'celegans_final_mbifan_k=2_number1.txt', header=None)
# benson_result = []
# for i in df_benson[0]:
#     benson_result.append(i)
# benson_result = np.array(benson_result)


com_list_symbol=membership#membership#benson_result

# membershipSet = list(set(com_list_symbol))
# membership_arr = np.asarray(com_list_symbol, dtype=int)
# membershipSet_arr = np.asarray(membershipSet, dtype=int)
# membershipSet_len = len(membershipSet)
# fai_m1_value2 


# nmi = ig.compare_communities(real_membership, list(com_list_symbol), method='nmi', remove_none=False)
# print('nmi', nmi)

Qw = ig.GraphBase.modularity(Gi, com_list_symbol ,directed = False, weights = wei)
print("Qw",Qw)
print("membership=",com_list_symbol)

# conductance = fai_m_func.fai_m(1, membership_arr, membershipSet_arr,membershipSet_len, node_motif_num_arr, 
#                               total_motif_arr, total_motif_len,node_num, M_N)
# print("faim",conductance)
# faimean=fai_m_func.fai_m(2, membership_arr, membershipSet_arr,membershipSet_len, node_motif_num_arr, 
#                               total_motif_arr, total_motif_len,node_num, M_N)
# print("faimean",faimean)













