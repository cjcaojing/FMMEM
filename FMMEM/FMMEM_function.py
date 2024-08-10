# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:55:16 2022

@author: WYW
"""
"""
    function: 各类功能函数
"""

import time
# from matplotlib import pyplot as plt

# 引入外部函数
# import find_motifs as fm

# C函数

# =============================================================================
# 获得无向模体结构
# =============================================================================
import numpy as np
import random  as rd
import networkx as nx
import igraph as ig
import copy
import itertools
from igraph.clustering import VertexClustering
# from sklearn import metrics
# from sklearn.metrics import precision_score, recall_score, f1_score
import cython_function as cfunc
def get_undirected_motif(motif):
    M_ins = nx.Graph()  # 模体按照Benson论文中设置
    if motif == "M1":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2, 3), (3, 1)])  # 连通
    elif motif == "M2":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3), (3, 2)])  # 连通
    elif motif == "M3":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 4), (2, 4), (3, 4)])  # 连通
    elif motif == "M4":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 3), (3, 4), (4, 2)])  # 连通
    elif motif == "M5":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 4), (2, 3), (2, 4), (3, 4)])  # 连通
    elif motif == "M6":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 3), (1, 4), (2, 4), (2, 3), (3, 4)])  # 连通
    elif motif == "M7":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from([(1, 2), (2, 4), (4, 3), (3, 1)])  # 连通
    elif motif == "M8":
        M_ins.add_nodes_from([1, 2, 3, 4])
        M_ins.add_edges_from(
            [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])  # 连通
    elif motif == "edge":
        M_ins.add_nodes_from([1, 2])
        M_ins.add_edges_from([[1, 2]])  # 连通
    return M_ins

# =============================================================================
#     fit_Qs: 计算种群中每个个体的模糊重叠社区划分的模块度函数Q值
#     Qs: 根据pop计算的模块度值
#     pop: 种群
#     adj: 网络邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群个体数目
#    flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov
# =============================================================================


def fit_Qs(Compactness,motif_info, Qs, pop, adj, n, c, NP, W, m, fit_flag):
#    W = np.sum(adj)  # 权值之和
#    m = np.ravel(np.sum(adj, axis=0))  # adj 各列之和
    for N in range(NP):
        # 计算每个个体的适应度函数值Q
        X = pop[:, N]
        # print(U)
        Q = fit_Q(Compactness,motif_info, X, adj, n, c, W, m, fit_flag)
        Qs.append(Q)

# =============================================================================
#     fit_Q: 计算单个个体的模糊重叠社区划分的模块度函数Q值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
#     flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov
#     return 返回Q值
# =============================================================================


def fit_Q(Compactness, motif_info, X, adj, n, c, W, m, fit_flag):
    Q = 0.0
    if fit_flag == "Q":
        ###Q###
        mod = X.astype('int32')
        Q = cfunc.fit_Q(Compactness, n,mod)
        # Q = Q_value(adj,n,W,m,X)
    elif fit_flag == "Qg":
        ###Qg###
        Q = cfunc.fit_Qg(Compactness, X, n, c)
    elif fit_flag == "Qc":
        ###Qc_FCD###
        Q = cfunc.fit_Qc(Compactness, X, n, c)
    elif fit_flag == "Qc_OCD":
        ###Qc_OCD###
        X_V1 = np.empty((c, n), dtype=float)
        # 离散化
        for k in range(c):
            for i in range(n):
                numki = X[k, i]
                if numki > 0.3:
                    X_V1[k, i] = 1
                else:
                    X_V1[k, i] = 0
        Q = cfunc.fit_Qc(X_V1, adj, n, c, W, m)
    elif fit_flag == "Qov":
        ##Qov###
        Q = cfunc.fit_Qov(X, adj, n, c, W, m)
    elif fit_flag == "MC":
        membershipSet = set(X)
        Xpartition = {}
        for cno in membershipSet:
            Xpartition[cno] = []
        for node, icno in enumerate(X):
            Xpartition[icno].append(node)
        node_motif_num_arr = motif_info['node_mnum']
        node_motif_numSum = np.sum(node_motif_num_arr)
        node_mdict = motif_info['node_mdict']
        total_motif_nums = motif_info['r']
        Q = 1 - cfunc.fai_m(2, Xpartition, node_mdict, node_motif_num_arr, node_motif_numSum, total_motif_nums, n, 3)  # 用Q最大来选择MC最小
    else:
        Q = -1
    return Q

# =============================================================================
# 计算deltaQ值，即将i节点从A社区移动到B社区的deltaQ值
# =============================================================================
def get_deltaQi(i, nodesA, nodesB, motif_adj, W, m):
    nodesB.append(i)
    nodesA_arr, nodesB_arr = np.asarray(nodesA),np.asarray(nodesB)
    A_deltaQ = cfunc.fit_deltaQ(i, motif_adj, nodesA_arr, len(nodesA), W, m,0)
    B_deltaQ = cfunc.fit_deltaQ(i, motif_adj, nodesB_arr, len(nodesB_arr), W, m,1)
    deltaQ = A_deltaQ + B_deltaQ
    return deltaQ



# =============================================================================
#     init_pop_V1: 基于社区编号编码的种群初始化
#     n: 网络节点数目
#     NP： 种群个体数目
#     return: 返回初始化后的种群
# =============================================================================


def init_pop_V1(n, NP):
    pop = np.random.randint(n, size=(n, NP))
    return pop

# =============================================================================
#     bias_init_pop_V1: 基于社区编号编码的种群有偏操作
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群个体数目
#     return: 返回初始化后的种群
# =============================================================================


def bias_init_pop_V1(pop, n, NP, adj):
    bias_pop = copy.deepcopy(pop)
    for N in range(NP):
        # 在该个体中，选择一个节点，将其隶属度赋值给所有相邻节点
        i_node = rd.randint(0, n-1)
        # 寻找节点 i 基于模体的邻居节点 j_m_nodes
        j_m_nodes = np.nonzero(np.ravel(adj[i_node, :]))
        for j in j_m_nodes[0]:
            bias_pop[j, N] = bias_pop[i_node, N]
    return bias_pop

# =============================================================================
# 根据当前的种群信息，获得当前网络中模体对各个社区的隶属度信息
# motif_info['MUs'], XMemMatrix, motif_info, n, D
# =============================================================================


def renewMUs(MUs, XMemMatrix, motif_info, n, D):
    r, mno_arr = motif_info['r'], motif_info['mno_arr']  # 模体数目，模体编号矩阵
    for i in XMemMatrix.keys():
        Xi = XMemMatrix[i]
        c = Xi.shape[0]
        cfunc.renewMUs(MUs[i], Xi, mno_arr, r, c, D)

# =============================================================================
#     bound_check_revise: 边界约束检查与修正
#     X: 个体隶属度矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群个体数目
#     return: 约束检测并修正后的个体
# =============================================================================
def bound_check_revise(X, n, lu_bound):
    low, up = lu_bound[0], lu_bound[1]
    cfunc.bound_check_revise(X, n, low, up)
    
# =============================================================================
# 转化隶属度矩阵
# =============================================================================
def conversion_code(Compactness, node_nei_info, X, me_adj, n):
    # 寻找种群中最大社区数目
    setX = list(set(X))
    setX_len = len(setX)
    cc_X = np.zeros((setX_len, n), dtype=float)
    # 将X转化为partition
    Xpartition = {}
    for cno in setX:
        Xpartition[cno] = []
    for node, icno in enumerate(X):
        Xpartition[icno].append(node)
    for i in range(n):
        # 计算节点对各个模体邻域社区的连接紧密度及隶属度程度
        Cno_Compactness = {}
        compactness_sum = 0.0
        for mjcno in Xpartition.keys():
            # mjcno_nodes = mcno_nodes[mjcno]
            mjcno_nodes = copy.deepcopy(Xpartition[mjcno])
            if i in mjcno_nodes: mjcno_nodes.remove(i) #如果i节点在jcno社区中，则去掉i节点
            Cno_Compactness[mjcno] = sum([Compactness[i,mj] for mj in mjcno_nodes]) #计算i节点对jcno的连接紧密程度
            if Cno_Compactness[mjcno]<0:
                Cno_Compactness[mjcno]=0.0
            compactness_sum+=Cno_Compactness[mjcno]
        # 计算节点对各模体邻域社区的隶属程度（隶属度总和为1）
        for cno in Cno_Compactness.keys():
            cc_X[cno,i]=Cno_Compactness[cno]/compactness_sum
    return cc_X

def conversion_code(Compactness, X, n):
    # 寻找种群中最大社区数目
    setX = list(set(X))
    setX_len = len(setX)
    cc_X = np.zeros((setX_len, n), dtype=float)
    # 将X转化为partition
    Xpartition = {}
    for cno in setX:
        Xpartition[cno] = []
    for node, icno in enumerate(X):
        Xpartition[icno].append(node)
    for i in range(n):
        # 计算节点对各个社区的连接紧密度及隶属度程度
        Cno_Compactness = {}
        compactness_sum = 0.0
        for mjcno in Xpartition.keys():
            mjcno_nodes = copy.deepcopy(Xpartition[mjcno])
            if i in mjcno_nodes: mjcno_nodes.remove(i) #如果i节点在jcno社区中，则去掉i节点
            Cno_Compactness[mjcno] = sum([Compactness[i,mj] for mj in mjcno_nodes]) #计算i节点对jcno的连接紧密程度
            if Cno_Compactness[mjcno]<0:
                Cno_Compactness[mjcno]=0.0
            compactness_sum+=Cno_Compactness[mjcno]
        # 计算节点对各模体邻域社区的隶属程度（隶属度总和为1）
        for cno in Cno_Compactness.keys():
            cc_X[cno,i]=Cno_Compactness[cno]/compactness_sum
    return cc_X

def conversion_codeV1(MB, node_nei_info, motif_info, X, me_adj, n):
    r, mno_arr = motif_info['r'], motif_info['mno_arr']  # 模体数目，模体编号矩阵
    # 寻找种群中最大社区数目
    setX = list(set(X))
    setX_len = len(setX)
    cc_X = np.zeros((setX_len, r), dtype=float)
    Xpartition = {}
    for cno in setX:
        Xpartition[cno] = []
    for node, icno in enumerate(X):
        Xpartition[icno].append(node)
    for Mno, M in enumerate(mno_arr):
        #计算节点对各个模体邻域社区的连接紧密度及隶属度程度
        Cno_Compactness = {}
        compactness_sum = 0.0
        for mjcno in Xpartition.keys():
            Cno_Compactness[mjcno]=0.0
            # mjcno_nodes = copy.deepcopy(Xpartition[mjcno])
            for i in M:
                mjcno_nodes = copy.deepcopy(Xpartition[mjcno])
                if i in mjcno_nodes: mjcno_nodes.remove(i) #去掉本模体
                Cno_Compactness[mjcno] += sum([MB[i,mj] for mj in mjcno_nodes])
            if Cno_Compactness[mjcno]<0:
                Cno_Compactness[mjcno]=0.0
            compactness_sum+=Cno_Compactness[mjcno]
        if compactness_sum==0:
            cnos = [icno for icno in M]
            Mcno = max(cnos,key=cnos.count)
            cc_X[cno,Mno] = 1.0
        else:
            # 计算模体对社区的隶属程度（隶属度总和为1）
            for cno in Cno_Compactness.keys():
                cc_X[cno,Mno]=Cno_Compactness[cno]/compactness_sum
    return cc_X

def conversion_codeV2(motif_info, X, XMem):
    r, mno_arr = motif_info['r'], motif_info['mno_arr']  # 模体数目，模体编号矩阵
    # 寻找种群中最大社区数目
    setX = list(set(X))
    setX_len = len(setX)
    cc_X = np.zeros((setX_len, r), dtype=float)
    Xpartition = {}
    for cno in setX:
        Xpartition[cno] = []
    for node, icno in enumerate(X):
        Xpartition[icno].append(node)
    for Mno, M in enumerate(mno_arr):
        #计算节点对各个模体邻域社区的连接紧密度及隶属度程度
        for jcno in Xpartition.keys():
            MemSum = 0.0
            for i in M:
                MemSum+=XMem[jcno,i]
            cc_X[jcno,Mno]=MemSum/3
    return cc_X

# =============================================================================
#     NMM_funcs: 四种基于邻域社区的节点社区修正操作函数【1:"NOMM",2:"NMM",3:"MNMM",4:"NWMM"】
#     pop: 种群
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体权重矩阵
#     threshold_value: 阈值
#     return: 返回nmm操作后的种群, 适应度值
# =============================================================================


def FNCM_funcs(MB,Compactness, motif_info, node_nei_info, nmm_pop, nmm_fit, n, NP, motif_adj, me_adj, Q_flag, W, m, XS_info, TCM_Flag):
    # 更新种群中个体信息,返回需要调整的种群
    ###FNCM###
    betterFNCM_number,LSCM_Num = FNCM(MB,Compactness,motif_info,node_nei_info,nmm_pop,nmm_fit,n,NP,motif_adj,me_adj,Q_flag,W,m,XS_info, TCM_Flag)
    return (nmm_pop,nmm_fit,betterFNCM_number,LSCM_Num)


# =============================================================================
#     MNMM: 基于邻居节点的社区修正（基于边邻居节点和模体邻居节点,使用了模体权重信息）
#     pop: 种群
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体加权网络的邻接矩阵
#     Q_flag: 模块度函数选择标识
#     nmm_pop: NMM种群
#     nmm_fit: NMM种群中个体对应的模块度函数值
# =============================================================================
def FNCM(MB, Compactness, motif_info, node_nei_info, nmm_pop, nmm_fit, n, NP, motif_adj, me_adj, Q_flag, W, m, XS_info, TCM_Flag):
    better_MCM,CM_Num = 0,0
    seeds = [i for i in range(n)]
    for N in range(NP):
        Xi = copy.deepcopy(nmm_pop[:,N])
        c = Xi.shape[0]
        # 将X转化为partition
        Xpartition = {}
        for cno in set(Xi):
            Xpartition[cno] = []
        for node, icno in enumerate(Xi):
            Xpartition[icno].append(node)
        #随机选择节点进行修正节点
        # rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)]
        # pick = seeds
        i_cnos = {}
        for i in pick: 
            # 获得节点 i 所在的社区
            i_node_c = Xi[i]
            # 获得节点 i 的所有模体邻居节点 j_m_nodes
            j_m_nodes = node_nei_info["motif_adj"][i]
            # 获得模体邻居节点 j 所在的社区
            j_m_nodes_c = [Xi[j] for j in j_m_nodes]
            j_m_nodes_setc = set(j_m_nodes_c)
            m_Cnei = len(j_m_nodes_setc)  # 模体邻居社区数目
            # 获得节点 i 的所有邻居节点(模体邻居+边邻居) j_nodes
            j_nodes = node_nei_info["me_adj"][i]
            # 获得邻居节点 j 所在的社区
            j_nodes_c = [Xi[j] for j in j_nodes]
            j_ndoes_setc = set(j_nodes_c)
            Cnei = len(j_ndoes_setc)  # 节点邻居社区数目
            # 使用CD判断该节点社区划分是否合理
            Vdata = [{},0,0.0]
            cd_i = FNCM_CD_func(MB, motif_info, i, i_node_c, j_m_nodes, j_m_nodes_c, j_m_nodes_setc, m_Cnei, j_nodes, j_nodes_c,j_ndoes_setc, Cnei, Xpartition, me_adj,Vdata)
            if cd_i <= 0:
                # 调整该节点的社区划分
                i_cno = FNCM_P_func(node_nei_info, i, j_nodes, j_nodes_c, j_m_nodes_c, j_m_nodes_setc, m_Cnei, motif_info, Xi, motif_adj, me_adj, c, N, Vdata)
                if i_cno!=i_node_c: i_cnos[i] = i_cno
         
        for i in i_cnos.keys():
            i_node_c = Xi[i]
            i_cno = i_cnos[i]
            Xpartition[i_node_c].remove(i)
            Xpartition[i_cno].append(i)
            Xi[i]= i_cno #调整个体节点社区划分
        if TCM_Flag:
            # 社区合并
            if XS_info[N][0]==nmm_fit[N]:
                XS_info[N][1]+=1
            else:
                XS_info[N][0],XS_info[N][1] = nmm_fit[N],0
            XS_info[N][2]=True if 30<XS_info[N][1]<36 else False 
            
            mergeFlage = merge_community(MB, node_nei_info, Xi, Xpartition, XS_info[N][2])
            if mergeFlage: CM_Num+=1
        Xi_Q = fit_Q(Compactness,motif_info,Xi,motif_adj,n,len(set(Xi)),W,m,Q_flag)
        if Xi_Q >= nmm_fit[N]:
            nmm_pop[:,N] = Xi
            nmm_fit[N] = Xi_Q
            better_MCM+=1
            
    return (better_MCM,CM_Num)


# =============================================================================
#     MNMM_CD_func: 寻找基于模体权重的不合理划分的节点
#     unreasonableNodes 划分不合理的节点
#     pick: 一定数量的随机节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     motif_adj: 模体邻接矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
# =============================================================================
def FNCM_CD_func(MB, motif_info, i, i_node_c, j_m_nodes, j_m_nodes_c, j_m_nodes_setc, m_Cnei, j_nodes, j_nodes_c,j_nodes_setc, Cnei, Xpartition, me_adj, Vdata):
    # =============================================================================
    # 1、针对社区内节点
    # =============================================================================
    # 如果i节点为社区内节点，则不调整
    ijcnoFlag = i_node_c in j_nodes_c 
    if Cnei == 1 and ijcnoFlag:
        return 1
    # =============================================================================
    # 节点i仅有一个候选社区
    # =============================================================================
    if len(j_nodes_setc) == 1:
        return -1
    
    j_nodes_setc = list(j_nodes_setc)
    Cno_P,maxattrCno = {},0
    if m_Cnei < 2:  # 如果该节点的邻接社区仅有一个模体社区
        maxattrCno = cfunc.getCno_mem(i, MB, Cno_P, Xpartition, j_nodes_setc)   

    # =============================================================================
    # 3、社区间节点
    # =============================================================================
    if m_Cnei >1: 
        #计算节点对各个模体邻域社区的连接紧密度及隶属度程度
        # 寻找i的点模体中的节点
        node_motifs = motif_info['node_mdict'][i]  #节点所参与构成的模体
        mnodes = set()
        for m in node_motifs:
            for mi in m: mnodes.add(mi)
        nodes_mem = {}
        for mi in mnodes:
            Cno_membership = {}
            cfunc.getCno_membership(mi, MB, Cno_membership, Xpartition, j_nodes_setc)   
            nodes_mem[mi] = Cno_membership 
        # 计算模体对社区的隶属度
        Cno_P,maxattrCno = {},0
        maxattrCno = cfunc.getCno_P(nodes_mem, node_motifs, j_nodes_setc, Cno_P)   
      
    Vdata[0] = Cno_P
    Vdata[1] = maxattrCno
    cd_i=-1 if i_node_c!=maxattrCno else 1

    return cd_i 

# =============================================================================
#     MNMM_P_func: 寻找节点应划分的社区号
#     node_cnos: 节点及对应的社区划分号
#     nodes: 未正确划分的节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     motif_adj: 模体邻接矩阵
# =============================================================================


def FNCM_P_func(node_nei_info, i, j_nodes, j_nodes_c, j_m_nodes_c, j_m_nodes_setc, m_Cnei, motif_info, Xi, motif_adj, me_adj, c, N, Vdata):
    # 初始化 i 节点对社区 c 的概率
    # 修改该节点的隶属度值，对该节点重新划分社区
    j_nodes_set_c = list(set(j_nodes_c))
    # =============================================================================
    # 节点i仅有一个候选社区
    # =============================================================================
    if len(j_nodes_set_c) == 1:
        i_cno = j_nodes_c[0]
        return i_cno
    
    # m_Cnei == 0, 依概率选择社区，无模体邻居社区
    # if m_Cnei == 0:
    #     i_cno = rd.choice(j_nodes_c)  # choice 依概率选择
    #     return i_cno

    # =============================================================================
    # 节点有多个邻居社区
    # =============================================================================
    # maxAttrCno = Vdata[1]
    # i_cno = maxAttrCno
    # 计算吸引力(节点i划分到Ck社区的概率)
    AttrMcno_dict = Vdata[0]
    c_ps = []
    for  ck in AttrMcno_dict.keys():
        c_ps.append((ck, AttrMcno_dict[ck]))
    # c_ps = list(sorted(c_ps, key=lambda x: x[1], reverse=True))
    # if len(c_ps)>3: c_ps=c_ps[:3]
    i_cno = choice_by_probability(c_ps)  #一定概率可进行随机选择候选社区
    return i_cno

# =============================================================================
# 社区合并
# =============================================================================
def merge_community(MB, node_nei_info, X, Xpartition, CM_flag):
    meger_falg = False
    if not CM_flag: return meger_falg
    # 社区合并
    cnoslist = list(Xpartition.keys())
    rd.shuffle(cnoslist)
    for cno in cnoslist:
        if cno not in Xpartition: continue
        nodes = Xpartition[cno]
        jnodes = set()  # cno的所有邻居节点
        for i in nodes:
            for ii in node_nei_info["me_adj"][i]:
                jnodes.add(ii)
        # 寻找cno的邻接社区
        jnodes_setc = set([X[i] for i in jnodes])
        if cno in list(jnodes_setc):
            jnodes_setc.remove(cno)
        if len(jnodes_setc) == 0:
            continue
        # 相邻社区合并
        JM = {} # cno与各相邻社区之间的紧密程度
        for jcno in jnodes_setc:
            JM[jcno] = 0.0
            for j in Xpartition[jcno]:
                for i in nodes:
                    JM[jcno]+=MB[i,j]
        for jcno in jnodes_setc:
            if JM[jcno]<=0:
                del JM[jcno]
        jM_sum = sum(JM.values())
        if jM_sum==0.0: continue
        Cmax, Tmax = list(jnodes_setc)[0],0
        for jcno in JM.keys():
            Tjcno = JM[jcno]
            if Tjcno>Tmax:
                Tmax,Cmax = Tjcno,jcno
                
        if Tmax>0:
            CMCNO = Cmax
            # print("\n######社团合并########")
            # print("连接紧密程度={},与邻域社区连接的总程度={}".format(JM[CMCNO],jM_sum))
            # print("C1=",Xpartition[cno])
            # print("C2=",Xpartition[CMCNO])
            newNodes = Xpartition[cno] + Xpartition[CMCNO]
            Xpartition[cno] = newNodes  # 更新社区内节
            # 合并节点
            for j in Xpartition[CMCNO]:
                X[j] = cno
            del Xpartition[CMCNO]  # 移除已被合并的节点
            meger_falg=True

    return meger_falg

# def merge_community(MB, node_nei_info, X, Xpartition,motif_adj, CM_flag):
#     meger_falg = False
#     if not CM_flag: return meger_falg
#     # 社区合并
#     cnoslist = list(Xpartition.keys())
#     rd.shuffle(cnoslist)
#     for cno in cnoslist:
#         if cno not in Xpartition: continue
#         nodes = Xpartition[cno]
#         jnodes = set()  # cno的所有邻居节点
#         for i in nodes:
#             for ii in node_nei_info["motif_adj"][i]:
#                 jnodes.add(ii)
#         # 寻找cno的邻接社区
#         jnodes_setc = set([X[i] for i in jnodes])
#         if cno in list(jnodes_setc):
#             jnodes_setc.remove(cno)
#         if len(jnodes_setc) == 0:
#             continue
#         # 相邻社区合并
#         #计算cno社区内和社区间的连接密度
#         Cno_M,cnoTin,cnoTout = 0.0,0,0
#         for i in nodes:
#             for ii in nodes:
#                 cnoTin+=motif_adj[i,ii]
#             for j in jnodes:
#                 cnoTout+=motif_adj[i,j]
#         Cno_M = cnoTin/cnoTout
#         #相邻社区合并准备
#         JM = {}
#         for jcno in jnodes_setc:
#             #计算jcno社区的模体密度
#             jtmp_ndoes = Xpartition[jcno]
#             jjnodes = set()  # jcno的所有邻居节点
#             for j in jtmp_ndoes:
#                 for jj in node_nei_info["motif_adj"][j]:
#                     jjnodes.add(jj)
#             #计算jcno社区内和社区间的模体密度
#             JCno_M,JcnoTin,JcnoTout = 0.0,0,0
#             for i in jtmp_ndoes:
#                 for ii in jtmp_ndoes:
#                     JcnoTin+=motif_adj[i,ii] #社区内
#                 for j in jjnodes:
#                     JcnoTout+=motif_adj[i,j] #社区间
#             JCno_M = JcnoTin/JcnoTout
#             JCno_Cno_Ave_M = (Cno_M + JCno_M)/2 #社区平均模体密度
            
#             newCNodes = nodes + Xpartition[jcno] #cno与jcno新生成的社区
#             #计算新生成的社区模体密度
#             newCno_M,newCnoTin,newCnoTout = 0.0,0,0
#             JnewCNodes = set()  # jcno的所有邻居节点
#             for j in newCNodes:
#                 for jj in node_nei_info["motif_adj"][j]:
#                     JnewCNodes.add(jj)
#             for i in newCNodes:
#                 for ii in newCNodes:
#                     newCnoTin+=motif_adj[i,ii] #社区内
#                 for j in JnewCNodes:
#                     newCnoTout+=motif_adj[i,j] #社区间
#             newCno_M = newCnoTin/newCnoTout
#             JM[jcno] = newCno_M - JCno_Cno_Ave_M
        
#         for jcno in jnodes_setc:
#             if JM[jcno]<=0:
#                 del JM[jcno]
#         jM_sum = sum(JM.values())
#         if jM_sum==0.0: continue
#         Cmax, Tmax = list(jnodes_setc)[0],0
#         for jcno in JM.keys():
#             Tjcno = JM[jcno]
#             if Tjcno>Tmax:
#                 Tmax,Cmax = Tjcno,jcno
                
#         if Tmax>0:
#             CMCNO = Cmax
#             # print("\n######社团合并########")
#             # print("连接紧密程度={},与邻域社区连接的总程度={}".format(JM[CMCNO],jM_sum))
#             # print("C1=",Xpartition[cno])
#             # print("C2=",Xpartition[CMCNO])
#             newNodes = Xpartition[cno] + Xpartition[CMCNO]
#             Xpartition[cno] = newNodes  # 更新社区内节
#             # 合并节点
#             for j in Xpartition[CMCNO]:
#                 X[j] = cno
#             del Xpartition[CMCNO]  # 移除已被合并的节点
#             meger_falg=True

#     return meger_falg


# =============================================================================
#     unreasonableNodes_revise: 修正节点社区编号
#     node_cno_list: 节点和社区编号
#     nmm_pop: nmm种群
#     N: 种群中的第N个个体的序列号
# =============================================================================


def unreasonableNodes_revise(node_cno_list, nmm_pop, N):
    for i_c in node_cno_list:
        i = i_c[0]
        cno = i_c[1]
        nmm_pop[i, N] = cno
    return


# =============================================================================
#     choice_by_probability: 依概率选择
#     c_p_list: 节点 i 的候选社区概率列表
#     return : 选择的社区c
# =============================================================================
def choice_by_probability(c_p_list):
    num = 1000
    choice_list = []
    for c_p in c_p_list:
        c = c_p[0]
        p = c_p[1]
        n = int(p*num)
        choice_list += [c]*n
    ic = rd.choice(choice_list)  # choice() 依概率选择
    return ic

# =============================================================================
# 寻找关于邻域结构对称的节点
# =============================================================================
# def find_target_nodes(node_nei_info, motif_info, Xi, nodes, bestMUs, motif_adj):
#     print("\n#################")
#     for i in nodes:
#         i_node_c = np.argmax(Xi[:, i])  # i节点被划分的社区
#         j_m_nodes = node_nei_info["motif_adj"][i]
#         # 获得模体邻居节点 j 所在的社区
#         j_m_nodes_c = list(np.argmax(Xi[:, j_m_nodes], axis=0))
#         # 获得节点 i 的所有邻居节点(模体邻居+边邻居) j_nodes
#         j_nodes = node_nei_info["me_adj"][i]
#         # 获得邻居节点 j 所在的社区
#         j_nodes_c = list(np.argmax(Xi[:, j_nodes], axis=0))
#         # 计算i节点是否是邻居社区中关于边、模体对称的节点
#         j_m_c_counts = [j_m_nodes_c.count(j_mc)
#                         for j_mc in set(j_m_nodes_c)]  # 模体邻居社区中节点数量
#         # i节点与各模体邻居社区之间的权重总和
#         i_mc_wsum = []
#         j_m_cs = set(j_m_nodes_c)
#         for j_m_c in j_m_cs:
#             i_mc_wsum.append(sum([motif_adj[i, j] for j_index, j in enumerate(
#                 j_m_nodes) if j_m_nodes_c[j_index] == j_m_c]))

#         if j_m_nodes_c.count(i_node_c) > 0 and j_m_c_counts.count(j_m_nodes_c.count(i_node_c)) > 1 and len(set(i_mc_wsum)) != len(i_mc_wsum):
#             print("#### {}节点关于邻域社区结构对称{} ###".format(i, i_node_c))
#             # 寻找i节点的点模体，获得该模体对i节点邻居社区的隶属度
#             im_set = motif_info['node_mdict'][i]  # 获得i所参与构成的模体
#             ms = np.asarray([motif_info['motif_dict'][im]
#                             for im in im_set], dtype=int)  # 节点i参与的模体编号列表mcno
#             # print("ms=",ms)
#             j_nodes_c_set = set(j_nodes_c)
#             j_cno_sum = {}
#             for j_cno in j_nodes_c_set:
#                 j_cno_nodes = [j for j_index, j in enumerate(
#                     j_nodes) if j_nodes_c[j_index] == j_cno]
#                 j_cno_sum[j_cno] = 0
#                 for mno in ms:
#                     mnodes = motif_info['mno_dict'][mno]
#                     wij = 0
#                     for no in mnodes:
#                         no_c = np.argmax(Xi[:, no])
#                         if no_c == j_cno:  # 该模体节点在jc社区中
#                             wij += motif_adj[i, no]
#                     j_cno_sum[j_cno] += bestMUs[j_cno, mno] * \
#                         wij/motif_info['mw'][mno]

#                 # j_cno_sum[j_cno] -= (len(ms)*Xi[j_cno,i])/3
#                 print("社区号{}：{}社区的模体隶属度为:{}".format(
#                     j_cno, j_cno_nodes, j_cno_sum[j_cno]))


# =============================================================================
# find_best： 寻找最优个体信息
# =============================================================================
def find_bestX(pop, fit, best_Q, real_mem, NMIflag):
    bestQ = int(best_Q*1000)/1000
    bestIndex,bestNMI = fit.index(best_Q),0
    # 当代最优值及最优个体
    fit_arr = np.asarray(fit)
    best_indexs = list(np.where(fit_arr>bestQ)[0])
    #选择NMI最高的个体
    if NMIflag==1:
        for best_index in best_indexs:
            membership_c = list(np.ravel(pop[:,best_index]))
            nmi=ig.compare_communities(real_mem, membership_c, method='nmi', remove_none=False) 
            if nmi > bestNMI:
                bestIndex,bestNMI = best_index,nmi
    bestx = copy.deepcopy(np.ravel(pop[:,bestIndex]))
    return (bestx, bestNMI, bestIndex)


def testXMemMatrix(motif_info, XMemMatrix, pop, fit, motif_adj, n, c, NP, W, m):
    keys_list = list(XMemMatrix.keys())
    # if len(keys_list)>0: print("**********keys_len={}*********".format(len(keys_list)))
    for N in keys_list:
        X = pop[:, N]
        Qw = fit_Q(motif_info, X, motif_adj, n, c, W, m, 'Q')
        if round(Qw, 6) != round(fit[N], 6):
            print("********* pop error! ***********")
        Xtemp = np.argmax(XMemMatrix[N], axis=0)
        Xtemp_fit = fit_Q(motif_info, Xtemp, motif_adj, n, c, W, m, 'Q')
        if round(Xtemp_fit, 6) != round(fit[N], 6):
            print("********* XMemMatrix error! ***********")

# =============================================================================
# 优化种群
# =============================================================================
def optimization_matrixV1(pop,NP):
    for N in range(NP):
        X = pop[:,N]
        XsetC_len = len(set(X))
        cno_list = [cno for cno in range(XsetC_len)]
        setX_dict = dict(zip(set(X), cno_list))
        for i, icno in enumerate(X): 
            pop[i,N]=setX_dict[icno]

# =============================================================================
# 隶属度矩阵优化
# =============================================================================
def optimization_matrix(pop, XMemMatrix, n, gen, X_XmemDict):
    Xs = XMemMatrix.keys()
    if gen%20==0 and len(Xs) > 0:
        for i in Xs:
            Xi = XMemMatrix[i]
            c = Xi.shape[0]
            membership_setc = set(np.argmax(Xi, axis=0))
            membership_setc_len = len(membership_setc)
            if membership_setc_len+10 > c:
                continue
            opt_Xi = np.zeros((membership_setc_len, n), dtype=float)
            for index, cno in enumerate(membership_setc):
                opt_Xi[index, :] = Xi[cno, :]
            XMemMatrix[i] = opt_Xi
            pop[:,i] = np.argmax(opt_Xi, axis=0)
            #初始化社区对应的社区号
            Xset = set(pop[:,i])
            cno_list = [cno for cno in range(len(Xset))]
            X_XmemDict[i] = dict(zip(Xset, cno_list))

# =============================================================================
# 多指标计算
# =============================================================================
# def MultiIndexCalculation(RITA, Gi, motif_info, bestx, real_mem, membership_c, motif_adj, n, c, W, m):
#     yyyy = list(filter(None, VertexClustering(Gi, membership_c)))
#     com_list_symbol = [0]*n
#     for i in range(len(yyyy)):
#         for j in yyyy[i]:
#             com_list_symbol[j] = i
#     Qw = fit_Q(motif_info, bestx, motif_adj, n, c, W, m, 'Q')
#     FM = 1-fit_Q(motif_info, bestx, motif_adj, n, c, W, m, 'MC')
#     nmi = ig.compare_communities(
#         real_mem, membership_c, method='nmi', remove_none=False)
#     ARI_score = metrics.adjusted_rand_score(real_mem, com_list_symbol)
#     # F1_score = f1_score(real_mem, com_list_symbol, average=None)
#     F1_score = f_measure(com_list_symbol, real_mem)
#     # Precision_score = precision_score(real_mem, com_list_symbol, average=None)[2]
#     # Recall_score = recall_score(real_mem, com_list_symbol, average=None)[2]
#     print("\n########### 多指标计算 ###########")
#     print("Qw={},FM={},NMI={},ARI={},F1_score={}".format(
#         Qw, FM, nmi, ARI_score, F1_score))
#     RITA['Qws'].append(Qw)
#     RITA['FMs'].append(FM)
#     RITA['nmis'].append(nmi)
#     RITA['ARIs'].append(ARI_score)
#     RITA['F1s'].append(F1_score)
#     # print("FM={},ARI={},F1={},Precision={},recall={}".format(1,ARI_score,F1_score,Precision_score,Recall_score))


# def f_measure(cluster, labels):
#     def get_set(data):
#         products = []
#         for i in itertools.product(data, data):
#             temp = list(i)
#             temp.sort()
#             products.append(tuple(temp))
#         products = [(i, j) for i, j in products if i != j]
#         return products
#     P, Q = [], []
#     for i in np.unique(labels):
#         setA = np.where(labels == i)[0]      #
#         P.extend(get_set(setA))

def choice_bestX_pop(pop_A, fits_A, pop_B, fits_B, NP):
    better_number = 0
    # 选择优秀个体并保留到种群
    for X_index in range(NP):
        if fits_B[X_index] > fits_A[X_index]:
            pop_A[:, X_index] = pop_B[:, X_index]  # 保存优秀个体
            fits_A[X_index] = fits_B[X_index]  # 保存优秀个体的适应度函数值
            better_number += 1
    return better_number
            



