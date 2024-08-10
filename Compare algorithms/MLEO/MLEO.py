# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:14:55 2022

@author: Administrator
"""
import igraph as ig
import networkx as nx
import pandas as pd
import find_motifs as fm
import numpy as np
import time as time
# 引入外部函数
import motif_network_construct as net_stru_func
import random as rd

# =============================================================================
# 获得无向模体结构
# =============================================================================
def get_undirected_motif(motif):
    M_ins = nx.Graph()####模体按照Benson论文中设置
    if motif == "M1":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 2), (2,3),(3, 1)])  # 连通
    elif motif == "M2":
        M_ins.add_nodes_from([1, 2, 3])
        M_ins.add_edges_from([(1, 3),(3,2)])  # 连通
    elif motif == "M3":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 4),(2,4),(3,4)])  # 连通
    elif motif == "M4":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 3),(3,4),(4,2)])  # 连通
    elif motif == "M5":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 4),(2,3),(2,4),(3,4)])  # 连通
    elif motif == "M6":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 3),(1,4),(2,4),(2,3),(3,4)])  # 连通
    elif motif == "M7":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 2),(2,4),(4,3),(3,1)])  # 连通
    elif motif == "M8":
        M_ins.add_nodes_from([1, 2, 3,4])
        M_ins.add_edges_from([(1, 2),(1, 3),(1,4),(2,3),(2,4),(3,4)])  # 连通
    elif motif == "edge":
        M_ins.add_nodes_from([1, 2])
        M_ins.add_edges_from([[1, 2]])  # 连通
    return  M_ins

# =============================================================================
# 获得社区的邻居间节点
# =============================================================================
def local_C_nei(local_C, remain_m_nodes, motif_adj):
    nodes = local_C["nodes"]
    local_C["nei"] = []
    for i in nodes:
        j_m_nodes = set(np.nonzero(motif_adj[i,:])[1]) #获得i的模体邻居节点集
        out_j_m_nodes = (j_m_nodes - set(nodes)) & set(remain_m_nodes)
        if len(out_j_m_nodes) > 0:
            local_C["nei"] = list(set(local_C["nei"]) | out_j_m_nodes)
            
# =============================================================================
# RM_C计算
# =============================================================================
def RM_C(local_C, edge_mdict, motif_adj, aerfa):
    motif_num,nodes = local_C["motif_num"],local_C["nodes"]
    Nin_C = motif_num #社区内模体数量
    Nout_C_set = set() #初始化社区外模体集合
    out_medges = [] #社区间模体加权边
    
    for i in nodes:
        j_m_nodes = set(np.nonzero(motif_adj[i,:])[1])
        out_j_m_nodes = j_m_nodes - (j_m_nodes & set(nodes))
        for out_j in out_j_m_nodes: 
            out_medges.append((i,out_j)) if i < out_j else out_medges.append((out_j,i))

    for out_edge in out_medges:
        for edge_motif in edge_mdict[out_edge]:
            Nout_C_set.add(edge_motif)
    Nout_C = len(Nout_C_set) #社区外的模体数量
    rm_c = Nin_C/(Nin_C + Nout_C)**aerfa #计算RM_C
    return rm_c, Nin_C, (Nin_C + Nout_C)

# =============================================================================
# RM_C_Ai 计算
# =============================================================================
def RM_C_Ai(local_C, Ai, node_mdict,aerfa):
    RM_C_fenzi,RM_C_fenmu,nodes = local_C["RM_C"]["fenzi"], local_C["RM_C"]["fenmu"],local_C["nodes"]
    Ein_Ai,Eout_Ai = 0,0
    # 求Ein_Ai 和 Eout_Ai
    Ein_m_set,Eout_m_set = [],[]
    for i in Ai:
        i_motifs = node_mdict[i] #i的所有点模体
        for im in i_motifs:
            im_tmp = set([ii for ii in im if ii!=i])
            if len(im_tmp & set(nodes)) == 0: 
                Eout_m_set.append(im)
            elif len(im_tmp & set(nodes)) == len(im_tmp):
                Ein_m_set.append(im)

    Ein_Ai,Eout_Ai = len(Ein_m_set),len(Eout_m_set) #获得Ein_Ai,Eout_Ai值
    fenzi,fenmu = (RM_C_fenzi + Ein_Ai),(RM_C_fenmu + Eout_Ai)
    rm_c_ai = fenzi / fenmu**aerfa #计算RM_C_Ai值
    rm_ai = rm_c_ai - local_C["RM_C"]["rm_c"]
    return rm_ai,fenzi,fenmu

# =============================================================================
# RM_C_V 计算
# =============================================================================
def RM_C_V0(local_C, V0, node_mdict,aerfa):
    RM_C_fenzi,RM_C_fenmu,nodes = local_C["RM_C"]["fenzi"], local_C["RM_C"]["fenmu"],local_C["nodes"]
    # 求Ein_Ai 和 Eout_Ai
    Ein_m_set,Eout_m_set = [],[]
    V0_motifs = node_mdict[V0] #V0的所有点模体
    for im in V0_motifs:
        im_tmp = set([ii for ii in im if ii!=V0])
        if len(im_tmp & set(nodes)) == 0: 
            Eout_m_set.append(im)
        elif len(im_tmp & set(nodes)) == len(im_tmp):
            Ein_m_set.append(im)

    Ein_VO,Eout_VO = len(Ein_m_set),len(Eout_m_set) #获得Ein_V0,Eout_V0值
    fenzi_v0, fenmu_v0 = RM_C_fenzi - Ein_VO, RM_C_fenmu - Eout_VO
    rm_c_v0 = fenzi_v0 / fenmu_v0**aerfa #计算RM_C_V0值=RM_C
    rm_cv0 = rm_c_v0 - local_C["RM_C"]["rm_c"]
    return rm_cv0,rm_c_v0,fenzi_v0,fenmu_v0

# =============================================================================
# 检测阶段
# =============================================================================
def checkPhase(local_C, remain_m_nodes, motif_adj, aerfa):
    result = False #初始化数据
    for v in local_C["nodes"]:
        if len(local_C["nodes"])==1: return result
        RM_C_v,rm_c,fenzi_V,fenmu_V = RM_C_V0(local_C, v, node_mdict,aerfa)
        if RM_C_v > 0:
            local_C["nodes"].remove(v) #从社区C中移除节点v
            remain_m_nodes.append(v) #剩余节点集中添加v节点
            local_C["RM_C"]["rm_c"],local_C["RM_C"]["fenzi"],local_C["RM_C"]["fenmu"] = rm_c,fenzi_V,fenmu_V
            #更新local_C的模体邻居节点集
            local_C_nei(local_C, remain_m_nodes, motif_adj)
            result = True
            
    return result

# =============================================================================
# 社区扩展
# =============================================================================
def expansionPhase(M_N, local_C, remain_m_nodes, motif_adj, aerfa, belta):
    i, j = 1, 0 #初始化数据
    while i<M_N and len(local_C["nei"])>0:   #社区扩展条件
        # print("k=",i)
        R, A, fenzi,fenmu = 0, set(),0,0  #初始化R,A
        for v0 in local_C["nei"]:
            v0_Ai_set = set()
            v0_Ai_set.add(v0)
            v0_m_jnodes = list(set(np.nonzero(motif_adj[v0,:])[1]) & set(local_C["nei"])) #节点v0的模体邻居节点集
            if len(v0_m_jnodes)>0: 
                # rd.shuffle(v0_m_jnodes)
                v0_Ai_set = v0_Ai_set | set(v0_m_jnodes)
            if i > len(v0_Ai_set): continue
            Ai  = set(list(v0_Ai_set)[:i])
            RM_CAi,fenzi,fenmu = RM_C_Ai(local_C,Ai,node_mdict,aerfa)
            if RM_CAi > R: 
                R,A,fenzi,fenmu = RM_CAi,Ai,fenzi,fenmu #保留最优值和集合
        if R<=0: 
            i+=1
        else:
            local_C["nodes"] = list(set(local_C["nodes"])|A)
            for A_i in A: remain_m_nodes.remove(A_i)
            local_C["RM_C"]["rm_c"],local_C["RM_C"]["fenzi"],local_C["RM_C"]["fenmu"] = fenzi / fenmu**aerfa,fenzi,fenmu
            #更新local_C的模体邻居节点集
            local_C_nei(local_C, remain_m_nodes, motif_adj)
            i,j = 1,j+1
            if j % belta ==0:
                check_result = checkPhase(local_C, remain_m_nodes, motif_adj, aerfa)

# 选择网络
network_name = "email"
# 获取网络数据中的边列表，并根据其使用igraph创建网络
G1 = nx.read_edgelist("data/realnetwork/" + network_name + '.txt', create_using=nx.Graph())
G1 = G1.to_undirected()
node_list = list(G1.nodes) #点集
edge_list = list(G1.edges) #边集

Gi = ig.Graph.Read_Edgelist("data/realnetwork/" + network_name + '.txt')
Gi = Gi.subgraph(map(int,G1.nodes()))          
Gi = Gi.as_undirected()
edge_all = Gi.get_edgelist()


# 各参数设置
belta = 1
aerfa = 1.0
# =============================================================================
n=G1.number_of_nodes()
# 各标记列表
Mlist = {1:"M1",2:"M2",3:"M3",4:"M4",5:"M5",6:"M6",7:"M7",8:"M8"} #模体选择列表
# 本次算法使用的标记
M_flag = Mlist[1]
nmi_flag = 0 #0:关，1：开
runs = 1 #独立运行次数
# ================================================== ===========================
# 构建基于模体M的加权网络
# =============================================================================
# 获得无权网络邻接矩阵
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
# =============================================================================
# 获得基于模体M1的，每条边参与构建的模体集合(点集与边集)
# =============================================================================
g = get_undirected_motif(M_flag)
M_N = nx.DiGraph.number_of_nodes(g)#模体阶数
node_motif_num_list=[] #####点模体度列表#####
for i in range(n):
    node_motif_num_list.append(fm.node_in_motif(G,g,i,directed=False,weighted=False))
node_motif_num_arr = np.asarray(node_motif_num_list)
# 获得模体集合：点集和边集
motif_node_set, motif_edge_set = fm.total_motif_list(G, g, directed=False, weighted=False)
# motif_node_set.reverse(), motif_edge_set.reverse()
motif_node_set = sorted(motif_node_set,key=lambda x:(x))
motif_num = len(motif_node_set) #网络中的模体数量
node_mdict,edge_mdict = dict(),dict()  #点模体集合,边模体集合
# 获得点模体集合
for i in range(n):
    NodeSet,EdgeSet = fm.node_in_motif_list(G, g, i , directed=False, weighted=False)
    node_mdict[i] = [tuple(sorted(im)) for im in NodeSet]
# 获得边模体集合
for edge in edge_list:
    NodeSet1, EdgeSet1 = fm.edge_in_motif_list(G1, g, edge, directed=False, weighted=False) 
    edgel = sorted([int(edge[0]),int(edge[1])])
    edge_mdict[tuple(edgel)] = [tuple(sorted(list(map(int,im)))) for im in NodeSet1]

if nmi_flag==1:
    # 获得真实社区划分
    fo = pd.read_table(r"data/realnetwork/real/" + network_name + "_groundtruth.txt",header = None)
    real_membership =[]
    for i in fo[0]:
        real_membership.append(i) 

#初始化数据
Qws,nmis = [],[]
for run in range(runs):
    print("####### {} 第{}次独立运行开始 ##########".format(network_name,run+1))
    run_start_time = time.process_time()
    partition = [-1]*n #初始化社区分区
    Cs = []
    remain_nodes = list(map(int,node_list)) #剩余节点集合(高阶+低阶)
    remain_m_nodes = [i for i in remain_nodes if node_motif_num_list[i]>0] #参与模体组件的高阶点
    remain_e_nodes = set(remain_nodes) - set(remain_m_nodes) #未参与模体组件的低阶点
    
    Cs_index = 0
    while len(remain_m_nodes) != 0:
        print("#######开始获得新的局部社区##########")
        #初始化本次局本社区信息
        local_C = {} #本次的局部社区初始化
        RM_C_info = {}
        RM_C_info["rm_c"], RM_C_info["fenzi"], RM_C_info["fenmu"] = 0.0,0,1
        local_C["motifs"], local_C["motif_num"],local_C["nodes"],local_C["nei"],local_C["jian_nodes"],local_C["RM_C"]= [],0,[],[],[],RM_C_info #motifs:模体集合 nodes:节点集合,局部社区模体邻居节点集合
        
        # print(gen,partition)
        remain_node_motif_num_list = node_motif_num_arr[remain_m_nodes]
        remain_node_motif_max_num = remain_node_motif_num_list.max()
        #种子节点选取
        seed_nodes = [remain_m_nodes[i_index] for i_index, num in enumerate(remain_node_motif_num_list) if num == remain_node_motif_max_num]
        seed_node = rd.choice(seed_nodes) #从待选节点中随机选择一个作为种子节点
        #核心种子社区获取
        seed_nodes = list(np.nonzero(motif_adj[seed_node,:])[1])#种子社区中的节点集
        seed_nodes = [i for i in seed_nodes if i in remain_m_nodes]
        seed_nodes.append(seed_node) #种子社区中的节点集
        # 获得种子社区内的所有模体
        motifs_set = set()
        for seed_i in seed_nodes: motifs_set = motifs_set | set(node_mdict[seed_i])
        
        seed_C_motifs = [motif for motif in motifs_set if len(set(motif)-set(seed_nodes))==0] #种子社区所有的模体
        seed_C_motif_num = len(seed_C_motifs) #种子社区中的模体数量
        remain_m_nodes = list(set(remain_m_nodes) - set(seed_nodes)) #更新剩余节点集
        local_C["motifs"],local_C["motif_num"],local_C["nodes"] = seed_C_motifs,seed_C_motif_num,seed_nodes #更新局部社区信息
        local_C_nei(local_C, remain_m_nodes, motif_adj) #获得该社区的邻居节点集合
        # 计算RM_C信息
        local_C["RM_C"]["rm_c"],local_C["RM_C"]["fenzi"],local_C["RM_C"]["fenmu"] = RM_C(local_C, edge_mdict, motif_adj, aerfa)
        #社区节点点检查阶段
        checkPhase(local_C, remain_m_nodes, motif_adj, aerfa)
        # 社区扩展阶段
        expansionPhase(M_N, local_C, remain_m_nodes, motif_adj, aerfa, belta)
        Cs.append(local_C["nodes"])
        print("####Cs_index={}####localC_len={}########".format(Cs_index, len(Cs[Cs_index])))
        Cs_index+=1
        
    # 还原节点社区号
    for C_index, Cnodes in enumerate(Cs):
        for node in Cnodes: partition[node] = C_index
    
    # =============================================================================
    # 低阶补全
    # =============================================================================
    Cno = len(Cs)
    for ei in remain_e_nodes:
        j_nodes = np.nonzero(adj[ei,:])[1] #ei的邻居节点
        # 获得节点 ei_j_nodes 所在的社区
        j_nodes_c = list(np.asarray(partition)[j_nodes])
        while j_nodes_c.count(-1)>0: 
            j_nodes_c.remove(-1)
        if len(j_nodes_c)==0: 
            partition[ei] = Cno
            Cno+=1
        else:
            ei_cno = rd.choice(list(j_nodes_c)) #依概率选择社区
            partition[ei] = ei_cno
    
    # test_data = [i_index for i_index, i in enumerate(partition) if i==-1]
    # for test in test_data:
    #     if test not in remain_e_nodes:
    #         print("M_i=",test)
    # =============================================================================
    # 计算结果
    # =============================================================================
    best_x_Qw=ig.GraphBase.modularity(Gi,partition,weights=ij_participate_motif_number_list)
    Qws.append(best_x_Qw)
    if nmi_flag==1:
        best_fit_x_nmi = ig.compare_communities(real_membership,partition,method='nmi',remove_none=False) 
        nmis.append(best_fit_x_nmi)
        print("NMI=", best_fit_x_nmi)
    print("QW=", best_x_Qw)
    # print("partition=", partition)
    run_end_time = time.process_time()
    run_spend_time = round((run_end_time - run_start_time)/60,3) #本次运行花费时长
    print("####### {} 第{}次独立运行结束！耗时{}分钟 ##########".format(network_name,run+1, run_spend_time))

# =============================================================================
# 输出结果
# =============================================================================
print("###### {} ### {} ####".format(network_name,M_flag))       
print("Cno=",Cno) 
print('QW_mean={},std={}, max={}'.format(round(np.mean(Qws),3), np.std(Qws), max(Qws)))
if nmi_flag==1:
    print('NMI_mean={},std={}, max={}'.format(round(np.mean(nmis),3), np.std(nmis), max(nmis)))
print("\nmembership_c=",partition)
