# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022

@author: WYW
"""
"""
    HCD-MF_main
   基于模体及模糊信息的高阶社区检测
"""
import numpy as np
import igraph as ig
import random  as rd
import networkx as nx
import copy
import os
import time
from tqdm import tqdm
import logging
import MyLogging as myLog

# 各模块函数
import motif_network_construct as net_stru_func
import algorithm_FCD_function as alg_func

import FMMEM_function as func
# import FMMEM_function_NOF as func
# import FMMEM_function_NOMerge as func
# import FMMEM_function_NOFMerge as func
# 引入外部函数
import find_motifs as fm
import cython_function as cfunc


# =============================================================================
# 网络信息 
# network
# =============================================================================
## 真实网络
path = r"data/realnetwork"
# 人工合成网络
LFR_path = r"data/LFRnetwork/LFR_1000"
# test_path = r"data/test"
network50 =  r'data/LFRnetwork/LFR_100_20_25/'

## 真实网络
karate_network = path + r'/karate.txt'
dolphins_network = path + r'/dolphins.txt'
polbooks_network = path + r'/polbooks.txt'
football_network = path + r'/football.txt'
jazz_network = path + r'/jazz.txt'
metabolic_network = path + r'/metabolic.txt'
netscience_network = path + r'/netscience.txt'
email_network = path + r'/email.txt'
email_Eu_core_network = path + r'/email_Eu_core.txt'
road_minnesota_network = path + r'/road_minnesota.txt'
cornell_network = path + r'/cornell.txt'
cora_network = path + r'/cora.txt'
citeseer_network = path + r'/citeseer.txt'
powergrid_network = path + r'/powergrid.txt'
powergrid_0_network = path + r'/powergrid_0.txt'
pgp_network = path + r'/pgp.txt'
comDBLP_network = path + r'/comDBLP.txt'   
polblogs_network = path + r'/polblogs.txt'
gplus_network = path + r'/gplus.txt'

benson_network = path + r'/benson.txt'
beican_9_network = path + r'/9_beican.txt'
test12_network = path + r'/test_12.txt'
zhang_network = path + r'/zhang.txt'
lesmis_network = path + r'/lesmis.txt'

## 功能网络
macaque_network = path + r'/macaque.txt'

# 选择网络
# real
network_name = 'gplus'
network = eval(network_name + '_network')
groundtruth_path = path + "/real/" + network_name + '_groundtruth.txt'

# LFR 
#lfr_no = '91'
#network = LFR_path + r'/network' + str(lfr_no)+'.txt'
#network_name = "network" + str(lfr_no)
#groundtruth_path = LFR_path + "/real/" + network_name + '_groundtruth.txt'

    
# test
# network = network50 + 'network50.txt' 
# network_name = 'network50'
# groundtruth_path = network + "/real/" + network_name + '_groundtruth.txt'

#初始化log
log = myLog.create_log(name="ros_log",level=logging.DEBUG,filename="log/FFA/HCD/"+ network_name +"_1.log",sh_level=logging.DEBUG,fh_level=logging.DEBUG)


# 获取网络数据中的边列表，并根据其使用igraph创建网络
G1 = nx.read_edgelist(network, create_using=nx.Graph())
G1 = G1.to_undirected()
Gi = ig.Graph.Read_Edgelist(network)
Gi = Gi.subgraph(map(int,G1.nodes()))          
Gi = Gi.as_undirected()

edge_all = Gi.get_edgelist()
for e in edge_all: #剔除自环边
    if e[0]==e[1]: edge_all.remove(e)
node_list = list(G1.nodes)
node_all = list(map(int,node_list))

# 各参数设置
# =============================================================================
n=G1.number_of_nodes()
NP = 100
Gen = 2000  #进化代数
lu_bound = [0,n-1] #元素取值范围
###### SOS算法参数 #########
SOS_info = {}
######DE算法参数###########
F = 0.9
CR = 0.3
D = n
DE_info = {}
DE_info['F'],DE_info['CR'],DE_info['D'] = F,CR,D
######### WOA算法参数 #########
a = 2.0 #控制搜索排列默认值
a_step = a/Gen #鲸鱼群包围搜索步长
b = 1 #常数，用于定义对数螺线的形状
WOA_info={}
WOA_info['a'],WOA_info['a_step'],WOA_info['b'] = a,a_step,b

######### GA算法参数 #########
REC_F = 'xovud' #均匀分布交叉
MUT_F = 'mutde' #变异算子 mutde：差分变异算子，mutswap：染色体两点互换变异算子
pc = 0.8
GA_info={}
GA_info['REC_F'],GA_info['MUT_F'],GA_info['pc'] = REC_F,MUT_F,pc

######### FFA算法参数 #########
FF_k = 4  #FF_k:农田分块数,通常取值范围为[2,8]
FF_n = NP//FF_k #FF_n:每块儿农田中解决方案的数量(个体数)
FF_alpha,FF_belta,FF_omiga = 0.6,0.4,1 #一般参数设置
FF_Q = 0.7 #用于确定最佳全局解的组合量，取值范围(0,1)
FFA_info={}
FFA_info['FF_k'],FFA_info['FF_n'],FFA_info['FF_alpha'],FFA_info['FF_belta'],FFA_info['FF_omiga'],FFA_info['FF_Q'] = FF_k,FF_n,FF_alpha,FF_belta,FF_omiga,FF_Q

# 各标记列表
Mlist = {1:"M1",2:"M2",3:"M3",4:"M4",5:"M5",6:"M6",7:"M7",8:"M8"} #模体选择列表
Qlist = {1:"Q",2:"Qg",3:"Qc_FCD",4:"Qc_OCD",5:"Qov",6:"MC"} # 模块度函数列表
OAlist = {"SOS":alg_func.SOS, "DE":alg_func.DE, "WOA":alg_func.WOA, "GA":alg_func.GA, "FFA":alg_func.FFA, "ESOS":alg_func.ESOS,} # 优化算法列表

# =============================================================================
# # 本次算法使用的参数配置
# =============================================================================
M_flag = Mlist[1] #模体选择 M1
Q_flag = Qlist[1] #模块度函数 Qw
OA = OAlist['FFA'] #优化策略选用
OA_info = FFA_info #
# 独立运行运行次数
Independent_Runs = 10 #本次实验独立运行次数
NMIflag = 0 #0:关闭NMI，1:开启NMI。 默认关闭
 
# =============================================================================
# 构建基于模体M的加权网络
# =============================================================================
# 获得无权网络邻接矩阵
G2 = nx.Graph() 
G2.add_nodes_from([i for i in range(n)])
G2.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G2)
adj=adj.todense() 
# =============================================================================
# 网络平均度计算
# =============================================================================
degrees = dict(G2.degree())
degrees_sum = sum(degrees.values())
average_degrees = round(degrees_sum/n,2)

G,ij_participate_motif_number_list = net_stru_func.construct_weighted_network(G2,edge_all,n,M_flag) #构建出基于M_flag模体加权的网络
del G2
# 构建基于模体的加权网络邻接矩阵motif_adj
motif_adj = nx.adjacency_matrix(G)
motif_adj = motif_adj.todense() 
me_adj = motif_adj + adj #总权重矩阵=模体权重+边权重
W = np.sum(motif_adj) # 权值之和
m1 = np.sum(motif_adj, axis=0)
m = np.ravel(m1) # adj 各列之和

# 初始化模体各节点的边邻域节点集、模体邻居节点集、点模体邻居节点集
node_nei_info, node_adj_neis, node_moadj_neis, node_meadj_neis = dict(),dict(),dict(),dict()
node_moadj_sum, node_meadj_sum = [0]*n, [0]*n
for i in range(n):
    node_adj_neis[i] = np.nonzero(adj[i,:])[1]
    node_moadj_neis[i] = np.nonzero(motif_adj[i,:])[1]
    node_meadj_neis[i] = np.nonzero(me_adj[i,:])[1]
    node_moadj_sum[i] = np.sum(motif_adj[i,:])
    node_meadj_sum[i] = np.sum(me_adj[i,:])
node_nei_info["adj"], node_nei_info["motif_adj"], node_nei_info["me_adj"] = node_adj_neis,node_moadj_neis,node_meadj_neis
node_nei_info["motif_adj_sum"], node_nei_info["me_adj_sum"] = node_moadj_sum, node_meadj_sum

# =============================================================================
# 获得基于模体M1的，每条边参与构建的模体集合(点集与边集)
# =============================================================================
g = func.get_undirected_motif(M_flag)
D = nx.DiGraph.number_of_nodes(g) #模体阶数
# 计算该网络的模体点、边覆盖率
node_coverage_rate = round(fm.node_coverage_rate_of_motif(G1,g,directed=False, weighted=False),2)
edge_coverage_rate = round(fm.edge_coverage_rate_of_motif(G1,g,directed=False, weighted=False),2)

node_motif_num_list=[]#####点模体度列表#####
for i in range(n):
    node_motif_num_list.append(fm.node_in_motif(G,g,i,directed=False,weighted=False))
node_motif_num_arr = np.asarray(node_motif_num_list,dtype=int)
# 获得模体集合：点集和边集
motif_node_set, motif_edge_set = fm.total_motif_list(G, g, directed=False, weighted=False)
# motif_node_set.reverse(), motif_edge_set.reverse()
del motif_edge_set
motif_node_set = sorted(motif_node_set,key=lambda x:(x))

r = len(motif_node_set) #网络中的模体数量
motif_dict,mno_dict,node_mdict,edge_mdict,motif_info = dict(),dict(),dict(),dict(),dict() # 初始化模体字典和模体编号字典
mno_mw_arr = np.zeros((r,1), dtype = int)
for mno, mo in enumerate(motif_node_set):
    motif_dict[tuple(mo)] = mno
    mno_dict[mno] = tuple(mo)
    mno_mw_arr[mno] = motif_adj[mo[0],mo[1]] + motif_adj[mo[1],mo[2]] + motif_adj[mo[0],mo[2]]
    
mno_arr = np.empty((r,D), dtype = int)
for mno, mo in enumerate(motif_node_set):
    for i in range(D):
        mno_arr[mno,i] = mo[i]

# 获得点模体集合，构建集合字典
for i in range(n):
    NodeSet,EdgeSet = fm.node_in_motif_list(G, g, i , directed=False, weighted=False)
    node_mdict[i] = [tuple(sorted(im)) for im in NodeSet]
# # 获得边模体集合
# for edge in edge_all:
#     NodeSet1, EdgeSet1 = fm.edge_in_motif_list(G, g, edge, directed=False, weighted=False) 
#     edgel = sorted([int(edge[0]),int(edge[1])])
#     edge_mdict[tuple(edgel)] = [tuple(sorted(list(map(int,im)))) for im in NodeSet1]

#网络中的模体信息，(模体数目，模体字典，模体编号字典,点模体集合字典)
motif_info['r'], motif_info['motif_dict'], motif_info['mno_arr'], motif_info['mno_dict'],motif_info['node_mdict'],motif_info['node_mnum'] = r,motif_dict,mno_arr,mno_dict,node_mdict,node_motif_num_arr
motif_info['mw'] = mno_mw_arr.flatten()
# 初始化NMi
nmilist = [] # 用于保存每一代的NMI值
# 获取真实社区划分列表
file_flag = os.path.exists(groundtruth_path)
if file_flag:
    NMIflag,real_mem = 1,[]
    with open(groundtruth_path, mode='r',encoding='UTF-8') as f:
          real_mem = list(map(int,f.read().splitlines()))

print("real_C: ",len(set(real_mem)))
# =============================================================================
# 计算网络种节点之间的紧密度
# =============================================================================
Compactness = np.zeros((n,n), dtype = float)
for i in range(n):
    for j in range(n):
        Compactness[i,j] = (motif_adj[i,j] - (m[i]*m[j])/W)/W
        
MB = np.zeros((n,n), dtype = float)
MBW = np.sum(me_adj) # 权值之和
MBm = np.ravel(np.sum(me_adj, axis=0))
for i in range(n):
    for j in range(n):
        if i!=j:
            MB[i,j] = me_adj[i,j] - (MBm[i]*MBm[j])/MBW
del G
# =============================================================================
# 程序运行
# =============================================================================
log.info("===net:{0}==n:{1}==node_rate:{2}==e:{3}==edge_rate:{4}=={5}:{6}==ave_degree:{7}===".format(network_name,n,node_coverage_rate,len(edge_all),edge_coverage_rate,M_flag,r,average_degrees))
del degrees
#初始化数据
Qws,Maxnmis,nmis = [],[],[]
# RITA = {}
# RITA['Qws'],RITA['nmis'],RITA['ARIs'],RITA['F1s'],RITA['FMs'] = [],[],[],[],[]
run = 0 # 本程序开始独立运行的次数
while (run < Independent_Runs):
    # =============================================================================
    # 全局变量设置
    # =============================================================================
    stop_flag = False #程序停止标识
    c = 0 
    # 全局变量设置
    best_in_history_Q = [] #用于保存历史最优Q值
    best_in_history_NMI = [] #用于保存历史NMI值
    # 用于保存到文件中
    Qs_history_FNCM_dict = {}
    NMIs_history_FNCM_dict = {}
    
    # =============================================================================
    # 种群初始化，有偏操作
    # =============================================================================
    #种群初始化
    pop = func.init_pop_V1(n,NP)  #初始化种群
    fit = []
    func.fit_Qs(Compactness,motif_info, fit,pop,motif_adj,n,c,NP, W, m,Q_flag)   #适应度函数值计算 

    #有偏操作                                                                                                                                                                   
    bias_pop = func.bias_init_pop_V1(pop, n, NP, adj) #对初始化后的种群进行有偏操作
    bias_fit_values,betterBias_number = [],0
    func.fit_Qs(Compactness,motif_info, bias_fit_values,bias_pop,motif_adj,n,c,NP,W, m,Q_flag) #适应度函数值计算 
    # 选择优秀个体并保留到种群
    betterBias_number = func.choice_bestX_pop(pop,fit,bias_pop,bias_fit_values,NP)
    del bias_pop
    # =============================================================================
    # Main
    #【使用优化算法进行社区检测】
    # =============================================================================
    startAll = time.process_time()
    betternmm_number_E, betternmm_number_FNCM, betterLSCM_number, LSCM_Flag = 0, 0, 0, False
    # 用于记录各个个体的社区数目及状态
    XS_info = {}
    for N in range(NP): 
        XS_info[N] = [0,0,False,False]  #XS_info[0]:当前个体的社区数目，XS_info[1]:当前个体的社区数目持续迭代次数，XS_info[2]:社区数是否收敛，XS_info[3]:当前社区数目是否收敛
    for gen in tqdm(range(Gen)):
    # for gen in range(Gen):
        # 优化算法
        (E_pop, E_fit, betternmmE_number) = OA(OA_info,Compactness,motif_info,lu_bound,pop,fit,n,NP,motif_adj,W,m,Q_flag)
        betternmm_number_E += betternmmE_number 
        
        # E_pop, E_fit = pop, fit
        # NMM操作
        # NCM_start = time.process_time()
        # (FNCM_pop, FNCM_fit, betternmm_number, LSCM_Num) = func.FNCM_funcs(MB,Compactness,motif_info,node_nei_info,E_pop,E_fit,n,NP,motif_adj,me_adj,Q_flag,W,m,LSCM_Flag)
        # betternmm_number_FNCM += betternmm_number
        # betterLSCM_number+=LSCM_Num
        # NCM_end = time.process_time()
        # log.info("\nNCM_spendTime=",NCM_end-NCM_start)
        
        FNCM_pop, FNCM_fit = E_pop, E_fit
        # 节点隶属度矩阵优化
        # func.optimization_matrixV1(FNCM_pop,NP)
                
        best_Q = max(FNCM_fit) #最优值
        best_index = FNCM_fit.index(best_Q)
        bestx = FNCM_pop[:,best_index]
        if gen>500 and (((abs(abs(best_in_history_Q[gen-70]) - best_Q) <= 1.0e-7) 
                          or (best_in_history_Q[gen-50] == best_Q))) or (gen+1) % Gen ==0:
            stop_flag=True
            log.info("\n****the last deidaicishu={}*****\n".format(best_in_history_Q.index(best_Q)))        
            
        membership_c = list(bestx) 
        c_len = len(set(membership_c))
        # 个体Xbest，并记录最优个体对应的Q值及NMI
        best_in_history_Q.append(best_Q)
        # LSCM_Flag=True if (gen>150 and gen%10==0 and best_in_history_Q[gen-20]==best_Q) else False
            
        if NMIflag==1:
            nmi=ig.compare_communities(real_mem, membership_c, method='nmi', remove_none=False)    
            best_in_history_NMI.append(nmi)
        
        # 更新pop,fit
        pop = FNCM_pop
        fit = FNCM_fit
        
        if (gen+1) % 100 ==0 or stop_flag:
            # 计算Q，Qg,Qc,QW
            log.info("#####"+ M_flag +"_FMMEM_NO__MaxCno" + Q_flag + "_#####")
            log.info("gen={}".format(gen+1))
            log.info("c_count={}".format(c_len))
            # log.info("max_cum={}".format(max_cum))
            if NMIflag==1:
                log.info("MAXNMI={},NMI={}".format(max(best_in_history_NMI),nmi))
            best_Q_result = (1 - best_Q) if Q_flag == "MC" else best_Q
            log.info("best_"+ Q_flag + "={}, bestN={}".format(best_Q_result,best_index))
            log.info("betternmm_E={}, betternmm_FNCM={}, betterLSCM={}".format(betternmm_number_E, betternmm_number_FNCM, betterLSCM_number))
            betternmm_number_E,betternmm_number_NMM, betterMeger_number = 0, 0, 0 
            # M_Qw=ig.GraphBase.modularity(Gi,membership_c,ij_participate_motif_number_list) 
            # F_QW = func.fit_Q(Compactness,motif_info,bestx,motif_adj,n,len(set(bestx)),W,m,Q_flag)
            # F_QW = (1 - F_QW) if Q_flag == "MC" else F_QW
            # log.info("M_Qw={},F_QW={}".format(M_Qw,F_QW)) 
            endAll = time.process_time()
            log.info("spendTime={}s".format(int(endAll-startAll)))
            startAll=endAll

        if stop_flag: 
            best_Q_result = (1 - best_Q) if Q_flag == "MC" else best_Q
            Qs_history_FNCM_dict[Q_flag] = best_Q_result #保存最优Q值
            if NMIflag==1:
                NMIs_history_FNCM_dict[Q_flag] = max(best_in_history_NMI) #保存最优nmi值            
            # 寻找关于邻域结构对称的节点
            # func.find_target_nodes(node_nei_info,motif_info,bestx,node_all,bestMUs,motif_adj)
            Qws.append(best_Q)
            if NMIflag==1:  
                Maxnmis.append(max(best_in_history_NMI))
                nmis.append(nmi)
                # 多指标
                # func.MultiIndexCalculation(RITA,Gi,motif_info,bestx,real_mem,membership_c,motif_adj,n,c,W,m)
            log.info(membership_c)
            break
    run+=1

# =============================================================================
# 输出结果
# =============================================================================
log.info("\n## FMMEM_NOMerge #### {} ### {} ####".format(network_name,M_flag))        
log.info('QW_mean={},std={}, max={}'.format(round(np.mean(Qws),6), np.std(Qws), max(Qws)))
if NMIflag==1:
    log.info('MAXNMI_mean={},std={}, max={}'.format(round(np.mean(Maxnmis),6), np.std(Maxnmis), max(Maxnmis)))
    log.info('NMI_mean={},std={}, max={}'.format(round(np.mean(nmis),6), np.std(nmis), max(nmis)))
    # log.info("\n########### 多指标计算 ###########")
    # log.info("Qw={},FM={},NMI={},ARI={},F1_score={}".format(round(np.mean(RITA['Qws']),6),round(np.mean(RITA['FMs']),6),round(np.mean(RITA['nmis']),6),round(np.mean(RITA['ARIs']),6),round(np.mean(RITA['F1s']),6))) 


 