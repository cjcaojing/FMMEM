# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:38:16 2020

@author: Administrator
"""
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

class Statistic:
    def __init__(self, graph, network_name, M, runs):
        self.graph = graph
        self.network_name = network_name
        self.M = M
        self.runs = runs
        self.motif_func_list = {"M1": self._m1, "M2": self._m2, "M3": self._m3, "M4": self._m4, 
                                "M5": self._m5, "M6": self._m6, "M7": self._m7, "M8": self._m8}
    
    def _m1(self, node_1, node_2):
        """
        Calculating the neighbourhood overlap for a pair of nodes.
        :param node_1: Source node 1.
        :param node_2: Source node 2.
        :return neighbourhood overlap: Overlap score.
        """
        nodes_1 = self.graph.neighbors(node_1)
        nodes_2 = self.graph.neighbors(node_2)
        return len(set(nodes_1).intersection(set(nodes_2)))
    def _m2(self, node_1, node_2):
        """
        Calculating the neighbourhood overlap for a pair of nodes.
        :param node_1: Source node 1.
        :param node_2: Source node 2.
        :return neighbourhood overlap: Overlap score.
        """
        u_friends = self.graph.neighbors(node_1)
        v_friends = self.graph.neighbors(node_2)
        u_friends=list(u_friends)
        v_friends=list(v_friends)
        if node_1 in v_friends:
            v_friends.remove(node_1)
        if node_2 in u_friends:
            u_friends.remove(node_2)
        u_mor = list(set(u_friends)-set(v_friends))
        v_mor = list(set(v_friends)-set(u_friends))
        num = len(u_mor) + len(v_mor)
        return num
    
    def _m3(self, node_1, node_2):    
        u,v = node_1,node_2
        edge_all=self.graph.edges()
        u_friends = self.graph.neighbors(u)
        v_friends = self.graph.neighbors(v)
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
        return deta2+deta1
    
    def _m4(self, node_1, node_2):
        m4_count = 0 #计数
        u,v = node_1,node_2
        edge_all=self.graph.edges()
        u_friends = self.graph.neighbors(u)
        v_friends = self.graph.neighbors(v)
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
            m4_count += 0
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
            m4_count+=deta
        return m4_count
    
    def _m5(self, node_1, node_2):
        m5_count = 0 #计数
        u,v = node_1,node_2
        edge_all=self.graph.edges()
        u_friends = self.graph.neighbors(u)
        v_friends = self.graph.neighbors(v)
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
            m5_count+=0
        else:
            #如果节点u的邻居节点除v外有多个，那么判断多个节点能构成几个全封闭三角形
            for i in itertools.combinations(u_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    u_list0.append((min_pq,max_pq))            
            deta1 = 0
            for p,q in u_list0:
                deta1 += 1
            m5_count+=deta1
        if len(v_mor) <= 1:
            m5_count+=0
        else:
            for i in itertools.combinations(v_mor,2):
                min_pq=min(i[0],i[1])
                max_pq=max(i[0],i[1])  
                if (min_pq,max_pq) in edge_all:
                    v_list0.append((min_pq,max_pq)) 
            deta2 = 0
            for p,q in v_list0:
                deta2 += 1
            m5_count+=deta2 
        return m5_count
    
    def _m6(self, node_1, node_2):
        m6_count = 0 #计数
        u,v = node_1,node_2
        edge_all=self.graph.edges()
        u_friends = self.graph.neighbors(u)
        v_friends = self.graph.neighbors(v)
        v_friends = list(v_friends)
        u_friends = list(u_friends)
        if (u_friends == []) or (v_friends == []):
            m6_count+=0
        else:
            cn = list(set(u_friends) & set(v_friends))
            if len(cn) <= 1:
                m6_count+=0
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
                m6_count+=deta
        return m6_count
    
    def _m7(self, node_1, node_2):
        m7_count = 0 #计数
        u,v = node_1,node_2
        edge_all=self.graph.edges()
        u_friends = self.graph.neighbors(u)
        v_friends = self.graph.neighbors(v)
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
            m7_count+=0
        else:
            for i in u_mor:
                for j in v_mor:
                    mor_list.append((i,j))
            deta = 0
            for p,q in mor_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    deta += 1
            m7_count+=deta
        return m7_count
    
    def _m8(self, node_1, node_2):
        u_friends = self.graph.neighbors(node_1)
        v_friends = self.graph.neighbors(node_2)
        edge_all=self.graph.edges()
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
            
    def _moti_num(self, all_edge_G):
        #计算M模体
        ij_participate_motif_number_list=[]
        for i,j in all_edge_G:
            ij_participate_motif_number=self.motif_func_list[self.M](i, j)        
            ij_participate_motif_number_list.append(ij_participate_motif_number)
              
        return ij_participate_motif_number_list                            
        
    
    def _read_test_conf(self, conf_path):
        f=open(conf_path,"r")
        test_config=eval(f.read())
        f.close()
        return test_config

    def fit(self):
        # 网络信息    
        network_name = self.network_name
        runs = self.runs
        G = nx.read_edgelist(r"../input/"+network_name + ".txt")
        G = G.to_undirected()
        n=G.number_of_nodes()
        #获取网络数据中的边列表，并根据其使用igraph创建网络
        Gi=ig.Graph.Read_Edgelist(r"../input/"+network_name+".txt")
        Gi=Gi.subgraph(map(int,G.nodes()))          
        Gi=Gi.as_undirected()
        ##################################################
        edge_all = Gi.get_edgelist()
        ij_participate_motif_number_list = self._moti_num(edge_all)
        
        #真实社区划分
        fo = pd.read_table(r"../input/" + network_name+"_groundtruth.txt",header = None)
        real=[]
        for i in fo[0]:
            real.append(i) 
        # print(real)
        
        Qws,nmis = [],[]
        for i in range(runs): 
            best_x_mem=self._read_test_conf("../output/" + network_name + '_membership_' + str(i) + '.json')
            partition1=[0]*n
            for key in best_x_mem.keys():
                partition1[int(key)] = best_x_mem[key]
            # best_x = list(best_x_mem.values())
            best_x=copy.deepcopy(list(partition1))
            # print(best_x)
            #计算QW
            best_x_Qw=ig.GraphBase.modularity(Gi,best_x,weights=ij_participate_motif_number_list)
            # if best_x_Qw < 0 :
            #     continue
            Qws.append(best_x_Qw)
            #计算NMI
            best_fit_x_nmi = ig.compare_communities(real,best_x,method='nmi',remove_none=False) 
            nmis.append(best_fit_x_nmi)
        print('QW_mean={},std={}, max={}'.format(mean(Qws), std(Qws), max(Qws)))
        print('NMI_mean={},std={}, max={}'.format(mean(nmis), std(nmis), max(nmis)))
        
