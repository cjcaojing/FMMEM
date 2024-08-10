""" EdMot clustering class."""

import community
import networkx as nx
from tqdm import tqdm
import itertools
import copy as copy
import numpy as np

class EdMot(object):
    """
    Edge Motif Clustering Class.
    """
    def __init__(self, graph, component_count, cutoff, M):
        """
        :param graph: NetworkX object.
        :param component_count: Number of extract motif hypergraph components.
        :param cutoff: Motif edge cut-off value.
        """
        self.graph = graph
        self.component_count = component_count
        self.cutoff = cutoff
        self.M = M
        self.motif_func_list = {"M1": self._m1_motifadd, "M2": self._m2_motifadd, "M3": self._m3_motifadd, "M4": self._m4_motifadd, 
                        "M5": self._m5_motifadd, "M6": self._m6_motifadd, "M7": self._m7_motifadd, "M8": self._m8_motifadd}

    def _overlap(self, node_1, node_2):
        """
        Calculating the neighbourhood overlap for a pair of nodes.
        :param node_1: Source node 1.
        :param node_2: Source node 2.
        :return neighbourhood overlap: Overlap score.
        """
        nodes_1 = self.graph.neighbors(node_1)
        nodes_2 = self.graph.neighbors(node_2)
        return len(set(nodes_1).intersection(set(nodes_2)))
    def _m1(self,G,edge_all):
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
    def _m1_motifadd(self,G,edge_all):
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
    def _m2(self,G,edge_all):
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
    def _m2_motifadd(self,G,edge_all):
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
    def _m3(self,G,edge_all):    
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
    def _m3_motifadd(self,G,edge_all):
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
    def _m4(self,G,edge_all):
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
    def _m4_motifadd(self,G,edge_all):
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
    def _m5(self,G,edge_all):
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
    def _m5_motifadd(self,G,edge_all):
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
    def _m6(self,G,edge_all):
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
    def _m6_motifadd(self,G,edge_all):
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
    def _m7(self,G,edge_all):
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
    def _m7_motifadd(self,G,edge_all):
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
    def _m8(self,G,edge_all):
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
    def _m8_motifadd(self,G,edge_all):
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
    def _calculate_motifs(self):
        """   
        Enumerating pairwise motif counts.
        """
        print("\nCalculating overlaps.\n")
        motif_adj = self.motif_func_list[self.M](self.graph,self.graph.edges())
        edges = [e for e in tqdm(self.graph.edges()) if motif_adj[e[0]][e[1]] >= self.cutoff]
        self.motif_graph = nx.from_edgelist(edges)

    def _extract_components(self):
        """
        Extracting connected components from motif graph.
        """
        print("\nExtracting components.\n")
        components = [c for c in nx.connected_components(self.motif_graph)]
        # print("components ",components )
        components = [[len(c), c] for c in components]
        components.sort(key=lambda x: x[0], reverse=True)
        important_components = [components[comp][1] for comp
                                in range(self.component_count if len(components)>=self.component_count else len(components))]
        self.blocks = [list(graph) for graph in important_components]

    def _fill_blocks(self):
        """
        Filling the dense blocks of the adjacency matrix.
        """
        print("Adding edge blocks.\n")
        new_edges = [(n_1, n_2) for nodes in self.blocks for n_1 in nodes for n_2 in nodes if n_1!= n_2]
        # print(new_edges)
        self.graph.add_edges_from(new_edges)

    def fit(self):
        """
        Clustering the target graph.
        """
        self._calculate_motifs()
        self._extract_components()
        self._fill_blocks()
        partition = community.best_partition(self.graph)
        return partition
