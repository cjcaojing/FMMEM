import networkx as nx
import time
from collections import deque
import copy
"""为网络可视化模体上色，找到模体的节点"""

def Feasibility_weighted(G,G_motif,n1,n2,N1,node,hasEdge):
    """
    根据同构条件判断是否是要寻找的模体
    :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
    :param G_motif: 模体图,可以有向或者无向但必须与图G一致
    :param n1: 模体网络G_motif中的节点
    :param n2: 模体网络G_motif中的节点
    :param N1: 目标网络G中的节点
    :param node: 目标网络G中的节点
    :param hasEdge: n1和n2之间是否有正向边
    :return True or False: 代表是否构成模体
    """
    if hasEdge:
        if(G_motif[n1][n2]['weight'])!= (G[N1][node]['weight']):
            return False

    return True

def Feasibility_weighted_directed(G,G_motif,n1,n2,N1,node,hasEdge,hasEdge1):
    """
    根据同构条件判断是否是要寻找的模体
    :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
    :param G_motif: 模体图,可以有向或者无向但必须与图G一致
    :param n1: 模体网络G_motif中的节点
    :param n2: 模体网络G_motif中的节点
    :param N1: 目标网络G中的节点
    :param node: 目标网络G中的节点
    :param hasEdge: n1和n2之间是否有正向边
    :param hasEdge1: n2和n1之间是否有反向边
    :return True or False: 代表是否构成模体
    """
    if hasEdge1:
        if (G_motif[n2][n1]['weight'])!= (G[node][N1]['weight']):
            return False

    if hasEdge:
        if (G_motif[n1][n2]['weight'])!= (G[N1][node]['weight']):
            return False

    return True

def preSort(G_motif,G2_node_list):
    """
    预处理：排序，调整搜索顺序，将度大的放前面，这样约束越多，剪枝越多，搜索越快。
    :param G_motif:
    :param G2_node_list:
    :param n_G2:
    """
    n_G2=len(G2_node_list)
    for i in range(1,n_G2):
        for j in range(0,n_G2-i):
            if G_motif.degree(G2_node_list[j+1])>G_motif.degree(G2_node_list[j]):
                t=G2_node_list[j]
                G2_node_list[j]=G2_node_list[j+1]
                G2_node_list[j+1]=t
    return G2_node_list

def DFS_nodes(M, m_nodes, rat, n, minQ, MS, deep, work, candi,result):

    if deep == len(m_nodes):
        if minQ >work:
            minQ=work
            for i in range(deep):
                result[i] = MS[i]
        # print(MS,minQ,work)
        return result,minQ
    work1=0
    candi1=0
    for node in m_nodes:
        if node in MS[:deep]:
            continue
        candi1 = candi
        for node1 in MS[:deep]:
            if M.has_edge(node,node1):
                candi1=rat*candi1
        candi1=candi1*n
        work1 = n * candi - candi1 + work
        MS[deep]=node
        result, minQ= DFS_nodes(M, m_nodes, rat, n, minQ, MS, deep + 1, work1, candi1, result)
    return result,minQ

def Motif_node_sort(M,m_nodes,deeps):
    """
    找到一种最快的搜索顺序
    :param M:
    :param m_nodes:
    :param deeps:
    :return:
    """
    rat=0.5
    n=10
    minQ=99999999999999
    MS=['*' for x in range(len(m_nodes))]
    work=0
    candi=1
    m_nodes1=['*' for x in range(len(m_nodes))]
    if deeps==1:
        MS[0]=m_nodes[0]
    if deeps == 2:
        MS[0]=m_nodes[0]
        MS[1]=m_nodes[1]
    m_nodes1,minQ=DFS_nodes(M, m_nodes, rat, n, minQ, MS, deeps, work, candi,m_nodes1)
    return m_nodes1


def find_motif(G,Motif,G_node_list,G2_node_list,n_G2,MS,deeps,derepeat,weighted=False):
    # start = time.time()
    number = 0  # 存储模体数量
    deep = deeps
    SL=[]
    for i in range(n_G2-deeps):
        SL.append([])
        for j in range(0,n_G2-deeps-i):
            SL[i].append([])

    searchset = set(G_node_list)
    searchset = searchset - set(MS[0:deeps])
    for j in range(deeps, n_G2):
        SL[0][j - deeps] = searchset
    for i in range(deeps):
        ss = []
        neighbor1 = searchset & set(G.neighbors(MS[i]))
        neighbor2 = searchset - set(G.neighbors(MS[i]))
        for j in range(deeps, n_G2):
            hasEdge = Motif.has_edge(G2_node_list[i], G2_node_list[j])
            if (hasEdge):
                searchset1 = neighbor1
            else:
                searchset1 = neighbor2

            if weighted:
                for k in searchset1:
                    if (not Feasibility_weighted(G, Motif, G2_node_list[i], G2_node_list[j], MS[i], k,hasEdge)):
                        ss.append(k)
            searchset1=searchset1-set(ss)
            if (not searchset1):
                return 0
            if i == 0:
                SL[0][j-deeps] =  searchset1
            else:
                SL[0][j - deeps] = set(SL[0][j - deeps]) & searchset1

    if n_G2 - deeps < 2:
        return len(SL[0][0])
    Quelist = []
    for i in range(n_G2 - deeps):
        Quelist.append(deque())

    Quelist[deep - deeps].extend(SL[deep - deeps][0])
    ddd = False
    while (1):
        if deep == n_G2 - 1:
            number +=len(SL[deep-deeps][0])
            # print(MS, SL[deep - deeps][0]);
            deep -= 1


        while (not Quelist[deep - deeps]):
            deep -= 1
            if deep < deeps:
                return number

        MS[deep] = Quelist[deep - deeps].pop()
        neighbor1 = set(G.neighbors(MS[deep]))
        for i in range(1,n_G2-deep):
            searchset = SL[deep-deeps][i]
            searchset = searchset - set(MS[0:deep+1])
            hasEdge= Motif.has_edge(G2_node_list[deep], G2_node_list[deep+i])
            if (hasEdge):
                searchset = searchset & neighbor1
            else:
                searchset = searchset - neighbor1

            if (not searchset):
                ddd = False
                break
            if derepeat[deep+i]!=-1 and derepeat[deep+i]<=deep:

                rrr=[]
                for k in list(searchset):
                    if int(k) <int(MS[derepeat[deep+i]]):
                        rrr.append(k)
                searchset= searchset - set(rrr)
            ss=[]
            if weighted:
                for k in searchset:
                    if (not Feasibility_weighted(G, Motif, G2_node_list[deep], G2_node_list[deep+i], MS[deep], k,hasEdge)):
                        ss.append(k)
            searchset = searchset-set(ss)
            if (not searchset):
                ddd = False
                break
            else:
                SL[deep - deeps+1][i-1] = searchset
                ddd = True
        if ddd:
            deep += 1
            if deep < n_G2-1:
                Quelist[deep - deeps].extend(SL[deep-deeps][0])

def find_motif_directed(G,Motif,G_node_list,G2_node_list,n_G2,MS,deeps,derepeat,weighted=False):
    # start = time.time()
    # print("@@@@@",G2_node_list)
    edgelist = list(nx.edges(G))
    edgelist1 = []
    for i in edgelist:
        edgelist1.append((i[1], i[0]))

    G2 = nx.DiGraph()
    G2.add_nodes_from(G_node_list)
    G2.add_edges_from(edgelist1)

    number = 0  # 存储模体数量
    deep = deeps
    SL=[]
    for i in range(n_G2-deeps):
        SL.append([])
        for j in range(0,n_G2-deeps-i):
            SL[i].append([])

    searchset = set(G_node_list) - set(MS[0:deeps])
    for j in range(deeps, n_G2):
        SL[0][j - deeps] = searchset

    for i in range(deeps):
        ss = []
        neighbor1 = searchset & set(G.neighbors(MS[i]))
        neighbor2 = searchset - set(G.neighbors(MS[i]))
        neighbor3 = searchset & set(G2.neighbors(MS[i]))
        neighbor4 = searchset - set(G2.neighbors(MS[i]))
        for j in range(deeps, n_G2):
            hasEdge = Motif.has_edge(G2_node_list[i], G2_node_list[j])
            hasEdge1 = Motif.has_edge(G2_node_list[j], G2_node_list[i])
            if (hasEdge):
                searchset1 = neighbor1
            else:
                searchset1 = neighbor2
            if (hasEdge1):
                searchset1 = searchset1 & neighbor3
            else:
                searchset1 = searchset1 & neighbor4
            if weighted:
                for k in searchset1:
                    if ( not Feasibility_weighted_directed(G, Motif, G2_node_list[i], G2_node_list[j], MS[i], k,hasEdge,hasEdge1)):
                        ss.append(k)
            searchset1=searchset1-set(ss)
            if (not searchset1):
                return 0

            if i == 0:
                SL[0][j - deeps] = searchset1
            else:
                SL[0][j - deeps] = set(SL[0][j - deeps]) & searchset1

    if n_G2 - deeps < 2:
        return len(SL[0][0])
    Quelist = []
    for i in range(n_G2 - deeps - 1):
        Quelist.append(deque())

    Quelist[deep - deeps].extend(SL[deep - deeps][0])
    ddd = False
    while (1):
        if  deep == n_G2 - 1:
            number += len(SL[deep-deeps][0])
            deep -= 1
        while (not Quelist[deep - deeps]):
            deep -= 1
            if deep < deeps:
                return number
        # if derepeat[deep] != -1:
        #     MS[deep] = Quelist[deep - deeps].pop()
        #
        #     while(int(MS[deep]) < int(MS[derepeat[deep]])):
        #         while (not Quelist[deep - deeps]):
        #             deep -= 1
        #             if deep < deeps:
        #                 return number
        #         MS[deep] = Quelist[deep - deeps].pop()
        #         if derepeat[deep] == -1:
        #             break
        # else:
        #     MS[deep] = Quelist[deep - deeps].pop()
        MS[deep] = Quelist[deep - deeps].pop()
        neighbor1 = set(G.neighbors(MS[deep]))
        neighbor2 = set(G2.neighbors(MS[deep]))
        for i in range(1,n_G2-deep):
            searchset = SL[deep-deeps][i]
            searchset = searchset - set(MS[0:deep+1])
            hasEdge= Motif.has_edge(G2_node_list[deep], G2_node_list[deep+i])
            hasEdge1 = Motif.has_edge(G2_node_list[deep+i], G2_node_list[deep])
            if (hasEdge):
                searchset = searchset & neighbor1
            else:
                searchset = searchset - neighbor1
            if (hasEdge1):
                searchset = searchset & neighbor2
            else:
                searchset = searchset - neighbor2

            if (not searchset):
                ddd = False
                break

            if derepeat[deep + i] != -1 and derepeat[deep + i] <= deep:
                rrr = []
                for k in list(searchset):
                    if int(k) < int(MS[derepeat[deep + i]]):
                        rrr.append(k)
                searchset = searchset - set(rrr)

            ss=[]
            if weighted:
                for k in searchset:
                    if ( not Feasibility_weighted_directed(G, Motif, G2_node_list[deep], G2_node_list[deep+i], MS[deep], k,hasEdge,hasEdge1)):
                        ss.append(k)
            searchset=searchset-set(ss)
            if (not searchset):
                ddd = False
                break
            else:
                SL[deep - deeps+1][i-1] = searchset
                ddd = True
        if ddd:
            deep += 1
            if deep < n_G2-1:
                Quelist[deep - deeps].extend(SL[deep-deeps][0])



def combine_motif_list(motif_list,MS):
    ll=copy.deepcopy(list(MS))
    motif_list.append(ll)
    # print(motif_list)

def filter_greater(node_list,num1):
    result=[]
    for i in node_list:
        if int(i) > num1:
            result.append(i)
    return result

def find_motif_list(G,Motif,G_node_list,G2_node_list,n_G2,MS,deeps,derepeat,weighted=False):
    # start = time.time()
    motif_list =[]
    number = 0  # 存储模体数量
    deep = deeps
    SL=[]
    for i in range(n_G2-deeps):
        SL.append([])
        for j in range(0,n_G2-deeps-i+1):
            SL[i].append([])

    searchset = set(G_node_list)
    searchset = searchset - set(MS[0:deeps])
    for j in range(deeps, n_G2):
        SL[0][j - deeps] = searchset

    for i in range(deeps):
        ss = []
        neighbor1 = searchset & set(G.neighbors(MS[i]))
        neighbor2 = searchset - set(G.neighbors(MS[i]))
        for j in range(deeps, n_G2):
            hasEdge=Motif.has_edge(G2_node_list[i], G2_node_list[j])
            if (hasEdge):
                searchset1 = neighbor1
            else:
                searchset1 = neighbor2

            if weighted:
                for k in searchset1:
                    if (not Feasibility_weighted(G, Motif, G2_node_list[i], G2_node_list[j], MS[i], k,hasEdge)):
                        ss.append(k)
                searchset1=searchset1-set(ss)
            if (not searchset1):
                return motif_list
            if i == 0:
                SL[0][j-deeps] =  searchset1
            else:
                SL[0][j - deeps] = set(SL[0][j - deeps]) & searchset1

    if n_G2 - deeps < 2:
        for i in SL[0][0]:
            MS[n_G2-1]=i
            combine_motif_list(motif_list, MS)
        return motif_list
    Quelist = []
    for i in range(n_G2 - deeps):
        Quelist.append(deque())

    Quelist[deep - deeps].extend(SL[deep - deeps][0])
    ddd = False

    while (1):
        # print("++++++",motif_list, MS)
        if deep > n_G2 - 1:
            deep -= 1
            # if derepeat[deep] !=-1:
            #     print("++++++++++++++",deep)
            #     SL[deep-deeps][0]=filter_greater(SL[deep-deeps][0],int(MS[derepeat[deep]]))
            number += len(SL[deep-deeps][0])
            combine_motif_list(motif_list,MS)

        while (not Quelist[deep - deeps]):
            deep -= 1
            if deep < deeps:
                return motif_list

        if derepeat[deep] != -1:
            MS[deep] = Quelist[deep - deeps].pop()
            while(int(MS[deep]) < int(MS[derepeat[deep]])):
                while (not Quelist[deep - deeps]):
                    deep -= 1
                    if deep < deeps:
                        return motif_list
                MS[deep] = Quelist[deep - deeps].pop()
                if derepeat[deep] == -1:
                    break
        else:
            MS[deep] = Quelist[deep - deeps].pop()
        neighbor1 = set(G.neighbors(MS[deep]))
        for i in range(1,n_G2-deep):
            searchset = set(SL[deep-deeps][i])
            searchset = searchset - set(MS[0:deep+1])
            hasEdge= Motif.has_edge(G2_node_list[deep], G2_node_list[deep+i])
            if (hasEdge):
                searchset = searchset & neighbor1
            else:
                searchset = searchset - neighbor1

            if (not searchset):
                ddd = False
                break
            ss=[]

            if weighted:
                for k in searchset:
                    if (not Feasibility_weighted(G, Motif, G2_node_list[deep], G2_node_list[deep+i], MS[deep], k,hasEdge)):
                        ss.append(k)
            searchset=searchset-set(ss)
            if (not searchset):
                ddd = False
                break
            else:
                SL[deep - deeps+1][i-1] = searchset
                ddd = True
        if ddd:
            deep += 1
            if deep < n_G2:
                Quelist[deep - deeps].extend(SL[deep-deeps][0])

def find_motif_directed_list(G,Motif,G_node_list,G2_node_list,n_G2,MS,deeps,derepeat,weighted=False):
    # start = time.time()
    # print("@@@@@",G2_node_list)
    motif_list = []
    edgelist = list(nx.edges(G))
    edgelist1 = []
    for i in edgelist:
        edgelist1.append((i[1], i[0]))
    G2 = nx.DiGraph()
    G2.add_nodes_from(G_node_list)
    G2.add_edges_from(edgelist1)

    number = 0  # 存储模体数量
    deep = deeps
    SL=[]
    for i in range(n_G2-deeps):
        SL.append([])
        for j in range(0,n_G2-deeps-i+1):
            SL[i].append([])
    searchset = set(G_node_list) - set(MS[0:deeps])
    for j in range(deeps, n_G2):
        SL[0][j - deeps] = searchset

    for i in range(deeps):

        ss = []
        neighbor1 = searchset & set(G.neighbors(MS[i]))
        neighbor2 = searchset - set(G.neighbors(MS[i]))
        neighbor3 = searchset & set(G2.neighbors(MS[i]))
        neighbor4 = searchset - set(G2.neighbors(MS[i]))
        for j in range(deeps, n_G2):
            hasEdge=Motif.has_edge(G2_node_list[i], G2_node_list[j])
            hasEdge1 = Motif.has_edge(G2_node_list[j], G2_node_list[i])
            if (hasEdge):
                searchset1 = neighbor1
            else:
                searchset1 = neighbor2
            if (hasEdge1):
                searchset1 = searchset1 & neighbor3
            else:
                searchset1 = searchset1 & neighbor4
            if weighted:
                for k in searchset1:
                    if ( not Feasibility_weighted_directed(G, Motif, G2_node_list[i], G2_node_list[j], MS[i], k,hasEdge,hasEdge1)):
                        ss.append(k)
            searchset1=searchset1-set(ss)
            if (not searchset1):
                return motif_list

            if i == 0:
                SL[0][j-deeps]=  searchset1
            else:
                SL[0][j - deeps] = set(SL[0][j - deeps]) & searchset1

    if n_G2 - deeps < 2:
        for i in SL[0][0]:
            MS[n_G2-1]=i
            combine_motif_list(motif_list, MS)

        return motif_list
        # return SL[0][0]
    Quelist = []
    for i in range(n_G2 - deeps):
        Quelist.append(deque())

    Quelist[deep - deeps].extend(SL[deep - deeps][0])
    ddd = False

    while (1):
        if  deep > n_G2 - 1:
            deep -= 1
            # if derepeat[deep] !=-1:
            #     print("++++++++++++++",deep)
            #     SL[deep-deeps][0]=filter_greater(SL[deep-deeps][0],int(MS[derepeat[deep]]))
            number += len(SL[deep-deeps][0])
            combine_motif_list(motif_list, MS)
        while (not Quelist[deep - deeps]):
            deep -= 1
            if deep < deeps:
                return motif_list

        if derepeat[deep] != -1:
            MS[deep] = Quelist[deep - deeps].pop()

            while(int(MS[deep]) < int(MS[derepeat[deep]])):
                while (not Quelist[deep - deeps]):
                    deep -= 1
                    if deep < deeps:
                        return motif_list
                MS[deep] = Quelist[deep - deeps].pop()
                if derepeat[deep] == -1:
                    break
        else:
            MS[deep] = Quelist[deep - deeps].pop()
        neighbor1 = set(G.neighbors(MS[deep]))
        neighbor2 = set(G2.neighbors(MS[deep]))
        for i in range(1,n_G2-deep):
            searchset = SL[deep-deeps][i]
            searchset = searchset - set(MS[0:deep+1])
            hasEdge= Motif.has_edge(G2_node_list[deep], G2_node_list[deep+i])
            hasEdge1 = Motif.has_edge(G2_node_list[deep+i], G2_node_list[deep])
            if (hasEdge):
                searchset = searchset & neighbor1
            else:
                searchset = searchset - neighbor1
            if (hasEdge1):
                searchset = searchset & neighbor2
            else:
                searchset = searchset - neighbor2

            if (not searchset):
                ddd = False
                break
            ss=[]
            if weighted:
                for k in searchset:
                    if ( not Feasibility_weighted_directed(G, Motif, G2_node_list[deep], G2_node_list[deep+i], MS[deep], k,hasEdge,hasEdge1)):
                        ss.append(k)
            searchset=searchset-set(ss)
            if (not searchset):
                ddd = False
                break
            else:
                SL[deep - deeps+1][i-1] = searchset
                ddd = True
        if ddd:
            deep += 1
            if deep < n_G2:
                Quelist[deep - deeps].extend(SL[deep-deeps][0])


def get_node_group(G2_node_list,n_G2,repetitions):

    nodes_group={}
    for i in G2_node_list:
        nodes_group[i]={}
        for j in G2_node_list:
            nodes_group[i][j]=-1
    for i in range(n_G2):
        for j in range(len(repetitions)):
            for k in range(len(repetitions)):
                nodes_group[repetitions[j][i]][repetitions[k][i]] = 1
                nodes_group[repetitions[k][i]][repetitions[j][i]] = 1
    return nodes_group

def Build_Motif_Tree(motif, nodelist):
    motif_tree={}
    mark={}
    # nodelist=list(nx.nodes(motif))
    for i in nodelist:
        mark[i]=0
        motif_tree[i]=[]
    mark[nodelist[0]]=1
    build_tree(motif,mark,nodelist[0],motif_tree)
    return motif_tree

def build_tree(motif,mark,node,motif_tree):
    a=list(motif.nodes())
    b=[]
    for i in a:
        if mark[i] ==0 and (motif.has_edge(i,node) or motif.has_edge(node,i)):
            b.append(i)
            mark[i]=1
    # if b==[]:
    #     return
    motif_tree[node]=b
    for i in motif_tree[node]:
        build_tree(motif,mark,i,motif_tree)

#
# def get_prevent_repetition_list(motif,G2_node_list,len1,MS,deep, weighted,derepet):
#     repetitions= find_motif_list(motif,motif, G2_node_list, G2_node_list, len1, MS, deep,derepet, weighted)
#     if len(repetitions)==1:
#         return derepet
#     nodes_group = get_node_group(G2_node_list,len1,repetitions)
#     for i in range(len1):
#         k=-1
#         s=0
#         for j in range(len1):
#             if(nodes_group[G2_node_list[i]][G2_node_list[j]]==1 and derepet[j]==-1):
#                 derepet[j]=k
#                 k=j
#                 s+=1
#         if(s>1):
#             break
#     # print(derepet)
#     return get_prevent_repetition_list(motif, G2_node_list, len1, MS, deep, weighted,derepet)


def get_prevent_repetition_list(motif,G2_node_list,len1,repetitions):
    isconnect=True
    direc=nx.is_directed(motif)
    if not direc:
        isconnect = nx.is_connected(motif)
    if( (not isconnect) and (not direc)):
        moti = nx.Graph()
        # edge_ss=[]
        moti.add_nodes_from(G2_node_list)
        for i in G2_node_list:
            for j in G2_node_list:
                if(i!=j):
                    if( not motif.has_edge(i,j)):
                        moti.add_edge(i,j)
                        # edge_ss.append((i,j))
                    if( not motif.has_edge(j,i)):
                        moti.add_edge(j, i)
                        # edge_ss.append((j, i))
        # moti.add_edges_from(edge_ss)
        motif = moti


    nodes_group = get_node_group(G2_node_list,len1,repetitions)


    motif_tree = Build_Motif_Tree(motif,G2_node_list)
    # print(motif_tree)
    mapnodelist={}
    for i in range(len(G2_node_list)):
        mapnodelist[G2_node_list[i]]=i

    derepet = [-1 for i in range(len1)]

    for i in G2_node_list[1:]:
        if nodes_group[G2_node_list[0]][i]==1:
            derepet[mapnodelist[i]]=0

    mark={}
    for i in G2_node_list:
        mark[i]=0
    for node in G2_node_list:
        for i in range(len(motif_tree[node])):
            if mark[motif_tree[node][i]]== 0:
                aaa=[]
                aaa.append(motif_tree[node][i])
                mark[motif_tree[node][i]] = 1
                for j in range(len(motif_tree[node])):
                    if i < j:
                        if nodes_group[motif_tree[node][i]][motif_tree[node][j]] == 1:
                            aaa.append(motif_tree[node][j])
                            mark[motif_tree[node][j]]=1
                bbb=[]
                for k in G2_node_list:
                    if k in aaa:
                        bbb.append(k)
                for k in range(len(bbb)):
                    if k==0:
                        continue
                    derepet[mapnodelist[bbb[k]]]=mapnodelist[bbb[k-1]]


    return derepet


def node_motif_num(G,G_motif,node,orbit_node,directed=False,weighted=False):
    """
        计算node节点作为模体的orbit所参与的模体数量
        :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
        :param G_motif: 模体图,可以有向或者无向但必须与图G一致
        :param node: 图G中的一个节点
        :param orbit_node: motif中的一个节点代表模体的一个轨道
        :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
        :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
        :return :点模体数
    """

    # start = time.time()
    G_node_list = list(nx.nodes(G))
    G2_node_list = list(nx.nodes(G_motif))
    len1 = len(G2_node_list)
    # 根据轨道调整节点顺序
    for nod in range(len1):
        if G2_node_list[nod] == orbit_node:
            ab = nod
            break
    G2_node_list[ab] = G2_node_list[0]
    G2_node_list[0] = orbit_node


    # G2_node_list[1:len1]=preSort(G_motif, G2_node_list[1:len1])  # 调整搜索顺序
    G2_node_list = Motif_node_sort(G_motif, G2_node_list, 1)
    MS = ['*' for x in range(len1)]
    node_motif_number = 0
    if directed:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        MS[0] = G2_node_list[0]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_directed_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
        # print(repetitions)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        # print(derepeat)
        # 下面查找node节点的模体数
        MS[0] = node
        node_motif_number = find_motif_directed(G, G_motif, G_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
        MS[0] = G2_node_list[0]
        # a = find_motif_directed(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
    else:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        MS[0] = G2_node_list[0]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        # 下面查找node节点的模体数
        MS[0] = node
        node_motif_number = find_motif(G, G_motif, G_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
        MS[0] = G2_node_list[0]
        # a = find_motif(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
    # end = time.time()
    # print("node_motif_num总共用时{}秒".format((end - start)))
    #结果是需要除以重复的数
    return node_motif_number

def edge_motif_num(G,G_motif,edge,orbit_edge,directed=False,weighted=False):
    """
        计算edge作为模体的orbit所参与的模体数量
        :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
        :param G_motif: 模体图,可以有向或者无向但必须与图G一致
        :param edge: 图G中的一条边
        :param orbit_edge: motif中的一个边代表模体的一个边轨道
        :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
        :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
        :return :返回值是边模体数
    """
    # if directed:
    #     twoWay = False
    #     if not G.has_edge(edge[0], edge[1]):
    #         print("There is no such edge in the network.",edge)
    #         return 0
    #     if G.has_edge(edge[1], edge[0]):
    #         twoWay = True
    # else :
    #     twoWay =True

    twoWay = False
    if (not G.has_edge(edge[0], edge[1])) and (not G.has_edge(edge[1], edge[0])):
        print("There is no such edge in the network.")
        twoWay = True
    if G.has_edge(edge[1], edge[0]) and G.has_edge(edge[0], edge[1]):
        twoWay = True

    # start = time.time()
    G_node_list = list(nx.nodes(G))
    G2_node_list = list(nx.nodes(G_motif))
    len1 = len(G2_node_list)
    # 根据轨道调整节点顺序
    for nod in range(len1):
        if G2_node_list[nod] == orbit_edge[0]:
            ab = nod
            break
    G2_node_list[ab] = G2_node_list[0]
    G2_node_list[0] = orbit_edge[0]
    for nod in range(len1):
        if G2_node_list[nod] == orbit_edge[1]:
            ab = nod
            break
    G2_node_list[ab] = G2_node_list[1]
    G2_node_list[1] = orbit_edge[1]


    # G2_node_list[2:len1]=preSort(G_motif, G2_node_list[2:len1])  # 调整搜索顺序
    G2_node_list = Motif_node_sort(G_motif, G2_node_list, 2)
    MS = ['*' for x in range(len1)]
    edge_motif_number=0

    if directed:

        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        MS[0] = G2_node_list[0]
        MS[1] = G2_node_list[1]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_directed_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 0, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list,len1, repetitions)
        if (derepeat[1] != -1):
            twoWay = False
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_directed_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat,weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        # 计算边模体数量01的顺序不同
        MS[0] = edge[0]
        MS[1] = edge[1]
        edge_motif_number = find_motif_directed(G, G_motif, G_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        MS[0] = G2_node_list[0]
        MS[1] = G2_node_list[1]
        # a = find_motif_directed(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        if twoWay:  # 如果是双向边还需要查找反方向的模体数
            MS[0] = edge[1]
            MS[1] = edge[0]
            edge_motif_number += find_motif_directed(G, G_motif, G_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
            MS[0] = G2_node_list[1]
            MS[1] = G2_node_list[0]
            # a += find_motif_directed(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)

    else:

        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        MS[0] = G2_node_list[0]
        MS[1] = G2_node_list[1]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 0, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        if (derepeat[1] != -1):
            twoWay = False

        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        #------------------------------#
        MS[0] = edge[0]
        MS[1] = edge[1]
        edge_motif_number= find_motif(G, G_motif, G_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        MS[0] = G2_node_list[0]
        MS[1] = G2_node_list[1]
        # a = find_motif(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        if twoWay:  # 如果是双向边还需要查找反方向的模体数
            MS[0] = edge[1]
            MS[1] = edge[0]
            edge_motif_number += find_motif(G, G_motif, G_node_list, G2_node_list, len1, MS, 2, derepeat,weighted)
            MS[0] = G2_node_list[1]
            MS[1] = G2_node_list[0]
            # a += find_motif(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
    # end = time.time()
    # print("edge_motif_num总共用时{}秒".format((end - start)))
    # 结果是需要除以重复的数
    return edge_motif_number


def total_motif_num(G,G_motif,directed=False,weighted=False):
    """
        计算模体总数
        :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
        :param G_motif: 模体图,可以有向或者无向但类型必须与图G一致
        :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
        :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
        :return :返回值是模体总数
    """
    G_node_list = list(nx.nodes(G))
    G2_node_list = list(nx.nodes(G_motif))
    n_G2 = len(G2_node_list)
    # start = time.time()

    # G2_node_list=preSort(G_motif, G2_node_list)  # 调整搜索顺序
    G2_node_list = Motif_node_sort(G_motif, G2_node_list,0)

    MS = ['*' for x in range(n_G2)]
    total_motif_number = 0
    # repetitions = 0
    # 计算模体总数量
    if directed:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        derepeat = [-1 for i in range(n_G2)]
        repetitions = find_motif_directed_list(G_motif, G_motif, G2_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, n_G2, repetitions)
        total_motif_number = find_motif_directed(G, G_motif, G_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)
        # a=find_motif_directed(G_motif, G_motif, G2_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)

    else:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        derepeat = [-1 for i in range(n_G2)]
        repetitions = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, n_G2, MS, 0,derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, n_G2, repetitions)
        # print(derepeat)
        total_motif_number = find_motif(G, G_motif, G_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)
        # a = find_motif(G_motif, G_motif, G2_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)
        # print(a)
        # print(total_motif_number)
    # end = time.time()
    # print("total_motif_num总共用时{}秒".format((end - start)))
    # 结果是需要除以重复的数
    return total_motif_number

def Isomorphism_Self_search(G,directed=False,weighted=False):
    G_node_list = list(nx.nodes(G))
    n_G2 = len(G_node_list)
    G2_node_list=G_node_list
    MS = ['*' for x in range(n_G2)]
    derepeat = [-1 for i in range(n_G2)]
    # 计算模体总数量
    if directed:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        total_motif = find_motif_directed_list(G, G, G_node_list, G_node_list, n_G2, MS, 0, derepeat, weighted)
    else:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        total_motif = find_motif_list(G, G, G_node_list, G_node_list, n_G2, MS, 0, derepeat, weighted)
    motifss = total_motif

    # print("motif num :",len(motifss))
    motif_edge = []
    for i in range(len(motifss)):
        motif_edge.append([])
    if directed:
        for i in range(len(G2_node_list)):
            for j in range(len(G2_node_list)):
                if (i != j):
                    if (G.has_edge(G2_node_list[i], G2_node_list[j])):
                        for k in range(len(motifss)):
                            motif_edge[k].append((motifss[k][i], motifss[k][j]))
    else:
        for i in range(len(G2_node_list)):
            for j in range(len(G2_node_list)):
                if (j > i):
                    if (G.has_edge(G2_node_list[i], G2_node_list[j])):
                        for k in range(len(motifss)):
                            motif_edge[k].append((motifss[k][i], motifss[k][j]))
    # print(motif_edge)
    return motifss, motif_edge

def total_motif_list(G,G_motif,directed=False,weighted=False):
    """
        计算模体总数
        :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
        :param G_motif: 模体图,可以有向或者无向但类型必须与图G一致
        :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
        :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
        :return :返回值是模体总数
    """
    G_node_list = list(nx.nodes(G))
    G2_node_list = list(nx.nodes(G_motif))
    n_G2 = len(G2_node_list)
    # start = time.time()

    # G2_node_list=preSort(G_motif, G2_node_list)  # 调整搜索顺序
    G2_node_list = Motif_node_sort(G_motif, G2_node_list,0)

    MS = ['*' for x in range(n_G2)]


    # 计算模体总数量
    if directed:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        derepeat = [-1 for i in range(n_G2)]
        repetitions = find_motif_directed_list(G_motif, G_motif, G2_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, n_G2, repetitions)
        total_motif = find_motif_directed_list(G, G_motif, G_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)
        # a=find_motif_directed(G_motif, G_motif, G2_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)

    else:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        derepeat = [-1 for i in range(n_G2)]
        repetitions = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, n_G2, MS, 0,derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, n_G2, repetitions)
        # print(derepeat)
        total_motif = find_motif_list(G, G_motif, G_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)
        # a = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, n_G2, MS, 0, derepeat, weighted)
    # print(a)


    motifss = total_motif

    # print("motif num :",len(motifss))
    motif_edge=[]
    for i in range(len(motifss)):
        motif_edge.append([])
    if directed:
        for i in range(len(G2_node_list)):
            for j in  range(len(G2_node_list)):
                if(i!=j):
                    if(G_motif.has_edge(G2_node_list[i],G2_node_list[j])):
                        for k in range(len(motifss)):
                            motif_edge[k].append((motifss[k][i],motifss[k][j]))
    else:
        for i in range(len(G2_node_list)):
            for j in range(len(G2_node_list)):
                if (j>i):
                    if(G_motif.has_edge(G2_node_list[i],G2_node_list[j])):
                        for k in range(len(motifss)):
                            motif_edge[k].append((motifss[k][i],motifss[k][j]))
    # print(motif_edge)
    return motifss,motif_edge


def node_orbit(G,directed=False,weighted=False):
    node_list1 = list(G.nodes)
    node_num = len(node_list1)
    # start = time.time()
    MS = ['*' for x in range(node_num)]
    derepeat = [-1 for i in range(node_num)]
    if directed:
        repetitions = find_motif_directed_list(G, G, node_list1, node_list1, node_num, MS, 0, derepeat,weighted)
    else:
        repetitions = find_motif_list(G, G, node_list1, node_list1, node_num, MS, 0, derepeat, weighted)


    repelen=len(repetitions)
    orbit_list=[]

    sum=0
    nodesss=[]
    for i in range(node_num):
        orbit = set()
        if (repetitions[0][i] in nodesss):
            continue
        for j in range(repelen):
            orbit.add(repetitions[j][i])
            nodesss.append(repetitions[j][i])
        sum=sum+len(orbit)
        orbit_list.append(list(orbit))
        if(sum >= node_num):
            break
    # print(orbit_list)
    return orbit_list


def edge_orbit(G,directed=False,weighted=False):
    edgelist=list(G.edges())
    nodes,edges=Isomorphism_Self_search(G,directed,weighted)
    m=len(edges)
    n=0
    if(m>0):
        n=len(edges[0])
    edge_orbits=[]
    edge_set=set()
    for i in range(n):
        edge_set=set()
        for j in range(m):
            if edges[j][i] in edgelist:
                edge_set.add(edges[j][i])
                edgelist.remove(edges[j][i])
        if(len(list(edge_set))>0):
            edge_orbits.append(list(edge_set))

    # print(edge_orbits)
    return edge_orbits

# def edge_orbit(G,directed=False,weighted=False):
#     edge_list=list(G.edges)
#     edge_num=len(edge_list)
#     edge2node= range(edge_num)
#     node2edge=[]
#     G2 = nx.Graph()
#     node_list1 = list(G.nodes)
#     node_num = len(node_list1)
#     if directed or weighted:
#         # start = time.time()
#         MS = ['*' for x in range(node_num)]
#         derepeat = [-1 for i in range(node_num)]
#         if directed:
#             repetitions = find_motif_directed_list(G, G, node_list1, node_list1, node_num, MS, 0, derepeat, weighted)
#         else:
#             repetitions = find_motif_list(G, G, node_list1, node_list1, node_num, MS, 0, derepeat, weighted)
#         print("motif",repetitions[1])
#         repelen = len(repetitions)
#         orbit_list = []
#
#         sum = 0
#         nodesss = []
#         for i in range(node_num-1):
#             for j in range(i+1,node_num):
#                 if(G.has_edge(node_list1[i],node_list1[j]) or G.has_edge(node_list1[j],node_list1[i])):
#                     orbit = set()
#                     if ((repetitions[0][i],repetitions[0][j]) in nodesss):
#                         continue
#                     for k in range(repelen):
#                         orbit.add((repetitions[k][i],repetitions[k][j]))
#                         nodesss.append((repetitions[k][i],repetitions[k][j]))
#                     sum = sum + len(orbit)
#                     orbit_list.append(list(orbit))
#                     if (sum >= edge_num):
#                         break
#         # print(orbit_list)
#         return orbit_list
#     else:
#         for i in edge2node:
#             for j in edge2node:
#                 if i >= j:
#                     continue
#                 if edge_list[i][0] == edge_list[j][0]:
#                     node2edge.append((i, j))
#                 if edge_list[i][0] == edge_list[j][1]:
#                     node2edge.append((i, j))
#                 if edge_list[i][1] == edge_list[j][0]:
#                     node2edge.append((i, j))
#                 if edge_list[i][1] == edge_list[j][1]:
#                     node2edge.append((i, j))
#
#
#         G2.add_edges_from(node2edge)
#         orbit_list= node_orbit(G2)
#         result=[]
#         for orbit in range(len(orbit_list)):
#             result.append([])
#             for i in orbit_list[orbit]:
#                 result[orbit].append(edge_list[i])
#
#         return result


# def edge_orbit(G,directed=False,weighted=False):
#     node_list=list(G.nodes)
#     edge_list=list(G.edges)
#     node_num=len(node_list)
#     edge_num=len(edge_list)
#     edge2node= range(edge_num)
#     node2edge=[]
#     G2 = nx.Graph()
#     node_lables= False
#     if directed and (not weighted):
#         node_lables = True
#         for i in range(node_num):
#             for j in range(node_num):
#                 if i >= j:
#                     continue
#                 if G.has_edge(node_list[i],node_list[j]) and G.has_edge(node_list[i],node_list[j]):
#                     G2.add_node(i,index='2')
#                     print(G2.nodes[i])
#                 elif G.has_edge(node_list[i],node_list[j]) :
#                     G2.add_node(i, index='1')
#                 elif G.has_edge(node_list[i],node_list[j]):
#                     G2.add_node(i, index='-1')
#     if weighted and (not directed):
#         node_lables = True
#         for i in range(edge_num):
#             edgess=edge_list[i]
#             G2.add_node(i,index=edgess["weight"])
#     if directed and weighted:
#         node_lables = True
#         for i in range(edge_num):
#             edgess=edge_list[i]
#             if g.has_edge(edgess[0],edgess[1]) and g.has_edge(edgess[1],edgess[0]):
#                 G2.add_node(i,index='2'+edgess["weight"])
#             elif int(edgess[0]) < int(edgess[1]):
#                 G2.add_node(i, index='1'+edgess["weight"])
#             else :
#                 G2.add_node(i, index='-1'+edgess["weight"])
#
#     for i in edge2node:
#         for j in edge2node:
#             if i >= j:
#                 continue
#             if edge_list[i][0] == edge_list[j][0]:
#                 node2edge.append((i, j))
#             if edge_list[i][0] == edge_list[j][1]:
#                 node2edge.append((i, j))
#             if edge_list[i][1] == edge_list[j][0]:
#                 node2edge.append((i, j))
#             if edge_list[i][1] == edge_list[j][1]:
#                 node2edge.append((i, j))
#
#
#     G2.add_edges_from(node2edge)
#     orbit_list= node_orbit(G2,node_lable=node_lables)
#     result=[]
#     for orbit in range(len(orbit_list)):
#         result.append([])
#         for i in orbit_list[orbit]:
#             result[orbit].append(edge_list[i])
#
#     return result



def node_in_motif(G,G_motif,node,directed=False,weighted=False):
    """
           计算node节点参与的模体数量
           :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
           :param G_motif: 模体图,可以有向或者无向但必须与图G一致
           :param node: 图G中的一个节点
           :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
           :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
           :return :点模体数
    """
    result=0
    orbit_list=node_orbit(G_motif)
    for i in range(len(orbit_list)):
        # start = time.time()
        result += node_motif_num(G, G_motif, node,orbit_list[i][0], directed, weighted)  # 点模体
    return result

def edge_in_motif(G,G_motif,edge,directed=False,weighted=False):
    """
           计算边edge参与构成模体的数量
           :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
           :param G_motif: 模体图,可以有向或者无向但必须与图G一致
           :param edge: 图G中的一条边
           :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
           :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
           :return :点模体数
    """
    result=0
    orbit_list=edge_orbit(G_motif,directed,weighted)
    for i in range(len(orbit_list)):
        # start = time.time()
        result += edge_motif_num(G, G_motif, edge,orbit_list[i][0], directed, weighted)  # 点模体
    return result


def node_coverage_rate_of_motif(G,G_motif,directed=False, weighted=False):
    G_node_list = list(nx.nodes(G))
    node_num=len(G_node_list)
    in_motif=0
    for node in G_node_list:
        a = node_in_motif(G, G_motif, node, directed, weighted)
        if a>0:
            in_motif+=1
    return in_motif/node_num

def edge_coverage_rate_of_motif(G,G_motif,directed=False, weighted=False):
    G_edge_list = list(nx.edges(G))
    edge_num = len(G_edge_list)
    in_motif = 0
    for edge in G_edge_list:
        a = edge_in_motif(G, G_motif, edge, directed, weighted)
        if a > 0:
            in_motif += 1
    return in_motif / edge_num


def node_motif_list(G,G_motif,node,orbit_node,directed=False,weighted=False):
    """
        计算node节点所参与的模体集合
        :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
        :param G_motif: 模体图,可以有向或者无向但必须与图G一致
        :param node: 图G中的一个节点
        :param orbit_node: motif中的一个节点代表模体的一个轨道
        :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
        :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
        :return :点模体集合
    """

    # start = time.time()
    G_node_list = list(nx.nodes(G))
    G2_node_list = list(nx.nodes(G_motif))
    len1 = len(G2_node_list)
    # 根据轨道调整节点顺序
    for nod in range(len1):
        if G2_node_list[nod] == orbit_node:
            ab = nod
            break
    G2_node_list[ab] = G2_node_list[0]
    G2_node_list[0] = orbit_node


    # G2_node_list[1:len1]=preSort(G_motif, G2_node_list[1:len1])  # 调整搜索顺序
    G2_node_list = Motif_node_sort(G_motif, G2_node_list, 1)
    MS = ['*' for x in range(len1)]
    node_motif_number = 0
    if directed:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        MS[0] = G2_node_list[0]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_directed_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
        # print(repetitions)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        # 下面查找node节点的模体数
        MS[0] = node
        node_motif = find_motif_directed_list(G, G_motif, G_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
        MS[0] = G2_node_list[0]
        # a = find_motif_directed(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
    else:
        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        MS[0] = G2_node_list[0]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
        # print(repetitions)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        # print(derepeat)
        # 下面查找node节点的模体数
        MS[0] = node
        node_motif = find_motif_list(G, G_motif, G_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
        MS[0] = G2_node_list[0]
        # a = find_motif(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 1, derepeat, weighted)
    # end = time.time()
    # print("node_motif_num总共用时{}秒".format((end - start)))
    #结果是需要除以重复的数

    motifss = node_motif

    # print("motif num :", len(motifss))
    motif_edge = []
    for i in range(len(motifss)):
        motif_edge.append([])
    if directed:
        for i in range(len(G2_node_list)):
            for j in range(len(G2_node_list)):
                if (i != j):
                    if (G_motif.has_edge(G2_node_list[i], G2_node_list[j])):
                        for k in range(len(motifss)):
                            motif_edge[k].append((motifss[k][i], motifss[k][j]))
    else:
        for i in range(len(G2_node_list)):
            for j in range(len(G2_node_list)):
                if (j > i):
                    if (G_motif.has_edge(G2_node_list[i], G2_node_list[j])):
                        for k in range(len(motifss)):
                            motif_edge[k].append((motifss[k][i], motifss[k][j]))
    # print(motif_edge)
    return motifss, motif_edge

def edge_motif_list(G,G_motif,edge,orbit_edge,directed=False,weighted=False):
    """
        计算edge所参与的模体集合
        :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
        :param G_motif: 模体图,可以有向或者无向但必须与图G一致
        :param edge: 图G中的一条边
        :param orbit_edge: motif中的一个边代表模体的一个边轨道
        :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
        :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
        :return :返回值是边模体集合
    """
    # if directed:
    #     twoWay = False
    #     if not G.has_edge(edge[0], edge[1]):
    #         print("There is no such edge in the network.",edge)
    #         return 0
    #     if G.has_edge(edge[1], edge[0]):
    #         twoWay = True
    # else :
    #     twoWay =True

    twoWay = False
    if (not G.has_edge(edge[0], edge[1])) and (not G.has_edge(edge[1], edge[0])):
        print("There is no such edge in the network.")
        twoWay = True
    if G.has_edge(edge[1], edge[0]) and G.has_edge(edge[0], edge[1]):
        twoWay = True


    # start = time.time()
    G_node_list = list(nx.nodes(G))
    G2_node_list = list(nx.nodes(G_motif))
    len1 = len(G2_node_list)
    # 根据轨道调整节点顺序
    for nod in range(len1):
        if G2_node_list[nod] == orbit_edge[0]:
            ab = nod
            break
    G2_node_list[ab] = G2_node_list[0]
    G2_node_list[0] = orbit_edge[0]
    for nod in range(len1):
        if G2_node_list[nod] == orbit_edge[1]:
            ab = nod
            break
    G2_node_list[ab] = G2_node_list[1]
    G2_node_list[1] = orbit_edge[1]


    # G2_node_list[2:len1]=preSort(G_motif, G2_node_list[2:len1])  # 调整搜索顺序
    G2_node_list = Motif_node_sort(G_motif, G2_node_list, 2)
    MS = ['*' for x in range(len1)]
    edge_motif_number=0


    if directed:

        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        MS[0] = G2_node_list[0]
        MS[1] = G2_node_list[1]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_directed_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 0, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list,len1, repetitions)
        if (derepeat[1] != -1):
            twoWay = False

        MS[1] = G2_node_list[1]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_directed_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat,weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        # 计算边模体数量01的顺序不同
        MS[0] = edge[0]
        MS[1] = edge[1]
        edge_motif = find_motif_directed_list(G, G_motif, G_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        MS[0] = G2_node_list[0]
        MS[1] = G2_node_list[1]
        # a = find_motif_directed(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        if twoWay:  # 如果是双向边还需要查找反方向的模体数
            MS[0] = edge[1]
            MS[1] = edge[0]
            edge_motif.extend( find_motif_directed_list(G, G_motif, G_node_list, G2_node_list, len1, MS, 2, derepeat, weighted))
            MS[0] = G2_node_list[1]
            MS[1] = G2_node_list[0]
            # a += find_motif_directed(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)

    else:

        # 搜索模体图中的模体数，如果大于1，说明有对称结构导致重复计算。
        MS[0] = G2_node_list[0]
        MS[1] = G2_node_list[1]
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 0, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)

        if (derepeat[1] != -1):
            twoWay = False
        derepeat = [-1 for i in range(len1)]
        repetitions = find_motif_list(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        derepeat = get_prevent_repetition_list(G_motif, G2_node_list, len1, repetitions)
        #------------------------------#
        MS[0] = edge[0]
        MS[1] = edge[1]
        edge_motif= find_motif_list(G, G_motif, G_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        MS[0] = G2_node_list[0]
        MS[1] = G2_node_list[1]
        # a = find_motif(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
        if twoWay:  # 如果是双向边还需要查找反方向的模体数
            MS[0] = edge[1]
            MS[1] = edge[0]
            edge_motif.extend(  find_motif_list(G, G_motif, G_node_list, G2_node_list, len1, MS, 2, derepeat,weighted))
            MS[0] = G2_node_list[1]
            MS[1] = G2_node_list[0]
            # a += find_motif(G_motif, G_motif, G2_node_list, G2_node_list, len1, MS, 2, derepeat, weighted)
    # end = time.time()
    # print("edge_motif_num总共用时{}秒".format((end - start)))
    # 结果是需要除以重复的数

    motifss = edge_motif

    # print("motif num :", len(motifss))
    motif_edge = []
    for i in range(len(motifss)):
        motif_edge.append([])
    if directed:
        for i in range(len(G2_node_list)):
            for j in range(len(G2_node_list)):
                if (i != j):
                    if (G_motif.has_edge(G2_node_list[i], G2_node_list[j])):
                        for k in range(len(motifss)):
                            motif_edge[k].append((motifss[k][i], motifss[k][j]))
    else:
        for i in range(len(G2_node_list)):
            for j in range(len(G2_node_list)):
                if (j > i):
                    if (G_motif.has_edge(G2_node_list[i], G2_node_list[j])):
                        for k in range(len(motifss)):
                            motif_edge[k].append((motifss[k][i], motifss[k][j]))
    # print(motif_edge)
    return motifss, motif_edge


def node_in_motif_list(G,G_motif,node,directed=False,weighted=False):
    """
           计算node节点参与的模体数量
           :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
           :param G_motif: 模体图,可以有向或者无向但必须与图G一致
           :param node: 图G中的一个节点
           :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
           :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
           :return :点模体数
    """
    result_node = []
    result_edge = []
    orbit_list=node_orbit(G_motif)
    for i in range(len(orbit_list)):
        # start = time.time()
        nodes, edges =node_motif_list(G, G_motif, node,orbit_list[i][0], directed, weighted)
        result_node.extend(nodes)  # 点模体
        result_edge.extend(edges)
    return result_node,result_edge

def edge_in_motif_list(G,G_motif,edge,directed=False,weighted=False):
    """
           计算边edge参与构成模体的数量
           :param G: 图或（网络），可以是有向或无向图，是networkx.Graph() 或 networkx.DiGraph()
           :param G_motif: 模体图,可以有向或者无向但必须与图G一致
           :param edge: 图G中的一条边
           :param directed: 是否是有向图，是有向图值为True，否则为False，默认为无向图False
           :param weighted: 是否具有权重，是否是加权网络，若是值为True，若否值为False，默认为False,符号网络视为加权网络。
           :return :点模体数
    """
    result_node=[]
    result_edge=[]
    orbit_list=edge_orbit(G_motif,directed,weighted)

    for i in range(len(orbit_list)):
        # start = time.time()
        nodes,edges=edge_motif_list(G, G_motif, edge, orbit_list[i][0], directed, weighted)
        result_node.extend(nodes)  # 点模体
        result_edge.extend(edges)

    return result_node,result_edge



if __name__ == "__main__":

    # G = nx.read_edgelist("mydata/USAir97.txt", create_using=nx.Graph())
    # # G_motif = nx.read_edgelist("mydata/motif/directed/motif4_3_0.txt", create_using=nx.DiGraph())
    # g = nx.Graph()
    # # g.add_nodes_from([1, 2, 3, 4,5,6,7])
    # # g.add_edges_from([(1, 2), (1, 3), (2, 4),(2,5),(3,6),(3,7)])
    # # g.add_nodes_from([1, 2, 3, 4])
    # # g.add_edges_from([(1, 2), (1, 3), (3, 4),(2,4)])
    # g.add_nodes_from([1, 2, 3])
    # g.add_edges_from([(1, 2), (1, 3),  (2, 3)])
    # edgelist = list(nx.edges(G))
    # node_list = list(nx.nodes(G))
    # number = 0
    # number1 = 0
    # number2 = 0
    # start3 = time.time()
    # motif_nodes,motif_edges=total_motif_list(G, g, directed=False, weighted=False)
    # end3 = time.time()
    # print('all motif time ', end3 - start3)
    # # print(motif_nodes)
    # print(motif_edges)

    # G = nx.Graph()
    # l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # s = [(1, 2), (1, 3), (1, 4),(2, 4), (3, 4), (4, 5), (3, 5), (5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (8, 9),(2,3)]
    # G.add_nodes_from(l)
    # G.add_edges_from(s)
    # f = nx.Graph()
    # f.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6),(3,7)])
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3,4,5,6,7])
    # g.add_edges_from([(1, 2), (2, 3), (1, 3), (3,4)])
    g.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6),(3,7)])  # M5
    # g.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 3)]) # M5
    # g.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (1, 3)]) # M6
    edge = (4, 5)
    # number = edge_motif_num(G, g, edge)
    # number1 = node_motif_num(G, g, 1)
    # number2 = total_motif_num(g, g)
    # number2,aaa= total_motif_list(f, g)
    # print('摸体数量', number2);
    # print('摸体数量', number1);
    # print('摸体数量', number);

