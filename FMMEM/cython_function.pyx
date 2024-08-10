# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:56:18 2022

@author: WYW
"""
# Cython —— Q函数
import numpy as np
cimport numpy as np 
cimport cython
from libc.math cimport exp,sqrt
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free, rand, RAND_MAX

# =============================================================================
#     fit_Q: 计算单个个体的模糊重叠社区划分的模块度函数Q值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和,即i节点的度
#     mod: membership,社区划分
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Q(double[:,:] MB, long n, long [:] mod):
    cdef double Q = 0.0
    for i in range(n):
        for j in range(n):
            if mod[i] == mod[j]:
                Q = Q + MB[i,j]
    return Q

# =============================================================================
#     fit_Qg: 计算单个个体的模糊社区划分的模块度函数Qg值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Qg(double[:,:] MB, double[:,:] X, long n, long c):
    cdef double Qg = 0.0
    for k in range(c):
        for i in range(n):
            for j in range(n):
                Qg = Qg + MB[i,j]*X[k,i]*X[k,j]
    return Qg

# # =============================================================================
# #     fit_Qgi: 计算单个个体中单个节点的模糊社区划分的增量模块度值
# #     X: 种群中的个体
# #     adj: 加权邻接矩阵
# #     n: 网络节点数目
# #     c: 社区划分的数目
# #     W: 加权网络邻接矩阵的总权值之和
# #     m: 邻接矩阵各行的权值之和
# # =============================================================================
# @cython.boundscheck(False) 
# @cython.wraparound(False) 
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef double fit_deltaQgi(long i, double[:,:] u, double[:] ui, double[:,:] uinew, double[:,:] B, long[:,:] adj, long n, long c, long W):
#     cdef double deltaQgi = 0.0
#     cdef double* uki0
#     # 计算uki0
#     #创建一维矩阵
#     uki0 = <double*>malloc(c * sizeof(double))
#     for k in range(c):
#         uki0[k]=0.0 #初始化uki0
#         for j in range(n):
#             if j != i:        
#                 uki0[k] += (B[i,j]*u[k,j])
#         uki0[k]=-uki0[k]/B[i,i]
#     # 计算deltaQgi
#     for k in range(c):
#         deltaQgi += ((uinew[k] - ui[k]) * (uinew[k] + ui[k] - 2*uki0[k]))
            
#     deltaQgi = deltaQgi*B[i,i]/W
#     return deltaQgi

# =============================================================================
#     fit_Qgi: 计算单个个体中单个节点的模糊社区划分的增量模块度值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_deltaQgi(double[:] uki0, double[:] ui, double[:] uinew, double Bii, long c, long W):
    cdef double deltaQgi = 0.0
    # 计算deltaQgi
    for k in range(c):
        deltaQgi += ((uinew[k] - ui[k]) * (uinew[k] + ui[k] - 2*uki0[k]))
    deltaQgi = deltaQgi*Bii/W
    return deltaQgi

# =============================================================================
#     fit_Qgi: 计算单个个体中单个节点的模糊社区划分的增量模块度值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double getUki0(long i, double[:,:] u, double[:] uki0, double[:,:] B, long n, long c):
    # 计算uki0
    for k in range(c):
        for j in range(n):
            if j != i:        
                uki0[k] += (B[i,j]*u[k,j])
        uki0[k]=-uki0[k]/B[i,i]

# =============================================================================
#     fit_Ql: 计算Ql值
#     adj: 加权邻接矩阵
#     nodes: 社区中的节点列表
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和,即i节点的度
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Ql(long[:,:] adj, long[:] nodes, long n, long W, long[:] m):
    cdef double Ql = 0.0
    for i_index in range(n):
        for j_index in range(n):
            Ql = Ql + (adj[nodes[i_index],nodes[j_index]] - (m[nodes[i_index]]*m[nodes[j_index]])*1.0/W)
    return Ql*1.0/W

# =============================================================================
#     fit_deltaQ: 计算deltaQ值,将i从nodes集合中移后Q值变化量，移后减去移前
#     adj: 加权邻接矩阵
#     nodes: 包含i节点的社区中的节点列表
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和,即i节点的度
#     flag: 0:移出，1：移入
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_deltaQ(long i, long[:,:] adj, long[:] nodes, long n, long W, long[:] m, long flag):
    cdef double deltaQ = 0.0
    for i_index in range(n):
        deltaQ += (adj[nodes[i_index],i] - (m[nodes[i_index]]*m[i])*1.0/W)
    for j_index in range(n):
        deltaQ += (adj[i,nodes[j_index]] - (m[i]*m[nodes[j_index]])*1.0/W)   
    if flag==0:
        deltaQ = -deltaQ
    return deltaQ*1.0/W

# =============================================================================
#     fit_Qc: 计算单个个体的模糊重叠社区划分的模块度函数Qc值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Qc(double[:,:] MB, double[:,:] X, long n, long c):
    cdef double minu_kij,temp,GG
    cdef double Qc = 0.0
    for i in range(n):
        for j in range(n):
            # 计算GG
            GG= 0.0
            for k in range(c):
                temp=X[k,i] if X[k,i]<X[k,j] else X[k,j]
                if temp > GG: GG = temp     
            GG = sqrt(GG)
            # 根据Gg求Qc
            Qc = Qc + MB[i,j]*GG
    return Qc

# =============================================================================
#     fit_deltaQci: 计算单个个体的模糊重叠社区划分的模块度函数Qc值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_deltaQci(double[:,:] MB, double[:,:] X, double[:,:] Xnew, long n, long c, long i):
    cdef double GG_temp,GGnew_temp,GG, GGnew, deltaQci=0.0
    for j in range(n):
        # 计算GG
        GGnew=0.0
        GG=0.0
        for k in range(c):
            GGnew_temp=Xnew[k,i] if Xnew[k,i]<Xnew[k,j] else Xnew[k,j]
            if GGnew_temp > GGnew: GGnew = GGnew_temp     
        GGnew = sqrt(GGnew)
        
        for k in range(c):
            GG_temp=X[k,i] if X[k,i]<X[k,j] else X[k,j]
            if GG_temp > GG: GG = GG_temp     
        GG = sqrt(GG)
        # 根据Gg求Qc
        deltaQci = deltaQci + (MB[i,j]*(GGnew-GG))
    return deltaQci



# =============================================================================
#     fit_Qov: 计算单个个体的模糊重叠社区划分的模块度函数Qg值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fit_Qov(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m):
    cdef double Qov = 0.0, wSum, lSum
    cdef double** r
    cdef double** w
    cdef int* pointk
    cdef int index, nk
    
    # 创建二维矩阵
    r = <double**>malloc(n * sizeof(double*))
    for i in range(n):
        r[i] = <double*>malloc(n * sizeof(double))
#        memset(r[i], 0, n * sizeof(double))
        
    w = <double**>malloc(n * sizeof(double*))
    for i in range(n):
        w[i] = <double*>malloc(n * sizeof(double))
#        memset(w[i], 0, n * sizeof(double))
        
    for k in range(c):
        #获得第K个社区的所有节点标号(非零元素的下标)
        nk = 0
        for i in range(n):
            if X[k,i] > 0.0:
                nk = nk + 1       
        pointk = <int*>malloc(nk * sizeof(int))
        index = 0
        for i in range(n):
            if X[k,i] > 0.0:
                pointk[index] = i
                index = index + 1 

        # 对矩阵赋值
        for i in range(nk):
            for j in range(nk):
                r[i][j] = 1.0/((1+exp(-(60*X[k,pointk[i]]-30)))*(1+exp(-(60*X[k,pointk[j]]-30))))  
        for i in range(nk):
            for j in range(nk):
                # 求和
                wSum=0
                for t in range(nk):
                    wSum = wSum + r[i][t]
                lSum=0
                for t in range(nk):
                    lSum = lSum + r[t][j]
                # 计算w
                w[i][j] = wSum*lSum*1.0/(nk*nk)
        # 计算Qov值
        for i in range(nk):
            for j in range(nk):
                Qov = Qov + (r[i][j]*adj[pointk[i],pointk[j]] - w[i][j]*(m[0,pointk[i]]*m[0,pointk[j]])*1.0/W)
                
        free(pointk)
    # 释放内存
    for i in range(n):
        free(r[i])
        free(w[i])
    free(r)
    free(w)
    Qov = Qov*1.0/W
    return Qov      

# =============================================================================
#     getMEM_adj: 获得Xi的模体，边及隶属度融合的权重矩阵
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double getEdgeW(double[:,:] X, long[:,:] me_adj, long[:,:] node_set, long[:,:,:] edge_set,long i, long j, long c, long lenSet, long M_n, long M_e):
    cdef double m_W = 0.0, tmp_m, sum_membership, sum_w
    cdef int M_c, c_max
    # 创建一个一维数组, 并以0值初始化
    i_M_c = <int*>malloc(c * sizeof(int*))
    for i_c in range(c):
        i_M_c[i_c]=0
    for m_index in range(lenSet): #第m个模体
        # 模体 M1
        # 获得该模体M当前所在的社区
        M_c = 0 #初始化M_c值
        for i_index in range(M_n): #模体中的第i个节点
            membership = 0.0 # 初始化membership
            c_max=0
            for c_index in range(c):#节点i所在的第c个社区
                tmp_m = X[c_index,node_set[i_index,m_index]]
                if tmp_m > membership:
                    membership = tmp_m
                    c_max=c_index         
            i_M_c[c_max] += 1
        for c_index in range(c): #获得模体M所在社区
            if i_M_c[c_index] > M_c:
                M_c = c_index
            i_M_c[c_index] = 0
        # 计算该模体的隶属度之和
        sum_membership=0.0
        for i_index in range(M_n):
            sum_membership += X[M_c,node_set[i_index,m_index]]
        # 计算该模体的权重之和
        sum_w = 0.0 #初始化sum_w值
        for e_index in range(M_e):
            sum_w += me_adj[edge_set[e_index,0,m_index],edge_set[e_index,1,m_index]]
        # 计算该边的融合权重
        m_W += (me_adj[i,j]/sum_w*sum_membership)
    free(i_M_c)
    return me_adj[i,j] + m_W

# =============================================================================
#     renewMUs: 更新MU隶属度矩阵
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
cpdef renewMUs(double[:,:] MU, double[:,:] X, long[:,:] mno_arr, long r, long c, long D):
    cdef double mmem 
    for M in range(r):
        for k in range(c):
            mmem = 0.0  #模体M对社区k的隶属度
            for d in range(D):
                mmem = mmem + X[k,mno_arr[M,d]]
            MU[k,M] = mmem/D 
                
# =============================================================================
#     getCDi: 获得节点i的CD(i)值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef getCDI(double[:] result, double[:] attr_cki_arr, long[:,:] mno, double[:,:] MU, long[:] ms, long m_len, double[:] mcnoWij_arr, 
             long[:] j_nodes_c_set, long jsetc_len, long[:,:] me_adj, long i, long imem_cno, long D):
    cdef double wij_sum = 0.0, ijcno_attr=0.0, attr_sum=0.0, maxAttr = 0.0
    cdef int jcno, no
    # print("======wij========",wij)
    if m_len > 0:
        for jsetc_index in range(jsetc_len):
            ijcno_attr = 0.0
            ijcno_attr+=mcnoWij_arr[jsetc_index]
                
            jcno = j_nodes_c_set[jsetc_index]
            for m_index in range(m_len):
                # print("i={},j_cno={},mno={}".format(i,jcno,mno[ms[m_index]])
                for D_index in range(D):
                    # 查询模体节点是否在jc社区中
                    no = mno[ms[m_index],D_index] #模体节点
                ijcno_attr+=MU[jcno,ms[m_index]]       
            
            #查找最大社区的连接强度
            if ijcno_attr>maxAttr: maxAttr=ijcno_attr
            
            attr_cki_arr[jsetc_index] = ijcno_attr
            attr_sum+=ijcno_attr
            if jcno == imem_cno:
                wij_sum = ijcno_attr
            # print("i={},j_cno={},wij={},wij_sum={},Cnei={}".format(i,j_nodes_c_set[jsetc_index],wij,wij_sum,Cnei))
    # 如果节点 i 与同一社区的邻居节点之间的权重总和 <= 其与所有邻居节点权重总和的平均值，则返回 i
    if wij_sum < maxAttr:
        result[0] = -1.0
    else:
        result[0] = 1.0
    result[1] = attr_sum

# cpdef getCDI(double[:] result,double[:] attr_cki_arr,long[:,:] mno, long[:] mw, double[:,:] MU, long[:] ms, long m_len, long[:] ijc_nodes_arr, long ijc_len,
#                     long[:] j_nodes, long j_len, long[:] j_nodes_c_arr, long[:] j_nodes_c_set, long jsetc_len, long[:,:] motif_adj, 
#                     long[:,:] me_adj, long Cnei, long i, long i_node_c, long D):
#     cdef double wij_sum = 0.0, wij = 0.0, meij=0.0, meij_sum=0.0
#     cdef int jcno, no, mno_jc_w
#     # 计算i与其所有邻居节点的权值总和
#     for j_index in range(j_len):
#         wij_sum = wij_sum + me_adj[i,j_nodes[j_index]] #边权重+模体权重
#     # 计算i所在社区的邻居节点权值
#     for j_index1 in range(ijc_len):
#         wij = wij + me_adj[i,ijc_nodes_arr[j_index1]]
#     # print("======wij========",wij)
#     if m_len > 0:
#         for jsetc_index in range(jsetc_len):
#             jcno = j_nodes_c_set[jsetc_index]
#             meij = 0.0
#             for m_index in range(m_len):
#                 # print("i={},j_cno={},mno={}".format(i,jcno,mno[ms[m_index]])
#                 for D_index in range(D):
#                     # 查询模体节点是否在jc社区中
#                     no = mno[ms[m_index],D_index] #模体节点
#                 meij = meij + MU[jcno,ms[m_index]]
                
#             attr_cki_arr[jsetc_index] = meij
#             meij_sum = meij_sum + meij
#             wij_sum = wij_sum + meij
#             if jcno == i_node_c:
#                 wij = wij + meij
#             # print("i={},j_cno={},wij={},wij_sum={},Cnei={}".format(i,j_nodes_c_set[jsetc_index],wij,wij_sum,Cnei))
#     # 如果节点 i 与同一社区的邻居节点之间的权重总和 <= 其与所有邻居节点权重总和的平均值，则返回 i
#     cd_i = wij - wij_sum*1.0/Cnei
#     result[0] = cd_i
#     result[1] = meij_sum

# cpdef getCDI(double[:] result,double[:] attr_cki_arr,long[:,:] mno, long[:] mw, double[:,:] MU, long[:] ms, long m_len, long[:] ijc_nodes_arr, long ijc_len,
#                     long[:] j_nodes, long j_len, long[:] j_nodes_c_arr, long[:] j_nodes_c_set, long jsetc_len, long[:,:] motif_adj, 
#                     long[:,:] me_adj, long Cnei, long i, long i_node_c, long D):
#     cdef double wij_sum = 0.0, wij = 0.0, meij=0.0, meij_sum=0.0
#     cdef int jcno, no, mno_jc_w
#     # 计算i与其所有邻居节点的权值总和
#     for j_index in range(j_len):
#         wij_sum = wij_sum + me_adj[i,j_nodes[j_index]] #边权重+模体权重
#     # 计算i所在社区的邻居节点权值
#     for j_index1 in range(ijc_len):
#         wij = wij + me_adj[i,ijc_nodes_arr[j_index1]]
#     # print("======wij========",wij)
#     if m_len > 0:
#         for jsetc_index in range(jsetc_len):
#             jcno = j_nodes_c_set[jsetc_index]
#             meij = 0.0
#             for m_index in range(m_len):
#                 mno_jc_w = 0 #该模体与jc社区的连接权重
#                 # print("i={},j_cno={},mno={}".format(i,jcno,mno[ms[m_index]])
#                 for D_index in range(D):
#                     # 查询模体节点是否在jc社区中
#                     no = mno[ms[m_index],D_index] #模体节点
#                     if no != i: 
#                         if j_nodes_c_arr[binarySearch1(j_nodes, j_len, no)] == jcno: #若该节点在社区jc中
#                             mno_jc_w = mno_jc_w + motif_adj[i, no]
#                 meij = meij + MU[jcno,ms[m_index]] * mno_jc_w*1.0/mw[ms[m_index]]
                
#             attr_cki_arr[jsetc_index] = meij
#             meij_sum = meij_sum + meij
#             wij_sum = wij_sum + meij
#             if jcno == i_node_c:
#                 wij = wij + meij
#             # print("i={},j_cno={},wij={},wij_sum={},Cnei={}".format(i,j_nodes_c_set[jsetc_index],wij,wij_sum,Cnei))
#     # 如果节点 i 与同一社区的邻居节点之间的权重总和 <= 其与所有邻居节点权重总和的平均值，则返回 i
#     cd_i = wij - wij_sum*1.0/Cnei
#     result[0] = cd_i
#     result[1] = meij_sum

# =============================================================================
#     getAttrSum: 获得节点i对其各个邻接社区的隶属度以及总隶属度
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double getAttrSum(long[:,:] mno,long[:] mw,double[:] arrt_cks_i, long wij_sum, double[:,:] MU,double[:] ic_memberships, long[:] ms, long m_len, long[:] j_nodes, long j_len,
                        long[:] j_nodes_c_set, long[:] j_nodes_c_arr, long jsetc_len, long[:,:] motif_adj, long[:,:] me_adj, long i, long D):
    cdef double arrt_sum = 0.0, arrt_cki=0.0, wij_m=0.0
    # print("==============")
    if m_len > 0:
        for jsetc_index in range(jsetc_len):
            arrt_cki = 0.0
            for m_index in range(m_len):
                wij_m = 0.0 #该模体与jc社区的连接权重
                for D_index in range(D):
                    # 查询模体节点是否在jc社区中
                    if j_nodes_c_arr[binarySearch1(j_nodes, j_len, mno[ms[m_index],D_index])] == j_nodes_c_set[jsetc_index]: #若该节点在社区jc中
                        wij_m = wij_m + motif_adj[i, mno[ms[m_index],D_index]]
                arrt_cki = arrt_cki + MU[j_nodes_c_set[jsetc_index],ms[m_index]] * (wij_m)/mw[ms[m_index]]
            arrt_sum = arrt_sum + arrt_cki
            arrt_cks_i[jsetc_index] = arrt_cki
            # print("i={},j_cno={},wij_sum={},arrt_cki={},arrt_sum={}".format(i,j_nodes_c_set[jsetc_index],wij_sum,arrt_cki,arrt_sum))
    return wij_sum + arrt_sum

# =============================================================================
#     getAttr: 获得节点i的Attr(ck,i)值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double getAttrCki(long wij_sum, double mem_sum, double arrt_cki, long[:] ck_nodes, long ckj_len, long[:,:] me_adj, long i):
    cdef double wij_ck = 0.0, arrtCki = arrt_cki
    for ckj_index in range(ckj_len):
        wij_ck = wij_ck + me_adj[i,ck_nodes[ckj_index]]
    return (arrtCki + wij_ck)*1.0/(mem_sum + wij_sum)


# =============================================================================
#     bound_check_revise: 边界约束检查与修正
#     X: 个体隶属度矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef bound_check_revise(long[:] X, long n, long low, long up):
    for i in range(n):
        if X[i] < low:
            X[i] = -X[i]
            if X[i] > up:
                X[i] = up
        elif X[i] > up:
            X[i] = 2*up - X[i]
            if X[i] < low:
                X[i] = low
            
# =============================================================================
#     fai_m: faim值计算
#     faim_flag: 选用方式【1，2】 1:计算整个社区中的最小模体电导，2:计算整个社区划分中的平均模体电导
#     paration: 标签型社区划分
#     parationSet: 由社区划分标签所组成的集合
#     parationSet_len: parationSet集合的长度
#     node_motif_num_list: 节点的点模体数量列表
#     total_motif: 网络中所有的模体集合[[1,2,3],[2,3,4],[2,4,5]]
#     n: 节点数量
#     M_: 模体阶数
#     return faim值
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fai_m(long faim_flag, dict Xpartition, dict node_mdict, long[:] node_motif_num_list, long node_motif_numSum, long motif_total_num, long n, long M_):
    cdef int partitionlen,intra_nnum=0,inter_mnum=0,other_mnum=0
    cdef double faim,min_faim=1.0,sum_faim=0.0
    
    partitionlen = len(Xpartition)
    if partitionlen != n and partitionlen != 1:  #所有节点一个社区 和 每个节点一个社区的情况为1.0
        for cno in Xpartition.keys():
            cno_nodes = Xpartition[cno]
            intra_nnum = 0 #社区内模体中节点的数量
            for i in cno_nodes:
                intra_nnum+=node_motif_num_list[i]
            if intra_nnum !=0 and len(cno_nodes)>=M_:
                #获得该社区中节点所参与构成的模体集合
                cno_ms = set()
                for i in cno_nodes:
                    ims = node_mdict[i]
                    for im in ims:
                        cno_ms.add(im)
                
                #计算该社区中的模体数量和社区内的模体数量
                inter_mnum = 0 #社区间模体数量/被切割的模体数量
                for m in cno_ms:
                    if not set(m)<=set(cno_nodes): #社区间模体
                        inter_mnum+=1
                if inter_mnum != motif_total_num:
                    other_nnum = node_motif_numSum - intra_nnum  #该社区外模体中节点的数量
                    minCno_nnum = intra_nnum if intra_nnum<other_nnum else other_nnum
                    #计算faim
                    if minCno_nnum==0:
                        faim=1.0
                    else:
                        faim = inter_mnum/minCno_nnum
                else:
                    faim=1.0
            else:
                faim=1.0
            sum_faim+=faim
            if min_faim>faim:
                min_faim=faim        
        
        # 返回结果
        if faim_flag == 2:
            return sum_faim / partitionlen #fai_m2
        elif faim_flag == 1:
            return min_faim
        elif faim_flag == 3:
            return sum_faim
        else:
            return 1.0
    else:
        return 1.0

# =============================================================================
#     optMem: 优化i节点的隶属度矩阵
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double optImem(double Qg, double[:] i_mem, double[:] itmp_mem, double[:] uki0, double Bii, long c, long W, long maxcno, long[:] neicno_arr, long neicno_len, long limit):
    cdef int count=0,opt_flag=1, index=0, jcno #优化停止标志,计数器
    cdef double itmp_mem_sum=0.0, max_Qg=Qg
    cdef double rd_V
    while opt_flag==1:
        for index in range(neicno_len):
            jcno = neicno_arr[index]
            itmp_mem[jcno] = i_mem[jcno]
        rd_V = random_uniform()/3
        # print("rdV=",rd_V)
        symbol = random_uniform()
        if symbol>0.5:
            itmp_mem[maxcno] += rd_V
            itmp_mem_sum = 1+rd_V
        else:
            itmp_mem[maxcno] -= rd_V
            itmp_mem_sum = 1-rd_V
        for index in range(neicno_len):
            jcno = neicno_arr[index]
            itmp_mem[jcno]/=itmp_mem_sum
        if itmp_mem[maxcno] >1.0: itmp_mem[maxcno]=1.0
        if itmp_mem[maxcno] <0.0: itmp_mem[maxcno]=0.0
        itmp_mem_sum = 0.0
        for index in range(neicno_len):
            jcno = neicno_arr[index]
            itmp_mem_sum+=itmp_mem[jcno]
        for index in range(neicno_len):
            jcno = neicno_arr[index]
            itmp_mem[jcno]/=itmp_mem_sum
            
        deltaQgi = fit_deltaQgi(uki0, i_mem, itmp_mem, Bii, c, W)
        if deltaQgi > 0:
            # 保留变异值
            max_Qg += deltaQgi
            for index in range(neicno_len):
                jcno = neicno_arr[index]
                i_mem[jcno] = itmp_mem[jcno]
            count = 0
        else:
            count += 1
        if count > limit: opt_flag=0 
    return max_Qg

# =============================================================================
#     binarySearch: 二分查找数据,非递归实现
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int binarySearch(int* arr1, int r1, int x1):
    cdef int mid, x=x1, l=0, r=r1
    cdef int* arr=arr1
    while(l<=r):
        mid = int(l + (r - l)/2)
        if x < arr[mid]: 
            r = mid - 1
        elif x > arr[mid]: 
            l = mid + 1
        else: 
            return mid 
    return -1 

# =============================================================================
#     binarySearch: 线性查找数据
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int binarySearch1(long[:] arr, int r1, int x1):
    cdef int index=0,
    for i in range(r1):
        if arr[i] == x1:
            return i
        index = index + 1
    return -1

# =============================================================================
# 用于产生随机数
# =============================================================================
cdef double random_uniform():
    cdef double r = rand()
    return r/float(RAND_MAX)

# =============================================================================
#     getCno_membership: 计算节点隶属度
#     return maxattrCno： 返回最大社区编号
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void getCno_membership(long i ,double[:,:] MB, dict Cno_membership, dict Xpartition, list j_nodes_setc):
    cdef double Compactness=0.0, compactness_sum=0.0
    
    for jcno in j_nodes_setc:
        Compactness=0.0
        for j in Xpartition[jcno]:
            Compactness+=MB[i,j]
        Compactness=0.0 if Compactness<0 else Compactness
        Cno_membership[jcno] = Compactness
        compactness_sum+=Compactness
        
    for jcno in j_nodes_setc:
        if compactness_sum>0.0:
            Cno_membership[jcno]=Cno_membership[jcno]/compactness_sum

# =============================================================================
#     getCno_membership: 计算节点隶属度
#     return maxattrCno： 返回最大社区编号
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int  getCno_P(dict nodes_mem, list node_motifs, list j_nodes_setc, dict Cno_P):
    cdef double Compactness=0.0, vtmp=0.0, compactness_sum=0.0
    cdef int maxattrCno=0
    
    for jcno in j_nodes_setc:
        Compactness=0.0
        for m in node_motifs:
            for mi in m:
               Compactness+=nodes_mem[mi][jcno]
        # Compactness/=3 #所有模体对该社区的隶属程度总和
        Cno_P[jcno]=Compactness
        compactness_sum+=Compactness
        if Compactness>vtmp: 
            maxattrCno=jcno
            vtmp=Compactness
        
    for jcno in j_nodes_setc:
        Cno_P[jcno]=Cno_P[jcno]/compactness_sum
    return maxattrCno


cpdef int getCno_mem(long i ,double[:,:] MB, dict Cno_membership, dict Xpartition, list j_nodes_setc):
    cdef double Compactness=0.0, vtmp=0.0, compactness_sum=0.0
    cdef int maxattrCno=0
    
    for jcno in j_nodes_setc:
        Compactness=0.0
        for j in Xpartition[jcno]:
            Compactness+=MB[i,j]
        Compactness=0.0 if Compactness<0 else Compactness
        Cno_membership[jcno] = Compactness
        compactness_sum+=Compactness
        if Compactness>vtmp: 
            maxattrCno=jcno
            vtmp=Compactness
        
    for jcno in j_nodes_setc:
        Cno_membership[jcno]=Cno_membership[jcno]/compactness_sum
    return maxattrCno



    
        
        

