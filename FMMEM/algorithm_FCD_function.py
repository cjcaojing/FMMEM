# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:04:38 2022

@author: WYW
"""
"""
    各优化算法函数
"""
import numpy as np
# import pandas as pd 
import random  as rd
import copy
import geatpy as ea
# import math
# import operator
# import time

# from random import shuffle

import MFHCD_function as func
# import MFHCD_function_NOF as func
# import MFHCD_function_NOMerge as func
# import MFHCD_function_NOFMerge as func

# =============================================================================
# MM_DECD: 基于模体及隶属度信息的差分进化算法
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: (加权)网络邻接矩阵
# Q_flag: 选择的模块度函数 Q_flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov 
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def DE(DE_info,Compactness,motif_info,lu_bound,pop,fit,n,NP,motif_adj,W,m,Q_flag):
    F,CR = DE_info['F'], DE_info['CR']
    pop_tmp, DE_fit, fit_tmp, betterDE_number = copy.deepcopy(pop), [], copy.deepcopy(fit), 0
    # Mutation【变异操作】
    mutation_pop = copy.deepcopy(pop)
    for i in range(NP):
        # 随机选取三个不相同的个体【a,b,c】
        pick_list = [i for i in range(NP)]
        Xa_index = rd.choice(pick_list)
        pick_list.remove(Xa_index)
        Xb_index = rd.choice(pick_list)
        pick_list.remove(Xb_index)
        Xc_index = rd.choice(pick_list)
        # 构造第i个个体对应的变异个体V
        Vi = mutation_pop[:,Xa_index] + F*(mutation_pop[:,Xb_index] - mutation_pop[:,Xc_index]) 
        # 边界约束检查与修正
        Vi = Vi.astype('int32')
        func.bound_check_revise(Vi,n,lu_bound)
        # 保存变异个体到种群
        Vi = Vi.astype('int32')
        mutation_pop[:,i] = Vi 

    # crossover 【交叉操作】
    crossover_pop = copy.deepcopy(pop)
    # 根据DE算法的交叉操作，以概率CR，保留变异种群mutation_pop中的社区性状
    for i in range(NP):
        # 获得变异个体Vi所有节点所在的社区
        Vi_nodes_c = mutation_pop[:,i]
        # 在[0, n-1]范围内，随机选择一维分量
        rand_j = rd.randint(0, n) #rand_j in 【0，D-1】
        for j in range(n):
            if rd.random()<=CR or j==rand_j:
                # 变异个体i中属于第j个分量所属社区
                Vjc = mutation_pop[j,i]
                # 变异个体i中属于该社区的节点集合
#                all_vjc_nodes = [i for i,vijc in enumerate(Vi_nodes_c) if vijc==Vjc]
                all_vjc_nodes = np.where(Vi_nodes_c==Vjc)
                # 交叉个体i中上述节点的社区均改为Vjc
                for k in all_vjc_nodes:
                    crossover_pop[k,i] = mutation_pop[k,i]     
 
    # choice 【选择操作】
    # 将crossover种群中的优秀个体保留到下一代
    # Q_start = time.process_time()
    func.fit_Qs(Compactness,motif_info, DE_fit, crossover_pop, motif_adj,n, 0,NP,W,m,Q_flag)   #适应度函数值计算
    # Q_end = time.process_time()
    # print("Q_spendti1e=", Q_end - Q_start)
    for index in range(NP):
        if DE_fit[index] > fit_tmp[index]:
            pop_tmp[:,index] = crossover_pop[:,index]    #保存优秀个体
            fit_tmp[index] = DE_fit[index] #保存优秀个体的适应度函数值
            betterDE_number += 1
    return (pop_tmp, fit_tmp, betterDE_number)


# =============================================================================
# SOSFCD: 共生生物搜索算法进行社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: (加权)网络邻接矩阵
# Q_flag: 选择的模块度函数 Q_flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov 
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def SOS(SOS_info, Compactness,motif_info,lu_bound,pop,fit,n,NP,adj,W,m,Q_flag):
    # Mutualism【互利共生】
    mutu_pop = copy.deepcopy(pop)
    mutu_fit = copy.deepcopy(fit)
    better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(mutu_fit)
        best_fit_index = mutu_fit.index(best_fit) 
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 互利共生算法
        Xbest =  mutu_pop[:,best_fit_index]
        Xi = mutu_pop[:,i]
        Xj = mutu_pop[:,j]
        mutual_vector = 0.5 * (Xi + Xj) # 互利共生向量
        BF1=round(1+rd.random())
        BF2=round(1+rd.random())
        # 生成Xinew和Xjnew
        Xinew = Xi + rd.random()*(Xbest - BF1*mutual_vector)
        Xjnew = Xj + rd.random()*(Xbest - BF2*mutual_vector)
        Xinew = Xinew.astype('int32')
        Xjnew = Xjnew.astype('int32')
        # 边界约束检查与修正
        func.bound_check_revise(Xinew,n,lu_bound)
        func.bound_check_revise(Xjnew,n,lu_bound)
        # 适应度函数值计算
        Xinew_fit = func.fit_Q(Compactness,motif_info, Xinew,adj,n,0,W,m,Q_flag)
        Xjnew_fit = func.fit_Q(Compactness,motif_info, Xjnew,adj,n,0,W,m,Q_flag)
        # 选择优秀个体并保留到种群
        if Xinew_fit > mutu_fit[i]:
            mutu_pop[:,i]=Xinew    # 保存优秀个体
            mutu_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
            better_number+=1
        if Xjnew_fit > mutu_fit[j]:
            mutu_pop[:,j]=Xjnew    # 保存优秀个体
            mutu_fit[j] = Xjnew_fit # 保存优秀   个体的适应度函数值
            better_number+=1
            
    # print("mutu_better_number={}".format(better_number))
    # print("mutu_best_Q={}".format(max(mutu_fit)))
    
    # Commensalism【共栖】
    comm_pop = mutu_pop
    comm_fit = mutu_fit
    # better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(comm_fit)
        best_fit_index = comm_fit.index(best_fit) 
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 共栖算法
        Xbest =  comm_pop[:,best_fit_index]
        Xi = comm_pop[:,i]
        Xj = comm_pop[:,j]
        Xinew = Xi + rd.uniform(-1, 1)*(Xbest - Xj)
        Xinew = Xinew.astype('int32')
        # 边界约束检查与修正
        func.bound_check_revise(Xinew,n,lu_bound)
        # 适应度函数值计算
        Xinew_fit = func.fit_Q(Compactness,motif_info,Xinew,adj,n,0,W,m,Q_flag)
        # 选择优秀个体并保留到种群
        if Xinew_fit > comm_fit[i]:
            comm_pop[:,i]=Xinew
            comm_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
            better_number+=1
    # print("comm_better_number={}".format(better_number))
    # print("comm_best_Q={}".format(max(comm_fit)))
   
    # Parasitism【寄生】
    para_pop = comm_pop
    para_fit = comm_fit
    # better_number = 0
    for i in range(NP):
        # 寄生算法
        para_vector =   copy.deepcopy(para_pop[:,i])# 寄生向量
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] # 随机选择一些节点
        # 在约束范围内随机化节点对应的隶属度值
        # para_vector[:,pick] = func.init_pop(len(pick),c,1)[:,:,0] 
        para_vector[pick]=func.init_pop_V1(1,len(pick))[0]
        # 边界约束检查与修正
        func.bound_check_revise(para_vector,n,lu_bound)
        # 适应度函数值计算
        para_vector_fit = func.fit_Q(Compactness, motif_info, para_vector,adj,n,0,W,m,Q_flag)
        # 选择优秀个体并保留到种群
        if para_vector_fit > para_fit[i]:
            para_pop[:,i] = para_vector
            para_fit[i] = para_vector_fit # 保存优秀个体的适应度函数值
            # better_number+=1
    # print("para_better_number={}".format(better_number))
    # print("para_best_Q={}".format(max(para_fit)))
    # 返回当前进化后的种群和适应的函数值
    return (para_pop, para_fit, better_number)

# =============================================================================
# ESOSFCD: 精英共生生物搜索算法进行社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: (加权)网络邻接矩阵
# Q_flag: 选择的模块度函数 Q_flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov 
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def ESOS(SOS_info, Compactness,motif_info, lu_bound, pop, fit_values, n, NP, adj, W, m, Q_flag):
    # Mutualism【互利共生】
    mutu_pop = copy.deepcopy(pop)
    mutu_fit = copy.deepcopy(fit_values)
    better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(mutu_fit)
        best_fit_index = mutu_fit.index(best_fit) 
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 互利共生算法
        Xbest =  mutu_pop[:,best_fit_index] 
        Xi = mutu_pop[:,i]
        Xj = mutu_pop[:,j]
        mutual_vector = 0.5 * (Xi + Xj) # 互利共生向量
        BF1=round(1+rd.random())
        BF2=round(1+rd.random())
        # 生成Xinew和Xjnew
        Xinew = Xi + rd.random()*(Xbest - BF1*mutual_vector)
        Xjnew = Xj + rd.random()*(Xbest - BF2*mutual_vector)
        Xinew = Xinew.astype('int32')
        Xjnew = Xjnew.astype('int32')
        # 边界约束检查与修正
        func.bound_check_revise(Xinew,n,lu_bound)
        func.bound_check_revise(Xjnew,n,lu_bound)
        # 适应度函数值计算
        Xinew_fit = func.fit_Q(Compactness,motif_info, Xinew,adj,n,0,W,m,Q_flag)
        Xjnew_fit = func.fit_Q(Compactness,motif_info, Xjnew,adj,n,0,W,m,Q_flag)
        # 选择优秀个体并保留到种群
        if Xinew_fit > mutu_fit[i]:
            mutu_pop[:,i]=Xinew    # 保存优秀个体
            mutu_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
            better_number+=1
        if Xjnew_fit > mutu_fit[j]:
            mutu_pop[:,j]=Xjnew    # 保存优秀个体
            mutu_fit[j] = Xjnew_fit # 保存优秀个体的适应度函数值
            better_number+=1
    # print("mutu_better_number={}".format(better_number))
    # print("mutu_best_Q={}".format(max(mutu_fit)))
    
    delta = 0.00001 # 最小正数参数
    # Elite_Commensalism【精英共栖】
    comm_pop = mutu_pop
    comm_fit = mutu_fit
    # better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(comm_fit)
        best_fit_index = comm_fit.index(best_fit) 
        fit_max,fit_min = best_fit,min(comm_fit)
        # Xi != Xj != Xk
        ijk_list = [i for i in range(NP)]
        ijk_list.remove(i)
        j = rd.choice(ijk_list) # j!=i
        ijk_list.remove(j)
        k = rd.choice(ijk_list) # k!=i!=j
        # 共栖算法
        Xbest = comm_pop[:,best_fit_index]
        Xi = comm_pop[:,i]
        Xj,fit_j = comm_pop[:,j],comm_fit[j]
        Xk,fit_k = comm_pop[:,k],comm_fit[k]
        lsj = (fit_j - fit_min)/(fit_max - fit_min + delta) #控制参数
        lsk = (fit_k - fit_min)/(fit_max - fit_min + delta) #控制参数
        Xinew = Xbest + lsj*(Xj - Xi) + lsk*(Xk - Xi)
        Xinew = Xinew.astype('int32')
        # 边界约束检查与修正
        func.bound_check_revise(Xinew,n,lu_bound)
        # 适应度函数值计算
        Xinew_fit = func.fit_Q(Compactness,motif_info,Xinew,adj,n,0,W,m,Q_flag)
        # 选择优秀个体并保留到种群
        # better_number += func.save_bestX(comm_pop,Xinew,comm_fit,Xinew_fit,i,motif_info,adj,n,c,W,m,V_flag)
        if Xinew_fit > comm_fit[i]:
            comm_pop[:,i]=Xinew
            comm_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
            better_number+=1
    # print("comm_better_number={}".format(better_number))
    # print("comm_best_Q={}".format(max(comm_fit)))
   
    # Parasitism【寄生】
    para_pop = comm_pop
    para_fit = comm_fit
    # better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(para_fit)
        best_fit_index = para_fit.index(best_fit) 
        fit_max,fit_min = best_fit,min(para_fit)
        Xbest = para_pop[:,best_fit_index]
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 寄生算法
        para_vector = copy.deepcopy(para_pop[:,i])   # 寄生向量
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] # 随机选择一些节点
        # 在约束范围内随机化节点对应的隶属度值
        para_vector[pick]=func.init_pop_V1(1,len(pick))[0]
        # 边界约束检查与修正
        func.bound_check_revise(para_vector,n,lu_bound)
        # # 适应度函数值计算
        para_vector_fit = func.fit_Q(Compactness,motif_info, para_vector,adj,n,0,W,m,Q_flag)
        # # 选择优秀个体并保留到种群
        if para_vector_fit > para_fit[i]:
            para_pop[:,i]=para_vector
            para_fit[i] = para_vector_fit # 保存优秀个体的适应度函数值
        # 利用寄生向量Xpara更新种群中的当前最佳个体Xbest
        lspara = (para_vector_fit - fit_min)/(fit_max - fit_min + delta) #控制参数
        Xbestnew = Xbest + lspara*(para_vector - Xbest) #生成Xbestnew
        Xbestnew = Xbestnew.astype('int32')
        # 边界约束检查与修正
        func.bound_check_revise(Xbestnew,n,lu_bound)
        Xbest_fitnew = func.fit_Q(Compactness,motif_info, Xbestnew,adj,n,0,W,m,Q_flag) #适应度值计算
        # better_number += func.save_bestX(para_pop,Xbestnew,para_fit,Xbest_fitnew,best_fit_index,motif_info,adj,n,c,W,m,V_flag)
        # 选择优秀个体保留到种群
        if Xbest_fitnew > best_fit:
            para_pop[:,best_fit_index]=Xbestnew    # 保存优秀个体
            para_fit[best_fit_index] = Xbest_fitnew # 保存优秀个体的适应度函数值
            better_number+=1
    # 返回当前进化后的种群和适应的函数值
    # print("para_fit={}\n{}".format(max(para_fit),para_fit))
    return (para_pop, para_fit, better_number)


# =============================================================================
# WOA: 鲸鱼搜索算法进行社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: 网络邻接矩阵
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def WOA(WOA_info,Compactness,motif_info,lu_bound,pop,fit,n,NP,adj,W,m,Q_flag):
    a,a_step,b = WOA_info['a'],WOA_info['a_step'],WOA_info['b']
    woa_pop, woa_fit, betterWOA_number = pop, copy.deepcopy(fit), 0
    # 找到目前最好的位置向量
    best_fit = max(woa_fit)
    best_fit_index = woa_fit.index(best_fit) 
    bestX = copy.deepcopy(woa_pop[:,best_fit_index])
    for i in range(NP):
        Xi = woa_pop[:,i]
        Xi_fit = woa_fit[i]
        Xinew = copy.deepcopy(Xi)
        A = 2*a*rd.random() - a
        C = 2*rd.random()
        l = rd.uniform(-1, 1)
        p = rd.random()
        if p<0.5:
            if abs(A)<1:
                # 执行收缩包围策略,采用最优位置的狩猎方式
                D = abs(C*bestX - Xi) 
                Xinew = bestX - A*D
            else:
                #执行搜索猎物策略,采用随个体位置更新，用于探索新的位置信息
                Xrand = woa_pop[:,rd.randint(0, NP-1)]
                D = abs(C*Xrand - Xi) 
                Xinew = Xrand - A*D         
        else:
            #执行螺旋更新策略，进行狩猎行为
            D = abs(bestX - Xi)                 
            Xinew = D*np.exp(b*l)*np.cos(2.0*np.pi*l)+bestX
        # 边界约束检查与修正
        Xinew = Xinew.astype('int32')
        func.bound_check_revise(Xinew,n,lu_bound)
        Xinew_fit = func.fit_Q(Compactness,motif_info, Xinew,adj,n,0,W,m,Q_flag) #适应度值计算
        # 保留优秀个体到种群
        if Xinew_fit>Xi_fit:
            woa_pop[:,i] = Xinew
            woa_fit[i] = Xinew_fit
            betterWOA_number+=1
    # 更新a
    WOA_info['a']-=a_step #更新数据
    return (woa_pop, woa_fit, betterWOA_number)

# =============================================================================
# GA: 遗传算法进行社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: 网络邻接矩阵
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def GA(GA_info,Compactness,motif_info,lu_bound,pop,fit,n,NP,adj,W,m,Q_flag):
    REC_F,MUT_F,pc = GA_info['REC_F'],GA_info['MUT_F'],GA_info['pc']
    GA_pop, GA_fit, betterGA_number = copy.deepcopy(pop), copy.deepcopy(fit), 0
    GA_pop_T =  GA_pop.transpose() #转置，转换成行优先存储
    # 交差
    cross_off_T = ea.recombin(REC_F, GA_pop_T, pc)
    
    # 变异
    ub = np.full(n,lu_bound[0]) #以固定值创建矩阵
    lb = np.full(n,lu_bound[1])
    varType = np.full(n,0) #包含边界
    mutu_off_T = ea.mutate(MUT_F,'RI',cross_off_T,np.vstack((ub,lb,varType)))
    mutu_off = mutu_off_T.transpose() #转置，转换成列优先存储
    
    #选择
    for i in range(NP):
        Xi = mutu_off[:,i]
        #计算Xi_fit
        Xi_fit = func.fit_Q(Compactness,motif_info, Xi,adj,n,0,W,m,Q_flag) #适应度值计算
        #保留优秀个体到种群
        if Xi_fit>GA_fit[i]: #选择
            GA_pop[:,i] = Xi
            GA_fit[i] = Xi_fit
    return (GA_pop, GA_fit, betterGA_number)
    

# =============================================================================
# RFF: 基于农田肥力算法的社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: 网络邻接矩阵
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# url:https://blog.csdn.net/weixin_43821559/article/details/122595030
# =============================================================================
def FFA(FFA_info,Compactness,motif_info,lu_bound,pop,fit,n,NP,adj,W,m,Q_flag):
    betterFF_number = 0 #优秀个体改变计数器
    FF_k,FF_n,alpha,belta,omiga,FF_Q = FFA_info['FF_k'],FFA_info['FF_n'],FFA_info['FF_alpha'],FFA_info['FF_belta'],FFA_info['FF_omiga'],FFA_info['FF_Q']
    M_local,worst = [],{} #农田各区域土壤信息，最差农田区域土壤
    # 第二阶段，确定农田各部分的土壤质量；更新全局内存和局部内存(最优解)。该部分只存储局部中的适应度均值，其对应的实际个体信息仍存储在种群pop中，可通过对应地址取出，可达到节省空间和计算时间的目的
    for s in range(FF_k): #遍历每块儿农田
        s_l = s*FF_n #该块儿农田起始区地址
        s_up = s_l+FF_n #该块儿农田结束区地址
        SFit = sum([fit[x_index] for x_index in range(s_l, s_up)])/FF_n #计算S农田区域的适应度值
        M_local.append(SFit)
    worst['fit'] = min(M_local)
    worst['s'] = M_local.index(worst['fit']) #当前农田中最差区域的块儿编号
    
    # 第三阶段，更新记忆；确定全局内存中解决方案的数量以及局部内存中解决方案的数量，并将解决方案根据这些内存中的大小和适应度值进行排序，确定最坏和最好的部分。
    t = rd.random()
    global_num = 1 + int(t*NP) #全局内存中解的数量
    local_num = 1 + int(t*FF_n) #局部内存中解的数量
    fit_indexs = [i for i in range(NP)]
    fit_dict = dict(zip(fit_indexs,fit))
    fit_dict = dict(sorted(fit_dict.items(), key=lambda x:x[1],reverse=True)) #降序排序
    fit_SortIndex = list(fit_dict.keys())
    M_Global = fit_SortIndex[:global_num]
      
    # 第四阶段，改变农田各部分的土壤质量
    for s in range(FF_k): #遍历每块儿农田
        s_flag = False #用于判定该片农田中是否有个体被更新
        s_l = s*FF_n #该块儿农田起始区地址
        s_up = s_l+FF_n #该块儿农田结束区地址
        if s==worst['s']: 
            # 如果该块儿农田为最差农田，则执行B策略
            for i in range(FF_n):
                h1 = alpha*rd.uniform(-1, 1) #公式9
                X_MGlobal_index = rd.choice(M_Global) #从全局种群内存中随机选一个解决方案(个体)
                Xsi_index = s*FF_n+i #当前农田中第i个解的地址
                Xinew = h1 * (pop[:,Xsi_index] - pop[:,X_MGlobal_index]) + pop[:,Xsi_index] #公式10
                # 边界约束检查与修正
                Xinew = Xinew.astype('int32')
                func.bound_check_revise(Xinew,n,lu_bound)
                Xinew_fit = func.fit_Q(Compactness,motif_info, Xinew, adj,n,0,W,m,Q_flag) #适应度值计算
                if Xinew_fit > fit[Xsi_index]:
                    # 更新worst田中的个体
                    pop[:,Xsi_index] = Xinew
                    fit[Xsi_index] = Xinew_fit
                    betterFF_number+=1
                    s_flag = True
            if s_flag:  # 如果S区域农田有个体被更新，则更新该区域农田的适应度均值
                worst['fit'] = sum([fit[x_index] for x_index in range(s_l, s_up)])/FF_n #更新最差农田区域的适应度值
                M_local[s] = worst['fit'] #更新局部变量中该块儿农田区域的适应度值          
                
        else:   
            #否则，对其余区域则执行A策略
            for i in range(FF_n):
                h2 = belta*rd.random()
                Xsi_index = s*FF_n+i #当前农田中第i个解的地址
                Xui = rd.randint(0, NP-1) #整个搜索空间中现有解的随机解
                Xinew = h2 * (pop[:,Xsi_index] - pop[:,Xui]) + pop[:,Xsi_index] #公式12
                # 边界约束检查与修正
                Xinew = Xinew.astype('int32')
                func.bound_check_revise(Xinew,n,lu_bound)
                Xinew_fit = func.fit_Q(Compactness,motif_info, Xinew, adj,n,0,W,m,Q_flag) #适应度值计算
                if Xinew_fit > fit[Xsi_index]:
                    # 更新S田中的个体
                    pop[:,Xsi_index] = Xinew
                    fit[Xsi_index] = Xinew_fit
                    betterFF_number+=1
                    s_flag = True
            if s_flag: # 如果S区域农田有个体被更新，则更新该区域农田的适应度均值
                M_local[s] = sum([fit[x_index] for x_index in range(s_l, s_up)])/FF_n #更新局部变量中S农田区域的适应度值 
                
    # 第五阶段，土壤组合
    Bset_Global_X = pop[:,fit.index(max(fit))] #全局最优解的解决方案
    for i in range(NP):
        Xinew = copy.deepcopy(pop[:,i]) #初始化xinew
        if FF_Q > rd.random():  #使用全局最优个体信息优化
            Xinew = pop[:,i] + omiga*(pop[:,i] - Bset_Global_X) 
            
        else: #使用局部最优个体信息优化
            Bset_Local_X_S = i//FF_n  #该个体所对应的所在农田块儿编号
            s_l = Bset_Local_X_S*FF_n #该块儿农田起始区地址
            s_up = s_l+FF_n #该块儿农田结束区地址
            S_Local_Xfits = [fit[Xfit_ind] for Xfit_ind in range(s_l, s_up)]  #S号农田中各个体对应的解的适应度值
            S_Best_Local_X_index = s_l + S_Local_Xfits.index(max(S_Local_Xfits)) #S号农田中最优个体对应的绝对地址
            S_Best_Local_X = pop[:,S_Best_Local_X_index] #S号农田中的最优个体
            Xinew = pop[:,i] + rd.random()*(pop[:,i] - S_Best_Local_X)
            
        # 边界约束检查与修正
        Xinew = Xinew.astype('int32')
        func.bound_check_revise(Xinew,n,lu_bound)
        Xinew_fit = func.fit_Q(Compactness,motif_info, Xinew, adj,n,0,W,m,Q_flag) #适应度值计算
        if Xinew_fit > fit[i]:  #保留优秀个体
            pop[:,i] = Xinew
            fit[i] = Xinew_fit
            betterFF_number+=1

    #更新参数信息
    FFA_info['omiga'] = omiga*rd.random()
    
    return (pop, fit, betterFF_number)
            
