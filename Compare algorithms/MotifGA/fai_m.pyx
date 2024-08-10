"""
Created on Tue Sep  6 08:26:23 2022

@author: WYW
"""

# Cython ——fai_m函数
cimport numpy as np 
cimport cython
from libc.stdlib cimport malloc, free

# =============================================================================
#     fai_m: fai_m值计算
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double fai_m(long faim_flag, long[:] partition, long[:] partitionSet, long partitionSet_len, long[:] node_motif_num_list, long[:,:] total_motif, long total_motif_len, long n, long M_):
    cdef int* list1
    cdef int* cno_inum
    cdef int** cno_iset
    cdef double* list2
    cdef double tmp_vlue, list2_minvalue, list2_sum, l1
    cdef int im_num, im_sum_num=0, cno, temp_value, shjian_mt, sqnei_s, sqnei_sfei, flag, fei_flag, i, list2_len
    
    if partitionSet_len != n and partitionSet_len != 1:  #所有节点一个社区 和 每个节点一个社区的情况为1.0
        cno_inum = <int*>malloc(n * sizeof(int*)) #各社区中节点的数目
        for i in range(n):
            cno_inum[i] = 0
            
        for c_index in range(n):
            cno = partition[c_index]
            cno_inum[cno] = cno_inum[cno] + 1
            
        cno_iset = <int**>malloc(n * sizeof(int*)) #各社区对应的节点集合，集合中节点按节点标号大小正序排列
        for cno_index in range(partitionSet_len):
            cno = partitionSet[cno_index]
            cno_iset[cno] = <int*>malloc(cno_inum[cno] * sizeof(int))
            for i_index in range(cno_inum[cno]):
                cno_iset[cno][i_index] = -1 #列表中的值均初始化为-1
                        
        for i in range(n):
            cno = partition[i]
            for i_index in range(cno_inum[cno]):
                if cno_iset[cno][i_index] == -1:
                    cno_iset[cno][i_index] = i    #将节点添加到所属社区中
                    break
        
        list1 = <int*>malloc(n * sizeof(int*)) #单个社区内点模体之和列表
        list2 = <double*>malloc(n * sizeof(double*)) #社区与其余节点  间 的模体
        list2_len = 0 #list2有效大小
        for c_index in range(n):
            list1[c_index]=0
            list2[c_index]=0.0
        
        for cno_index in range(n):  #c_index==i
            cno = partition[cno_index]
            im_num = node_motif_num_list[cno_index]
            list1[cno] = list1[cno] + im_num
            im_sum_num = im_sum_num + im_num

                
        for cno_index in range(partitionSet_len):             
            cno = partitionSet[cno_index]
            temp_value = im_sum_num-list1[cno]
            if temp_value < list1[cno]:
                list1[cno] = temp_value      #最小的单个社区  与  其余节点的点模体作为分母
        im_sum_num=0
        
        for cno_index in range(partitionSet_len):  
            cno = partitionSet[cno_index]
            l1 = list1[cno]
            if l1 != 0 and cno_inum[cno] >= M_:
                sqnei_s = 0
                sqnei_sfei = 0
                for mt_index in range(total_motif_len):
                    flag = 0   #模体中节点在当前社区的数量
                    fei_flag = 0   #模体中节点在非当前社区的数量
                    for i_index in range(M_):
                        i = total_motif[mt_index, i_index]   #获得模体中的节点
                        if binarySearch(cno_iset[cno], cno_inum[cno]-1, i) > -1:
                        # if cno_iset[cno][i_index] > -1:
                            flag = flag + 1                   
                        else:
                            fei_flag = fei_flag + 1
                    if flag == M_:
                        sqnei_s = sqnei_s + 1   #若该模体的所有节点均在该社区中，及该模体存在于cno社区中
                    if fei_flag == M_:
                        sqnei_sfei = sqnei_sfei + 1   #若该模体的所有节点均在非该社区中，及该模体存在于非cno社区中
                shjian_mt = total_motif_len - sqnei_s - sqnei_sfei #社区间模体数量
                if shjian_mt != total_motif_len:
                    list2[cno] = shjian_mt/l1
                    list2_len = list2_len + 1
            else:
                list2[cno] = 1.0
                list2_len = list2_len + 1
               
        list2_minvalue = 1.0
        list2_sum = 0.0
        for cno_index in range(partitionSet_len):
            cno = partitionSet[cno_index]
            tmp_vlue = list2[cno]
            if tmp_vlue < list2_minvalue:
                list2_minvalue = tmp_vlue
            list2_sum = list2_sum + tmp_vlue
        # 释放内存
        free(cno_inum)
        free(list1)
        free(list2)
        for i in range(partitionSet_len):
            free(cno_iset[partitionSet[i]])
        free(cno_iset)
        # 返回结果
        if faim_flag == 2:
            return list2_sum / list2_len #fai_m2
        elif faim_flag == 1:
            return list2_minvalue
        elif faim_flag == 3:
            return list2_sum
        else:
            return 1.0
    else:
        return 1.0

# =============================================================================
#     binarySearch: 二分查找数据,非递归实现
#     return x 在 arr 中的索引，如果不存在返回 -1
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cdef binarySearch(int* arr1, int r1, int x1):
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

