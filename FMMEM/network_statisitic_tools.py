# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022

@author: WYW
"""
"""
    FFM_CD_main_v1_NMM
   使用各种优化算法，基于模体的加权网络的社区检测
"""
import os
import numpy as np

Path = r"email/30_Q_log.txt"
lfr_path = r"network2/50_Q_log.txt"
print("================================Q值==================================")
Q_file_path = r"logs/QW/" + Path
Q_NWMMlist = []
# Q_MNMMlist = []
Q_NMMlist = []
# Q_NOMMlist = []
with open(Q_file_path, mode='r') as f:
        datas = f.readlines()
        for line in datas:
            if line.find('NWMM') != -1:
                Q_NWMMlist.append(line.split('=')[1][:-1])
            # elif line.find('MNMM') != -1:
            #     Q_MNMMlist.append(line.split('=')[1][:-1])
            # elif line.find('NMM') != -1:
            #     Q_NMMlist.append(line.split('=')[1][:-1])
            # elif line.find('NOMM') != -1:
            #     Q_NOMMlist.append(line.split('=')[1][:-1])
# mean      
Q_NWMM_mean = np.asarray(Q_NWMMlist, dtype=float).mean()
# Q_MNMM_mean = np.asarray(Q_MNMMlist,dtype=float).mean()
# Q_NMM_mean = np.array(Q_NMMlist,dtype=float).mean()
# Q_NOMM_mean = np.array(Q_NOMMlist, dtype=float).mean()
# std
Q_NWMM_std = np.asarray(Q_NWMMlist, dtype=float).std()
# Q_MNMM_std = np.asarray(Q_MNMMlist,dtype=float).std()
# Q_NMM_std = np.array(Q_NMMlist,dtype=float).std()
# Q_NOMM_std = np.array(Q_NOMMlist, dtype=float).std()
# max
Q_NWMM_max = max(Q_NWMMlist)
# Q_MNMM_max = max(Q_MNMMlist)
# Q_NMM_max = max(Q_NMMlist)
# Q_NOMM_max = max(Q_NOMMlist)
# min
Q_NWMM_min = min(Q_NWMMlist)
# Q_NMM_min = min(Q_NMMlist)
# len
Q_NWMM_len = len(Q_NWMMlist)
print("============={}===================".format(Q_NWMM_len))
print("Q_NWMM_mean={}，Q_NWMM_std={}，Q_NWMM_max={}，Q_NWMM_min={}".format(Q_NWMM_mean,Q_NWMM_std,Q_NWMM_max,Q_NWMM_min))
# print("Q_MNMM_mean={}，Q_MNMM_std={}，Q_MNMM_max={}".format(Q_MNMM_mean,Q_MNMM_std,Q_MNMM_max))
# print("Q_NMM_mean={}，Q_NMM_std={}，Q_NMM_max={}，Q_NMM_min={}".format(Q_NMM_mean,Q_NMM_std,Q_NMM_max,Q_NMM_min))
# print("Q_NOMM_mean={}，Q_NOMM_std={}，Q_NOMM_max={}".format(Q_NOMM_mean,Q_NOMM_std,Q_NOMM_max))

# print("================================NMI值==================================")
# NMI_file_path = r"logs/NMIW/" + Path
# NWMMlist = []
# # MNMMlist = []
# NMMlist = []
# # NOMMlist = []
# with open(NMI_file_path, mode='r') as f:
#         datas = f.readlines()
#         for line in datas:
#             if line.find('NWMM') != -1:
#                 NWMMlist.append(line.split('=')[1][:-1])
#             # elif line.find('MNMM') != -1:
#             #     MNMMlist.append(line.split('=')[1][:-1])
#             # elif line.find('NMM') != -1:
#             #     NMMlist.append(line.split('=')[1][:-1])
#             # elif line.find('NOMM') != -1:
#             #     NOMMlist.append(line.split('=')[1][:-1])
# # mean      
# NWMM_mean = np.asarray(NWMMlist, dtype=float).mean()
# # MNMM_mean = np.asarray(MNMMlist,dtype=float).mean()
# # NMM_mean = np.array(NMMlist,dtype=float).mean()
# # NOMM_mean = np.array(NOMMlist, dtype=float).mean()
# # std
# NWMM_std = np.asarray(NWMMlist, dtype=float).std()
# # MNMM_std = np.asarray(MNMMlist,dtype=float).std()
# # NMM_std = np.array(NMMlist,dtype=float).std()
# # NOMM_std = np.array(NOMMlist, dtype=float).std()
# # max
# NWMM_max = max(NWMMlist)
# # MNMM_max = max(MNMMlist)
# # NMM_max = max(NMMlist)
# # NOMM_max = max(NOMMlist)
# print("NMI_NWMM_mean={}，NMI_NWMM_std={}，NMI_NWMM_max={}".format(NWMM_mean,NWMM_std,NWMM_max))
# # print("NMI_MNMM_mean={}，NMI_MNMM_std={}，NMI_MNMM_max={}".format(MNMM_mean,MNMM_std,MNMM_max))
# # print("NMI_NMM_mean={}，NMI_NMM_std={}，NMI_NMM_max={}".format(NMM_mean,NMM_std,NMM_max))
# # print("NMI_NOMM_mean={}，NMI_NOMM_std={}，NMI_NOMM_max={}".format(NOMM_mean,NOMM_std,NOMM_max))












