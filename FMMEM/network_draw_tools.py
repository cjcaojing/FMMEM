# -*- coding: utf-8 -*-
"""
Created on Sat May 14 23:05:29 2022

@author: WYW
"""
import networkx as nx                  #导入networkx网络分析模块，缩写为nx。Networkx为基于Python的最基本的网络分析库。
from matplotlib import pyplot as plt
import numpy as np



real_mem = []
with open(r"data/经典数据集/real/" + 'dolphins_groundtruth.txt', mode='r',encoding='UTF-8') as f:
     real_mem = list(map(int,f.read().splitlines()))

print(real_mem)

a_list=[]
b_list=[]
for i in range(62):
    if real_mem[i]==1:
        a_list.append(i)
    else:
        b_list.append(i)

colors = ['#eb8f90', '#ffb471', '#adbed2', '#12406a']   
options = {'font_family': 'serif','font_size': '8', 'font_color': '#ffffff'} 
G = nx.read_edgelist(r"data/经典数据集/dolphins.txt",nodetype=int)
G = G.to_undirected()
# community_list=[[0, 1, 2, 3, 7, 9, 11, 12, 13, 17, 19, 21],[4, 5, 6, 10, 16],[8, 14, 15, 18, 20, 22, 26, 29, 30, 32, 33],[23, 24, 25, 27, 28, 31]]#karate
community_list=[a_list,b_list,[28,30,39]]#karate
pos = nx.spring_layout(G) 
nx.draw(G,pos,edge_color="gray",with_labels=True, node_size = 270,**options)
for i in range(len(community_list)):
    nx.draw_networkx_nodes(G,pos,nodelist=community_list[i], node_size=300, node_color=colors[i],label=True)
plt.savefig("基于节点网络可视化.svg",dpi=6000)