# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 09:45:57 2017

@author: QiYue
"""
#输入到matlab里
#fid=fopen('C:\Users\QiYue\Desktop\GN_1.txt','w');
#fprintf(fid,' %.12f ',adj_matrix);
#fclose(fid);
#得到网络的矩阵形式
l1=[]
for line in open("GN_8.txt"):
    l1.append(line.split())
l1=l1[0]
l2=[[0]*128 for i in range(128)]
i=0
while i<128:
    a=l1[128*i:128*(i+1)]
    l2[i]=map(float,a)
    i+=1
edge=[]
for i in range(128):
    for j in range(128):
        if l2[i][j]==1:
            edge.append((i,j))
def text_save(content,filename,mode='a'):
   file = open(filename,mode)
   for i in range(len(content)):
       file.write(str(content[i][0])+' '+str(content[i][1])+'\n')
   file.close()
text_save(edge,'GN8.txt')

