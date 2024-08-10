import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.pyplot import MultipleLocator

sheet='LFR_NMI'

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

def excel_one_line_to_list(j):
    df_news = pd.read_excel(r'./LFR_NMI_All.xlsx',sheet_name=sheet,header = None)
    list=[]
    for i in df_news[j]:
        list.append(i)
    return list
#Q_NMI=excel_one_line_to_list(0)
fmri_Edmot=excel_one_line_to_list(1)
fmri_ME_k_means=excel_one_line_to_list(0)
fmri_Motif_GA=excel_one_line_to_list(2)
fmri_MLEO=excel_one_line_to_list(3)
fmri_MELPA=excel_one_line_to_list(4)
fmri_FMMEM=excel_one_line_to_list(5)

# labels = ['MFFACD', 'MWLP', 'EdMot-Louvain', 'Motif-LinLog', 'Motif-SC', 'Motif-DECD']
X = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# plt.style.reload_library()  # 重新加载样式
# ['science','ieee'(4色),'nature','scatter'(7色),'grid','notebook','dark_background','high-vis'(6色),'bright'(7色),'vibrant'(7色),'muted'(10色),'high-contrast'(3色),'light'(9色),'std-colors'(7色),'retro'(6色),'']
with plt.style.context(['science', 'no-latex', 'ieee']):  # 使用指定绘画风格 'no-latex’:默认使用latex，则需要电脑安装下载LaTeX
    plt.rcParams.update({
        "font.family": "Times New Roman",  # 字体系列 默认的衬线字体
        "font.serif": ["Times"],  # 衬线字体，Times为Times New Roman
        "font.size": 8})  # 字体大小
    fig, ax = plt.subplots()
    ax.plot(X, fmri_FMMEM, marker='D', markersize=3,linewidth = 1.75, color='red', label='FMMEM')
    ax.plot(X, fmri_ME_k_means, marker='s', markersize=3, color='black',label='ME+k-means')
    ax.plot(X, fmri_MLEO, marker='<', markersize=3, color='brown', linestyle='-.', label='MLEO')
    ax.plot(X, fmri_MELPA, marker='v', markersize=3, color='orange', linestyle=':', label='MELPA')
    ax.plot(X, fmri_Edmot, marker='o', markersize=3, color='blue', label='EdMot')
    ax.plot(X, fmri_Motif_GA, marker='1', markersize=5,linestyle='--', label='MotifGA')

    ax.legend()  # 标识信息 title='Title',loc='upper right'
    ax.set(xlabel='Mixing Parameter (μ)')  # x轴的标题
    ax.set(ylabel='Normalized Mutual Information (NMI)')  # y轴的标题
    # ax.set(ylabel='Adjusted Rand Index (ARI)')  # y轴的标题
    # ax.set(ylabel='F1 Score (F1)')  # y轴的标题
    # ax.set(ylabel='Fowlkes–Mallows index (FMI)')  # y轴的标题
    # ax.autoscale(tight=True) # 自动缩放：是否紧密，最后一个刻度为图边缘
    # ax.set_xlim(0,1) # x轴的刻度范围
    plt.ylim(-0.04, 1.04)
    x_major_locator = MultipleLocator(0.1)
    y_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    fig.savefig('./FMMEM_'+sheet+'.pdf', dpi=1600)  # 输出
