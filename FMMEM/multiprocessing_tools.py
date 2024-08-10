# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:40:49 2023

@author: WYW

用与多进程计算
"""
import multiprocessing


# =============================================================================
# 创建进程池
# =============================================================================
def createPool(pnum):
    pool = multiprocessing.Pool(pnum)
    return pool
# =============================================================================
# 关闭进程池
# =============================================================================
def closePool(pool):
    pool.close()
    pool.join()

# def model1(n):
#     v = 0
#     for i in range(n):
#         v = v+i*i
#     return v

def run_model(model, data):
    result = model(data)
    return result

# =============================================================================
# #运行进程池
# # =============================================================================
def run(pool, model, datas):
    # 运行模型并获取结果
    res = [pool.apply_async(model, data) for data in datas]
    result = [x.get() for x in res]
    return result

# if __name__ == '__main__':
#     poolNum = 10
#     pool = createPool(poolNum)
#     datas = [(i) for i in range(100)]
#     models = [model]*100
#     run(pool, models, datas)
#     closePool(pool)
