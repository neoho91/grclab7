# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:13:42 2017

@author: yw_10
"""
import itertools

import numpy as np
import scipy.optimize

def sim(x, p):
    a, b, c  = p
    return a*np.exp(-b * x) + c

def err(p, x, y):
    return sim(x, p) - y


# set up the data
data_x = np.linspace(0, 40, 50)
p1 = [2.5, 1.3, 0.5]       # parameters for the first trajectory
p2 = [4.2, 1.3, 0.2]       # parameters for the second trajectory, same b
p3 = [8.1, 1.3, 0.9]
p4 = [3.3, 1.3, 0.4]
p5 = [6.4, 1.3, 0.7]
data_y1 = sim(data_x, p1)
data_y2 = sim(data_x, p2)
data_y3 = sim(data_x, p3)
data_y4 = sim(data_x, p4)
data_y5 = sim(data_x, p5)
ndata_y1 = data_y1 + np.random.normal(size=len(data_y1), scale=0.01)
ndata_y2 = data_y2 + np.random.normal(size=len(data_y2), scale=0.01)
ndata_y3 = data_y3 + np.random.normal(size=len(data_y3), scale=0.01)
ndata_y4 = data_y4 + np.random.normal(size=len(data_y4), scale=0.01)
ndata_y5 = data_y5 + np.random.normal(size=len(data_y5), scale=0.01)

# global fit

# new err functions which takes a global fit
def err_global(p, x, y1, y2):
    # p is now a_1, b, c_1, a_2, c_2, with b shared between the two
    p1 = p[0], p[1], p[2]
    p2 = p[3], p[1], p[4]

    err1 = err(p1, x, y1)
    err2 = err(p2, x, y2)
    return np.concatenate((err1, err2))

p_global = [2., 1., 0.2, 4., 0.1]    # a_1, b, c_1, a_2, c_2
p_best, p_cov, nil,nil,nil = scipy.optimize.leastsq(err_global, p_global,
                                args=(data_x, ndata_y1, ndata_y2),full_output=True)

print("Global fit results")
print(p_best)
print(np.sqrt(np.diag(p_cov)))

# global fit 2
# new err functions which takes a global fit
def err_global2(ps, x, ys,var_num,fx,global_idxs=[]):
    all_err=[]
    curr_ps = generate_p(ps,var_num,len(ys),global_idxs)
    for i,y in enumerate(ys):
        curr_err = fx(x,next(curr_ps)) - y
        all_err = np.concatenate((all_err, curr_err))
    return all_err

def generate_p(ps, var_num, y_num,global_idxs):
    idxs = list(range(var_num*y_num-len(global_idxs)*(y_num-1)))
    for i in range(y_num-1):
        for k,global_idx in enumerate(global_idxs):
            idxs.insert((i+1)*var_num+(k),global_idx)
    _n=0
    for i in range(y_num):
        curr_ps = []
        for j in range(var_num):
            curr_ps.append(ps[idxs[_n]])
            _n+=1
        yield curr_ps

def pack_p(ps, var_num, y_num, global_idxs):
    idxs = list(range(var_num*y_num-len(global_idxs)*(y_num-1)))
    for i in range(y_num-1):
        for k,global_idx in enumerate(global_idxs):
            idxs.insert((i+1)*var_num+(k),global_idx)
    ans = []
    _n=0
    for i in range(y_num):
        curr_ps=[]
        for j in range(var_num):
            curr_ps.append(ps[idxs[_n]])
            _n+=1
        ans.append(curr_ps)
    return ans

ys=(ndata_y1,ndata_y2,ndata_y3,ndata_y4,ndata_y5)
As=[2.5,4,1,1,1]
Bs=[1.3]
Cs=[0.5,0.2,1,1,1]
var_num=3
global_idxs=[1]

ps = list(filter(lambda x: x!=9999,np.array(list(itertools.zip_longest(As,Bs,Cs,fillvalue=9999))).flatten()))
p_best, p_cov, nil,nil,nil = scipy.optimize.leastsq(err_global2, ps, 
                                    args=(data_x, ys,var_num,sim,global_idxs),full_output=True)

print("\nGlobal fit 2 results")
print(p_best)
print(np.sqrt(np.diag(p_cov)))