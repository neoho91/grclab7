# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:24:59 2017

@author: Neo
"""

import itertools
import numpy as np
import scipy.optimize

def mallus_fx(theta,p):
    A,phi,R = p
    return A*( np.square(np.cos((theta-phi)/180.*np.pi)) + R*np.square(np.sin((theta-phi)/180.*np.pi)) )

def reVP1Pump_fx(beta,p):
    A,beta_off,gamma = p
    return A/4.*(2+np.cos(2*(beta-beta_off)/180.*np.pi)+np.cos((2*(beta-beta_off)-4*gamma)/180.*np.pi))

def reVP1SHG_fx(beta,p, beta_off,gamma):
    k,phi = p
    beta = beta/180.*np.pi
    beta_off = beta_off/180.*np.pi
    gamma = gamma/180.*np.pi
    phi = phi/180.*np.pi
    
    return 1/16.*(12 + 7*np.square(k)
                  - 4*(1 + 2*np.square(k))*np.cos(4*gamma)
                  + np.square(k)*np.cos(8*gamma)
                  + 2*np.cos(2*(beta-beta_off))
                  + (4 + np.square(k)*(np.cos(8*gamma) - 1))*np.cos(2*(2*gamma + beta-beta_off))
                  + 2*np.cos(2*(4*gamma + beta-beta_off))
                  - 16*k*np.square(np.cos(2*gamma))*np.cos(phi)*np.sin(2*gamma)*np.sin(2*(2*gamma + beta-beta_off))
                  + 32*k*np.square(np.sin(2*gamma))*np.sin(phi))

def gfit_reVP1Pump(I,beta,beta_off=[-5]):
    A=[1]
    gamma = list(np.arange(-45,46,5))
    global_idxs=[0,1]
    _ps = list(itertools.zip_longest(
            A,beta_off,gamma #<<<<<<<<
            ,fillvalue=9999))
    fx = reVP1Pump_fx #<<<<<<<<<<<<<<
    
    ps = list(filter(lambda x: x!=9999,np.array(_ps).flatten()))
    var_num=len(_ps[0])
    p_best, p_cov, nil1,nil2,nil3 = scipy.optimize.leastsq(err_global, ps, 
                                    args=(beta, I,var_num,fx,global_idxs #<<<<<<<<<
                                          ),full_output=True)
    
    return p_best,np.sqrt(np.diag(p_cov))

def gfit_reVP1SHG(I,beta,beta_off,gammas):
    k=[-0.6]
    phi=[45]*len(I)
    global_idxs=[0]
    args = list(zip([beta_off]*len(gammas),gammas))
    _ps = list(itertools.zip_longest(
            k,phi #<<<<<<<<
            ,fillvalue=9999))
    fx = reVP1SHG_fx #<<<<<<<<<<<<<<
    
    ps = list(filter(lambda x: x!=9999,np.array(_ps).flatten()))
    var_num=len(_ps[0])
    p_best, p_cov, nil1,nil2,nil3 = scipy.optimize.leastsq(err_global, ps, 
                                    args=(beta, I,var_num,fx,global_idxs,args #<<<<<<<<<
                                          ),full_output=True,maxfev=1000000)
    return p_best,np.sqrt(np.diag(p_cov))


#_____________________________________________________________________________________________________#
def err_global(ps, x, ys,var_num,fx,global_idxs=[],args=[]):
    all_err=[]
    curr_ps = generate_p(ps,var_num,len(ys),global_idxs)
    for i,y in enumerate(ys):
        if args == []:
            curr_err = fx(x,next(curr_ps)) - y
        else:
            curr_err = fx(x,next(curr_ps),*args[i]) - y
        all_err = np.concatenate((all_err, curr_err))
    return all_err

def generate_p(ps, var_num, y_num,global_idxs):
    idxs = list(range(var_num*y_num-len(global_idxs)*(y_num-1)))
    for i in range(y_num-1):
        for k,global_idx in enumerate(global_idxs):
            idxs.insert((i+1)*(var_num)+(k),global_idx)
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