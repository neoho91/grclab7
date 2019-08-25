# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:29:48 2017

@author: Neo
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import sys

def fit_all_data(sample):
    main_path=os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol2',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    ana_angs=np.load(os.path.join(main_path,'ana_angs.npy'))
    powers_files = list(filter(lambda name: 'powers' in name,npy_files))
    
    global hwp_angs,all_qwp_angs,A_data,phi_data,R_data,gamma_data
    A_data,phi_data,R_data = [],[],[]
    
    def get_h_from_name(name):
        return float(name.split('h')[1].split('_')[0])/100
    def get_q_from_name(name):
        return float(name.split('q')[1].split('.npy')[0])/100
    
    hwp_angs = list(map(lambda name: get_h_from_name(name),powers_files))
    hwp_angs = list(set(hwp_angs))
    hwp_angs.sort()
    hwp_angs = np.array(hwp_angs)
    
    all_qwp_angs=np.load(os.path.join(main_path,'qwp_angs.npy'))
    
    for hwp_ang in hwp_angs:
        powers_files_h = list(filter(lambda name: hwp_ang == get_h_from_name(name),powers_files))
        qwp_angs = list(map(lambda name: get_q_from_name(name),powers_files_h))
        qwp_angs = list(set(qwp_angs))
        qwp_angs.sort()
        qwp_angs = np.array(qwp_angs)
        
        curr_A_data,curr_phi_data,curr_R_data = [],[],[]
        for qwp_ang in qwp_angs:
            curr_powers = np.load(os.path.join(main_path,'powers_h%i_q%i.npy'%(hwp_ang*100,qwp_ang*100)))
            try:
                curr_A,curr_phi,curr_R = two_petals_fitter(ana_angs,curr_powers)
                curr_A_data.append(curr_A)
                curr_phi_data.append(curr_phi)
                curr_R_data.append(curr_R)
            except ValueError:
                pass

        if len(curr_A_data) < len(all_qwp_angs):
            curr_A_data.extend([0]*(len(all_qwp_angs)-len(curr_A_data)))
            curr_phi_data.extend([0]*(len(all_qwp_angs)-len(curr_phi_data)))
            curr_R_data.extend([1]*(len(all_qwp_angs)-len(curr_R_data)))
        
        A_data.append(curr_A_data)
        phi_data.append(curr_phi_data)
        R_data.append(curr_R_data)
    
    A_data=np.array(A_data)*1e3 #mW
    phi_data=np.array(phi_data) #deg
    R_data=np.array(R_data)
    gamma_data=np.arctan(R_data)*180./np.pi #deg
        
    return A_data,phi_data,R_data,gamma_data
    
def anal_compensating_QWP(sample,alpha_step=10,alpha_min=0,alpha_max=360,gamma_max=0.1):
    A_data,phi_data,R_data,gamma_data=fit_all_data(sample)
    alpha_bins = np.arange(alpha_min,alpha_max,alpha_step)
    global processed_phi_data,processed_gamma_data,processed_hwp,processed_qwp
    processed_phi_data,processed_gamma_data,processed_hwp,processed_qwp = [],[],[],[]
    for alpha_bin in alpha_bins:
        filtered1_phi_data,filtered1_gamma_data,filtered1_hwp,filtered1_qwp=filter1_phi(alpha_bin,alpha_bin+alpha_step)
        filtered2_phi_data,filtered2_gamma_data,filtered2_hwp,filtered2_qwp=filter2_gamma(-10,gamma_max)
        if len(filtered2_gamma_data) < 1:
            break
        filtered2_gamma_data_abs = np.abs(filtered2_gamma_data)
        min_gamma = np.min(filtered2_gamma_data_abs)
        idx = list(filtered2_gamma_data_abs).index(min_gamma)
        processed_phi_data.append(filtered2_phi_data[idx])
        processed_gamma_data.append(filtered2_gamma_data[idx])
        processed_hwp.append(filtered2_hwp[idx])
        processed_qwp.append(filtered2_qwp[idx])
    for i,h in enumerate(processed_hwp):
        plot_petals(sample,h,processed_qwp[i])
    plt.legend(loc='best')
    return processed_phi_data,processed_gamma_data,processed_hwp,processed_qwp
        
    
    
    
#_____________________________________________
def two_petals(ang,A,phi,R):
    return A*(np.square(np.cos((ang-phi)/180.*np.pi)) + R*np.square(np.sin((ang-phi)/180.*np.pi)))
def two_petals_fitter(angs,intens):
    A = np.max(intens)
    phi = angs[intens.argmax()]
    if phi > 180:
        phi-=180
    R = np.min(intens)
    p0 = (A, phi, R)
    popt, pcov = sp.optimize.curve_fit(two_petals, angs, intens, p0)
#    perr = np.sqrt(np.diag(pcov))
    return popt#, perr

def filter1_phi(min_phi,max_phi):
    global filtered1_phi_data,filtered1_gamma_data,filtered1_hwp,filtered1_qwp
    filtered1_phi_data,filtered1_gamma_data,filtered1_hwp,filtered1_qwp=[],[],[],[]
    for hwp_i,hwp_ang in enumerate(hwp_angs):
        for qwp_i,qwp_ang in enumerate(all_qwp_angs):
            curr_phi = phi_data[hwp_i][qwp_i]
            if curr_phi > min_phi and curr_phi < max_phi:
                filtered1_phi_data.append(curr_phi)
                filtered1_gamma_data.append(gamma_data[hwp_i][qwp_i])
                filtered1_hwp.append(hwp_ang)
                filtered1_qwp.append(qwp_ang)
            else:
                continue
    return filtered1_phi_data,filtered1_gamma_data,filtered1_hwp,filtered1_qwp

def filter2_gamma(min_gamma,max_gamma):
    global filtered2_phi_data,filtered2_gamma_data,filtered2_hwp,filtered2_qwp
    filtered2_phi_data,filtered2_gamma_data,filtered2_hwp,filtered2_qwp=[],[],[],[]
    for i,curr_gamma in enumerate(filtered1_gamma_data):
        if curr_gamma > min_gamma and curr_gamma < max_gamma:
            filtered2_phi_data.append(filtered1_phi_data[i])
            filtered2_gamma_data.append(curr_gamma)
            filtered2_hwp.append(filtered1_hwp[i])
            filtered2_qwp.append(filtered1_qwp[i])
        else:
            continue
    return filtered2_phi_data,filtered2_gamma_data,filtered2_hwp,filtered2_qwp

def plot_petals(sample,h,q):
    main_path=os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol2',sample)
    ana_angs=np.load(os.path.join(main_path,'ana_angs.npy'))
    powers=np.load(os.path.join(main_path,'powers_h%i_q%i.npy'%(h*100,q*100)))
    global A,phi,R
    A,phi,R=two_petals_fitter(ana_angs,powers)
    
    _fig = plt.figure(r'%s'%(sample))
    fig = _fig.add_subplot(111,projection='polar')
    fig.plot(ana_angs[:len(powers)]/180.*np.pi,powers,'o',label='h%.1f q%.1f'%(h,q))
    fig.plot(np.linspace(0,360,100)/180.*np.pi,two_petals(np.linspace(0,360,100),A,phi,R),color='grey')
    fig.set_title(sample)
    fig.set_xticks(np.arange(0,360,60)/180.*np.pi)

def put_angles_to_same_range(angs,period=90,up=45,lo=-45):
    if len(angs) < 2:
        return angs
    else:
        ans = []
        for i,ang in enumerate(angs):
#            if i == 0:
            while ang < lo:
                ang += period
            while ang > up:
                ang -= period
            ans.append(ang)
#            else:
#                while ang < ans[-1] - period*0.95:
#                    ang += period
#                while ang > ans[-1] + period*0.95:
#                    ang -= period
#                ans.append(ang)
        return ans