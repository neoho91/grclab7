# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:08:59 2018

@author: Neo
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from fit_rotWP import *
from fit_mallus import *

def anal_rot_pol_ana_together(sample,plot=True):
    main_path=os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol3',sample)
    pol_angs = np.load(os.path.join(main_path,'pol_angs.npy'))
    powers = np.load(os.path.join(main_path,'powers.npy'))
    if 2*np.abs(powers[0]-powers[-1])/np.abs(powers[0]+powers[-1])>0.1:
        powers[0],powers[-1]=[max(powers[0],powers[-1])]*2
    return rotWP_fitter(pol_angs,powers,plot=plot,title=sample)

def _anal_rot_pol_ana_together(plot=True):
    samples = list(filter(lambda x: 'rot_pol_ana_together_1560_' in x,os.listdir(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol3')))
    samples = samples[25:] #start from 27Aug18_1648 until 28Aug18_1107
    As = []
    As_e=[]
    tilts = []
    tilts_e=[]
    retards = []
    retards_e=[]
    
    for i in range(len(samples)/2):
        try:
            main_path_para =os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol3',samples[2*i])
            main_path_perp =os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol3',samples[2*i+1])
            pol_angs = np.load(os.path.join(main_path_para,'pol_angs.npy'))
            powers_para = np.load(os.path.join(main_path_para,'powers.npy'))
            powers_perp = np.load(os.path.join(main_path_perp,'powers.npy'))
            popt,perr = rotWP2_fitter(pol_angs,powers_para,powers_perp,plot=plot,title=2*i*10)
            As.append(popt[0])
            As_e.append(perr[0])
            tilts.append(popt[1])
            tilts_e.append(perr[1])
            retards.append(popt[2])
            retards_e.append(perr[2])
        except (ValueError, RuntimeError, IndexError):
            pass
    
    ts = np.arange(len(retards))*10
    _fig=plt.figure()
    fig=_fig.add_subplot(111)
    fig.errorbar(ts,tilts,yerr=tilts_e,color='C0',capsize=2,marker='o')
    fig.set_ylabel('tilt, deg',color='C0')
    fig.set_xlabel('time, min')
    fig2=fig.twinx()
    fig2.errorbar(ts,retards,yerr=retards_e,color='C1',capsize=2,marker='o')
    fig2.set_ylabel('retardance, wave',color='C1')
    fig2.grid()
    plt.pause(0.1)
    plt.tight_layout()
    plt.pause(0.1)
    return As,tilts,retards

def _anal_rot_pol_ana_together2(plot=True):
    samples = list(filter(lambda x: 'rot_pol_ana_together_1560_' in x,os.listdir(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol3')))
    samples = samples[249:] #start from rot_pol_ana_together_1560_28Aug18_1114
    As = []
    As_e=[]
    tilts = []
    tilts_e=[]
    retards = []
    retards_e=[]
    
    for sample in samples:
        try:
            popt,perr = anal_rot_pol_ana_together(sample,plot)
            As.append(popt[0])
            As_e.append(perr[0])
            tilts.append(popt[1])
            tilts_e.append(perr[1])
            retards.append(popt[2])
            retards_e.append(perr[2])
        except (ValueError, RuntimeError, IndexError, IOError):
            pass
    
    ts = np.arange(len(retards))*5
    _fig=plt.figure()
    fig=_fig.add_subplot(111)
    fig.errorbar(ts,tilts,yerr=tilts_e,color='C0',capsize=2,marker='o')
    fig.set_ylabel('tilt, deg',color='C0')
    fig.set_xlabel('time, min')
    fig2=fig.twinx()
    fig2.errorbar(ts,retards,yerr=retards_e,color='C1',capsize=2,marker='o')
    fig2.set_ylabel('retardance, wave',color='C1')
    fig2.grid()
    plt.pause(0.1)
    plt.tight_layout()
    plt.pause(0.1)
    return As,tilts,retards

def anal_rot_ana(sample,plot=True):
    main_path=os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol3',sample)
    try:
        ana_angs = np.load(os.path.join(main_path,'pol_angs.npy'))
    except IOError:
        ana_angs = np.load(os.path.join(main_path,'ana_angs.npy'))
    powers = np.load(os.path.join(main_path,'powers.npy'))
    if 2*np.abs(powers[0]-powers[-1])/np.abs(powers[0]+powers[-1])>0.1:
        powers[0],powers[-1]=[max(powers[0],powers[-1])]*2
    return mallus_fitter(ana_angs,powers,plot=plot,title=sample)