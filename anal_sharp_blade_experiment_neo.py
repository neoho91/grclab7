# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:48:48 2017

@author: Neo
"""
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import threading
import scipy.io as io
import shutil
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from fit_CDF import *
from fit_zR import *
from neo_common_code import *

def anal_sharp_blade_scan(sample,show_plot=False):
    """
    return FW1/e2 and its fitting error
    """
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan',sample)
    if not os.path.isdir(main_path):
        len_to_discard = len(sample[-6:].split('_y')[-1])+2
        sup_sample = sample[:-len_to_discard]
        main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan',sup_sample)
    blade_poss = np.load(os.path.join(main_path,'%s_blade_pos.npy'%sample))
    powers = np.load(os.path.join(main_path,'%s_power.npy'%sample))
    popt, perr = CDF_fitter(blade_poss,powers,plot=show_plot)
    if show_plot:
        plt.title(sample)
    y0,A,x0,w = popt
    y0_e,A_e,x0_e,w_e = perr
    datum_to_excel = [blade_poss[:-1],np.diff(powers)]
    return 4*w, 4*w_e, x0, x0_e, datum_to_excel
    

def anal_sharp_blade_scan_loop(sample,show_plots=False):
    main_path= os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan',sample)
#    main_path= os.path.join(r'D:\Nonlinear_setup\system calibration\VP2\beam chopping',sample)
    powers = list(filter(lambda name: name.endswith('_power.npy'),os.listdir(main_path)))
    def get_y_from_name(name):
        return float(name[-6-len('_power.npy'):].split('_y')[-1].split('_power.npy')[0])
    powers.sort(key=get_y_from_name)
    _samples = list(map(lambda name: name[:-len('_power.npy')],powers))
    global FWe2s,FWe2s_e,Ys, x0s,x0s_e,data_to_excel
    Ys = list(map(get_y_from_name,powers))
    FWe2s = []
    FWe2s_e = []
    x0s = []
    x0s_e = []
    data_to_excel = []
    for i,_sample in enumerate(_samples):
        FWe2, FWe2_e, x0, x0_e, datum_to_excel = anal_sharp_blade_scan(_sample,show_plot=show_plots)
        FWe2s.append(FWe2)
        FWe2s_e.append(FWe2_e)
        x0s.append(x0)
        x0s_e.append(x0_e)
        data_to_excel.append(datum_to_excel[0])
        data_to_excel.append(datum_to_excel[1])
    data_to_excel = np.transpose(data_to_excel)
    
    fig = plt.figure(sample)
    ax=fig.add_subplot(111)
    ax.errorbar(Ys,FWe2s,yerr=FWe2s_e,capsize=2,marker='o',color='C0')
    ax.set_ylabel('FW$\\frac{1}{e^2}$, $\mu$m',color='C0')
    ax.set_xlabel('axial position, $\mu$m')
    ax2=ax.twinx()
    ax2.errorbar(Ys,x0s,yerr=x0s_e,capsize=2,marker='o',color='C1')
    ax2.set_ylabel('beam central position, $\mu$m',color='C1')
    ax2.format_coord = make_format(ax2, ax)
    plt.title('%s'%sample)
    ax.grid(color='C0')
    ax2.grid(color='C1')
    
#    try:
    popt,perr = zR_fitter(Ys,np.array(FWe2s)/2.,title='Fitted zR %s'%sample)
    z0,w0,zR = popt
    z0_e,w0_e,zR_e = perr
    title = 'fitted zR %s\nHW$\\frac{1}{e^2}$ = (%.2f $\\pm$ %.2f) $\\mu$m\nz$_R$ = (%.2f $\\pm$ %.2f) $\\mu$m\nz$_0$ = (%.2f $\\pm$ %.2f) $\\mu$m'%(sample,w0,w0_e,zR,zR_e,z0,z0_e)
    plt.title(title)
    plt.ylabel('HW$\\frac{1}{e^2}$, $\mu$m')
    plt.xlabel('axial position, $\mu$m')
    plt.tight_layout()
    plt.pause(0.01)
    print('HW1/e^2 = (%.2f +- %.2f) um'%(w0,w0_e))
    print('zR = (%.2f +- %.2f) um'%(zR,zR_e))
    print('z0 = (%.2f +- %.2f) um'%(z0,z0_e))
    
    (m,c),pcov = np.polyfit(Ys, x0s, 1,cov=True)
    m_e,c_e =np.sqrt(np.diag(pcov))
    x0_fx = np.poly1d((m,c))
    x0 = x0_fx(z0)
    tilt_angle = np.arctan(m)/np.pi*180
    tilt_angle_e = (1./np.square(m)+1)*m_e
    plt.figure('Fitted x0 %s'%sample)
    title = 'fitted x0 %s\ngradient = (%.4f $\pm$ %.4f)\ny-intercept = (%.4f $\pm$ %.4f) $\mu$m\nx0 @ waist = %.1f um, piezo tilted (%.1f $\pm$ %.1f) deg'%(sample,m,m_e,c,c_e,x0,tilt_angle,tilt_angle_e)
    plt.title(title)
    plt.errorbar(Ys,x0s,yerr=x0s_e,capsize=2,marker='o',lw=0,elinewidth=2)
    _Ys=np.linspace(min(Ys),max(Ys),100)
    plt.plot(_Ys,x0_fx(_Ys))
    plt.xlabel('axial position, $\mu$m')
    plt.ylabel('beam location, $\mu$m')
    plt.tight_layout()
    plt.pause(0.01)
    
    return w0, w0_e,zR,zR_e,z0,z0_e,x0
#    except Exception as e:
#        print e
#        pass


def clean_up_sharp_blade_scan_loop(sample,suffix='_y'):
    main_path= os.path.join(r'D:\Nonlinear_setup\Experimental_data\sharp_blade_scan')
    source_folders = list(filter(lambda name: name.startswith('%s%s'%(sample,suffix)),os.listdir(main_path)))
    suffixes = list(map(lambda name: name.split('%s'%sample)[-1],source_folders))
    dest_path = os.path.join(main_path,sample)
    
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
    for i,source_folder in enumerate(source_folders):
        files = os.listdir(os.path.join(main_path,source_folder))
        files.remove('log.txt') 
        for f in files:
            shutil.move(os.path.join(main_path,source_folder,f),os.path.join(dest_path,f))
        shutil.move(os.path.join(main_path,source_folder,'log.txt'),os.path.join(dest_path,'log%s.txt'%suffixes[i]))
        os.rmdir(os.path.join(main_path,source_folder))