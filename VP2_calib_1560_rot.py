# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 19:18:20 2017

@author: Neo
"""
import numpy as np
import os
import sys
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file
import matplotlib.pyplot as plt
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from neo_common_code import *
try:
    move_rot1()
except:
    try:
        from rot_stages_noGUI import *
    except:
        print('HWP, QWP and POL rotational stages offline.')    
try:
    from powermeter_analog import *
except:
    print("powermeter analog not connected.")
try:
    import TL_slider_3
    pass
except:
    print("laser shutter not connected.")
#from shutter_rotational_stage import *
try:
    from laser_control import *
except:
    print("laser control not connected.")

import threading
import scipy.interpolate
import scipy as sp
calib_path = os.path.join(r'D:\Nonlinear_setup\Python_codes\VP2_calib_1560')
CALIB_alphas = np.load(os.path.join(calib_path,'alphas.npy'))
CALIB_HWP_angs = np.load(os.path.join(calib_path,'HWP_angs.npy'))
CALIB_QWP_angs = np.load(os.path.join(calib_path,'QWP_angs.npy'))
#CALIB_alphas_offset = np.load(os.path.join(calib_path,'alphas_offset.npy'))
try:
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(calib_path)))
    alphas_offset_files = list(filter(lambda name: 'alphas_offset' in name,npy_files))
    CALIB_alphas_offset = np.zeros(len(CALIB_alphas))
    for alphas_offset_file in alphas_offset_files:
        CALIB_alphas_offset = np.load(os.path.join(calib_path,alphas_offset_file)) + CALIB_alphas_offset
except:
    CALIB_alphas_offset = np.zeros(len(CALIB_alphas))

#interp_H_fx_in = scipy.interpolate.interp1d(CALIB_alphas + CALIB_alphas_offset,CALIB_HWP_angs,kind='cubic')
#interp_Q_fx_in = scipy.interpolate.interp1d(CALIB_alphas + CALIB_alphas_offset,CALIB_QWP_angs,kind='cubic')
#interp_H_fx_ex = scipy.interpolate.interp1d(CALIB_alphas + CALIB_alphas_offset,CALIB_HWP_angs,fill_value='extrapolate')
#interp_Q_fx_ex = scipy.interpolate.interp1d(CALIB_alphas + CALIB_alphas_offset,CALIB_QWP_angs,fill_value='extrapolate')
#def interp_H_fx(a):
#    try:
#        return interp_H_fx_in(a)
#    except ValueError:
#        return interp_H_fx_ex(a)
#def interp_Q_fx(a):
#    try:
#        return interp_Q_fx_in(a)
#    except ValueError:
#        return interp_Q_fx_ex(a)
interp_H_fx = scipy.interpolate.interp1d(CALIB_alphas + CALIB_alphas_offset,CALIB_HWP_angs,fill_value='extrapolate')
interp_Q_fx = scipy.interpolate.interp1d(CALIB_alphas + CALIB_alphas_offset,CALIB_QWP_angs,fill_value='extrapolate')
HWP_off=79.387
ANA_off=105.56
QWP_off=20.78

def move_1560_to_alpha(a,period=360):
    while a > period:
        a-=period
    while a < 0:
        a+=period
    interp_H = interp_H_fx(a)
    interp_Q = interp_Q_fx(a)
    move_HQ(interp_H,interp_Q)
    return (interp_H,interp_Q)

def check_alpha(sample,num=0):
    calib_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol2',sample)
    if num == 0:
        confirm_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol2',sample+'_confirm')
    else:
        confirm_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_circular_pol2',sample+'_confirm_%i'%num)
    calib_alphas = np.load(os.path.join(calib_path,'alphas.npy'))
    calib_HWP_angs = np.load(os.path.join(calib_path,'HWP_angs.npy'))
    calib_QWP_angs = np.load(os.path.join(calib_path,'QWP_angs.npy'))
    confirm_ana_angs = np.load(os.path.join(confirm_path,'min_ana_angs.npy'))
    confirm_min_powers = np.load(os.path.join(confirm_path,'min_powers.npy'))
    confirm_curvs = np.load(os.path.join(confirm_path,'min_curvs.npy'))
    
    confirm_alphas = confirm_ana_angs + 90
    alphas_offset = confirm_alphas - calib_alphas[:len(confirm_alphas)] #real_alpha = calib_alpha + alpha_offset
    fig=plt.figure(sample)
    ax = fig.add_subplot(111)
    ax.plot(calib_alphas[:len(confirm_alphas)],alphas_offset,'o',color='C0')
    ax.set_title(sample)
    ax.set_ylabel('alpha offset, deg',color='C0')
    ax.set_xlabel('calib alpha, deg')
    ax.axhline(0)
    ax2=ax.twinx()
    ax2.plot(calib_alphas[:len(confirm_alphas)],confirm_min_powers*1e9,'v',color='C1')
    ax2.set_ylabel('min power, nW',color='C1')
    
    np.save(os.path.join(confirm_path,'alphas_offset.npy'),alphas_offset)
    
    return alphas_offset

def confirm_calibrated_search_for_min():    
    main_path = os.path.join(r'D:\Nonlinear_setup\Python_codes\VP2_calib_1560\calibration')
    if not os.path.isdir(main_path):
        os.mkdir(main_path)    
    
    pma_wl(1560)
    home_all_rot()
    min_ps=[]
    min_As =[]
    curvs=[]
    prev_line = ''
    prints('\n')
    for i,alpha in enumerate(CALIB_alphas):
        TL_slider_3.block_laser_3()
        plt.pause(1)
        pma_zero()
        plt.pause(2)
        TL_slider_3.unblock_laser_3()
        
        line = "At alpha = %.4f deg (%.1f percent)"%(alpha,(100.*i/len(CALIB_alphas)))
        prints(line,prev_line)
        prev_line=line
        global powers, A_range
        powers=[]
        move_1560_to_alpha(alpha)
        A_range=np.arange(alpha-90-1,alpha-90+1+0.01,0.05)
        for A in A_range:
            move_A(A)
            c_powers=[]
            for i in range(3):
                plt.pause(0.1)
                c_powers.append(pma_power())
            powers.append(np.average(c_powers))
        min_p,min_A,curv=parabolic_fitter(A_range,powers)
        if min_A < np.min(A_range):
            print('alpha %.1f deg out of range. Fitted alpha = %.2f deg'%(alpha,min_A))
            powers=[]
            A_range=np.arange(alpha-90-5,alpha-90+1+0.01,0.05)
            for A in A_range:
                move_A(A)
                c_powers=[]
                for i in range(3):
                    plt.pause(0.1)
                    c_powers.append(pma_power())
                powers.append(np.average(c_powers))
        if min_A > np.max(A_range):
            print('alpha %.1f deg out of range. Fitted alpha = %.2f deg'%(alpha,min_A))
            powers=[]
            A_range=np.arange(alpha-90-1,alpha-90+5+0.01,0.05)
            for A in A_range:
                move_A(A)
                c_powers=[]
                for i in range(3):
                    plt.pause(0.1)
                    c_powers.append(pma_power())
                powers.append(np.average(c_powers))
        min_p,min_A,curv=parabolic_fitter(A_range,powers)
        min_As.append(min_A)
        min_ps.append(min_p)
        curvs.append(curv)
        np.save(os.path.join(main_path,'min_ana_angs'),np.array(min_As))
        np.save(os.path.join(main_path,'min_powers'),np.array(min_ps))
        np.save(os.path.join(main_path,'min_curvs'),np.array(curvs))
        home_A()

def contineous_confirm_calibrated_search_for_min(n,just_to_check=False):
    main_path = os.path.join(r'D:\Nonlinear_setup\Python_codes\VP2_calib_1560')
    for j in range(n):
        print('\ncontineous_confirm_calibrated_search_for_min at n = %i'%j)
        confirm_calibrated_search_for_min()
        anal_confirm_calibrated_search_for_min()
        calib_folders = list(filter(lambda name: name.startswith('calibration_') and 'aft' not in name,os.listdir(main_path)))
        next_calib_folder_num = len(calib_folders) + 1
        
        if just_to_check:
            check_calib_folders = list(filter(lambda name: name.startswith('calibration_aft%i'%(next_calib_folder_num-1)),os.listdir(main_path)))
            next_check_calib_folder_num = len(check_calib_folders) + 1
            dst = os.path.join(main_path,r'calibration_aft%i_%i'%(next_calib_folder_num-1,next_check_calib_folder_num))
            if not os.path.exists(dst):
                os.makedirs(dst)
            copy_tree(os.path.join(main_path,r'calibration'),dst)
            plt.pause(1e-3)
        else:
            dst = os.path.join(main_path,r'calibration_%i'%next_calib_folder_num)
            if not os.path.exists(dst):
                os.makedirs(dst)
            copy_tree(os.path.join(main_path,r'calibration'),dst)
            copy_file(os.path.join(main_path,r'calibration\alphas_offset.npy'),os.path.join(main_path,r'alphas_offset_%i.npy'%next_calib_folder_num))
            plt.pause(1e-3)
            
            npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(calib_path)))
            alphas_offset_files = list(filter(lambda name: 'alphas_offset' in name,npy_files))
            CALIB_alphas_offset = np.zeros(len(CALIB_alphas))
            for alphas_offset_file in alphas_offset_files:
                CALIB_alphas_offset = np.load(os.path.join(calib_path,alphas_offset_file)) + CALIB_alphas_offset
            global interp_H_fx,interp_Q_fx
            interp_H_fx = scipy.interpolate.interp1d(CALIB_alphas + CALIB_alphas_offset,CALIB_HWP_angs,fill_value='extrapolate')
            interp_Q_fx = scipy.interpolate.interp1d(CALIB_alphas + CALIB_alphas_offset,CALIB_QWP_angs,fill_value='extrapolate')
            refresh_this_code()

def anal_confirm_calibrated_search_for_min(suffix=''):
    calib_path = os.path.join(r'D:\Nonlinear_setup\Python_codes\VP2_calib_1560')
    calib_alphas = np.load(os.path.join(calib_path,'alphas.npy'))
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(calib_path)))
    confirm_path = os.path.join(r'D:\Nonlinear_setup\Python_codes\VP2_calib_1560\calibration%s'%suffix)
    confirm_ana_angs = np.load(os.path.join(confirm_path,'min_ana_angs.npy'))
    confirm_min_powers = np.load(os.path.join(confirm_path,'min_powers.npy'))
    confirm_curvs = np.load(os.path.join(confirm_path,'min_curvs.npy'))
    
    confirm_alphas = confirm_ana_angs + 90
    alphas_offset = confirm_alphas - calib_alphas[:len(confirm_alphas)] #real_alpha = calib_alpha + alpha_offset
    fig=plt.figure('Calibrated alpha')
    fig.clf()
    ax = fig.add_subplot(111)
    ax.axhspan(-0.1,0.1,color='C0',alpha=0.2)
    ax.plot(calib_alphas[:len(confirm_alphas)],alphas_offset,'o',color='C0')
    ax.set_title('Calibrated alpha')
    ax.set_ylabel('alpha offset, deg',color='C0')
    ax.set_xlabel('calib alpha, deg')
    ax.axhline(0)
    ax2=ax.twinx()
    ax2.plot(calib_alphas[:len(confirm_alphas)],confirm_min_powers*1e9,'v',color='C1')
    ax2.set_ylabel('min power, nW',color='C1')
    ax2.format_coord = make_format(ax2, ax)
    plt.pause(1e-3)
    plt.savefig(os.path.join(confirm_path,'calibrated_alpha.png'))
    
    np.save(os.path.join(confirm_path,'alphas_offset.npy'),alphas_offset)
    
    return alphas_offset
#_______________________________________________
def move_HQ(H,Q):
    def HWP_loop():
        move_rot1(H + HWP_off)
    def QWP_loop():
        move_rot3(Q + QWP_off)
    HWP_th=threading.Thread(target=HWP_loop)    
    QWP_th=threading.Thread(target=QWP_loop)
    HWP_th.start()
    plt.pause(0.01)
    QWP_th.start()
    HWP_th.join()
    QWP_th.join()

def home_HQ():
    def HWP_loop():
        move_rot1(0)
        home_rot1()
    def QWP_loop():
        move_rot3(3)
        home_rot3()
    HWP_th=threading.Thread(target=HWP_loop)    
    QWP_th=threading.Thread(target=QWP_loop)
    HWP_th.start()
    plt.pause(0.01)
    QWP_th.start()
    HWP_th.join()
    QWP_th.join()

def move_A(a=None):
    if a==None:
        return move_rot2() - ANA_off
    return move_rot2(a+ANA_off) - ANA_off

def home_A():
    move_rot2(0)
    home_rot2()

def home_all_rot():
    def HWP_loop():
        move_rot1(0)
        home_rot1()
    def QWP_loop():
        move_rot3(0)
        home_rot3()
    def ANA_loop():
        move_rot2(0)
        home_rot2()
    pol_th=threading.Thread(target=HWP_loop)
    QWP_th=threading.Thread(target=QWP_loop)
    ANA_th=threading.Thread(target=ANA_loop)
    pol_th.start()
    plt.pause(0.01)
    QWP_th.start()
    plt.pause(0.01)
    ANA_th.start()
    pol_th.join()
    QWP_th.join()
    ANA_th.join()

def parabolic_curve(x,min_y,central_x,curvature):
    return min_y + curvature*np.square(x-central_x)
def parabolic_fitter(x,y):
    min_y = np.min(y)
    central_x = x[y.index(min_y)]
    curvature = 1e-6
    p0 = (min_y, central_x, curvature)
    try:
        popt, pcov = sp.optimize.curve_fit(parabolic_curve, x, y, p0)
    except RuntimeError:
        print('Fitting not converged.')
        popt = p0
    return popt

def refresh_this_code():
    runfile('D:/Nonlinear_setup/Python_codes/VP2_calib_1560_rot.py', wdir='D:/Nonlinear_setup/Python_codes')