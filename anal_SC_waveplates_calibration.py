# -*- coding: utf-8 -*-
"""
Created on Thu Aug 08 18:08:50 2019

@author: Neo
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
import cv2
import time
import si_prefix
import copy
import scipy.ndimage
import scipy.optimize
sys.path.append(r'D:/WMP_setup/Python_codes')
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from neo_common_code import *
from correction_factor_4mirrors import *
from fit_gaussian import *
from fit_poly import *
import joblib
from joblib import Parallel,delayed
import threading

def anal_calibrate_sc_waveplates(sample):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\sc_waveplates_calib',sample)
    h_angs = np.load(os.path.join(main_path,'h_angs.npy'))
    q_angs = np.load(os.path.join(main_path,'q_angs.npy'))
    data = np.load(os.path.join(main_path,'data.npy'))
    
    global alphas_aft_100x, gammas_aft_100x, alphas_ref,gammas_ref
    
    alphas_aft_100x = list(map(
            lambda datum_same_h: list(map(lambda datum: datum[0][0], datum_same_h)),
            data))
    gammas_aft_100x = list(map(
            lambda datum_same_h: list(map(lambda datum: datum[0][1], datum_same_h)),
            data))
    alphas_ref = list(map(
            lambda datum_same_h: list(map(lambda datum: datum[1][0], datum_same_h)),
            data))
    gammas_ref = list(map(
            lambda datum_same_h: list(map(lambda datum: datum[1][1], datum_same_h)),
            data))
    
    _fig = plt.figure(sample)
    fig1 = _fig.add_subplot(221)
    fig2 = _fig.add_subplot(222)
    fig3 = _fig.add_subplot(223)
    fig4 = _fig.add_subplot(224)

    fig1.set_title('alphas_aft_100x')
    fig2.set_title('gammas_aft_100x')
    fig3.set_title('alphas_ref')
    fig4.set_title('gammas_ref')
    
    fig1.imshow(alphas_aft_100x)
    fig2.imshow(gammas_aft_100x)
    fig3.imshow(alphas_ref)
    fig4.imshow(gammas_ref)

def anal_calibrate_sc_waveplates2(sample):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\sc_waveplates_calib2',sample)
    h_angs = np.load(os.path.join(main_path,'h_angs.npy'))
    q_angs_off = np.load(os.path.join(main_path,'q_angs_off.npy'))
    data = np.load(os.path.join(main_path,'data.npy'))
    
    global alphas_aft_100x, gammas_aft_100x, alphas_ref,gammas_ref, alphas_aft_100x_2
    
    alphas_aft_100x = list(map(
            lambda datum_same_h: list(map(lambda datum: datum[0][0], datum_same_h)),
            data))
    gammas_aft_100x = list(map(
            lambda datum_same_h: list(map(lambda datum: datum[0][1], datum_same_h)),
            data))
    alphas_ref = list(map(
            lambda datum_same_h: list(map(lambda datum: datum[1][0], datum_same_h)),
            data))
    gammas_ref = list(map(
            lambda datum_same_h: list(map(lambda datum: datum[1][1], datum_same_h)),
            data))
    
    alphas_aft_100x_2 = (alphas_aft_100x)
    
    for r,row in enumerate(alphas_aft_100x):
        for c,col in enumerate(row):
            a = alphas_aft_100x[r][c]
            g = gammas_aft_100x[r][c]
            a_new, g_new = put_polarimeter_angles_to_smallest_range(a,g)
            a_new2, g_new2 = put_polarimeter_angles_to_smallest_range2(a,g)
            alphas_aft_100x[r][c] = a_new # 0 <= alpha < 180
            gammas_aft_100x[r][c] = g_new # -45 <= gamma <= 45
            alphas_aft_100x_2[r][c] = a_new2 # -90 <= alpha < 90
    
    for r,row in enumerate(alphas_ref):
        for c,col in enumerate(row):
            a = alphas_ref[r][c]
            g = gammas_ref[r][c]
            a_new,g_new = put_polarimeter_angles_to_smallest_range(a,g)
            alphas_ref[r][c] = a_new
            gammas_ref[r][c] = g_new
    
    min_val,sigmap_g,min_loc,sigmap = cv2.minMaxLoc(np.array(gammas_aft_100x)-45)
    sigmam_g,max_val,sigmam,max_loc = cv2.minMaxLoc(np.array(gammas_aft_100x)+45)
    h_pol_ag,max_val,h_pol,max_loc = cv2.minMaxLoc(np.abs(np.array(gammas_aft_100x))+np.abs(np.array(alphas_aft_100x_2)))
    v_pol_ag,max_val,v_pol,max_loc = cv2.minMaxLoc(np.abs(np.array(gammas_aft_100x))+np.abs(np.array(alphas_aft_100x)-90))
    
    sigmap_g += 45
    sigmam_g -= 45
    
    sigmap_hq = (h_angs[sigmap[1]],q_angs_off[sigmap[0]])
    sigmap_ref_ag = (alphas_ref[sigmap[1]][sigmap[0]],gammas_ref[sigmap[1]][sigmap[0]])
    
    sigmam_hq = (h_angs[sigmam[1]],q_angs_off[sigmam[0]])
    sigmam_ref_ag = (alphas_ref[sigmam[1]][sigmam[0]],gammas_ref[sigmam[1]][sigmam[0]])
    
    h_pol_hq = (h_angs[h_pol[1]],q_angs_off[h_pol[0]])
    h_pol_ref_ag = (alphas_ref[h_pol[1]][h_pol[0]],gammas_ref[h_pol[1]][h_pol[0]])
    
    v_pol_hq = (h_angs[v_pol[1]],q_angs_off[v_pol[0]])
    v_pol_ref_ag = (alphas_ref[v_pol[1]][v_pol[0]],gammas_ref[v_pol[1]][v_pol[0]])
    
    result = ((sigmap_hq,sigmam_hq,h_pol_hq,v_pol_hq),(sigmap_g,sigmam_g,h_pol_ag,v_pol_ag),(sigmap_ref_ag,sigmam_ref_ag,h_pol_ref_ag,v_pol_ref_ag))
    
    q_resol = np.diff(q_angs_off)[0]
    h_resol = np.diff(h_angs)[0]
    def format_coord(x, y):
        x += 1
        y += 1
        x = round(x*q_resol/q_resol)*q_resol-q_resol + q_angs_off[0]
        y = round(y*h_resol/h_resol)*h_resol-h_resol + h_angs[0]
        x2 = 0.5*y + x
        return 'h=%1.3f, q off=%1.3f (q = %1.3f)'%(y,x,x2)
    qlb=list(np.linspace(q_angs_off[0],q_angs_off[-1],10,dtype=int))
    hlb=list(np.linspace(h_angs[0],h_angs[-1],10,dtype=int))
    
    _fig = plt.figure(sample)
    _fig.suptitle(r'$\sigma^+$ ($\gamma$ = %.2f deg) @ (H,Q_off) = (%.2f,%.2f) | (ref $\alpha$,ref $\gamma$) = (%.2f,%.2f)''\n'
                  r'$\sigma^-$ (%.2f deg) @ (%.2f,%.2f) | (%.2f,%.2f)''\n'
                  r'H-pol ($\gamma$ + $\alpha$ = %.2f deg) @ (%.2f,%.2f) | (%.2f,%.2f)''\n'
                  r'V-pol (%.2f deg) @ (%.2f,%.2f) | (%.2f,%.2f)'%(
                  sigmap_g,sigmap_hq[0],sigmap_hq[1],sigmap_ref_ag[0],sigmap_ref_ag[1],
                  sigmam_g,sigmam_hq[0],sigmam_hq[1],sigmam_ref_ag[0],sigmam_ref_ag[1],
                  h_pol_ag,h_pol_hq[0],h_pol_hq[1],h_pol_ref_ag[0],h_pol_ref_ag[1],
                  v_pol_ag,v_pol_hq[0],v_pol_hq[1],v_pol_ref_ag[0],v_pol_ref_ag[1]))
    
    fig1 = _fig.add_subplot(221)
    fig2 = _fig.add_subplot(222)
    fig3 = _fig.add_subplot(223)
    fig4 = _fig.add_subplot(224)

    fig1.set_title('alphas_aft_100x')
    fig2.set_title('gammas_aft_100x')
    fig3.set_title('alphas_ref')
    fig4.set_title('gammas_ref')
    
    im1 = fig1.imshow(alphas_aft_100x)
    im2 = fig2.imshow(gammas_aft_100x)
    im3 = fig3.imshow(alphas_ref)
    im4 = fig4.imshow(gammas_ref)
    
    _fig.colorbar(im1,ax=fig1)
    _fig.colorbar(im2,ax=fig2)
    _fig.colorbar(im3,ax=fig3)
    _fig.colorbar(im4,ax=fig4)

    fig1.set_xticks(np.arange(0,len(q_angs_off),10/q_resol))
    fig1.set_yticks(np.arange(0,len(h_angs),10/h_resol))
    fig1.set_xticklabels(qlb)
    fig1.set_yticklabels(hlb)
    fig1.set_xlabel('Q offset (deg)')
    fig1.set_ylabel('H (deg)')
    fig1.format_coord = format_coord
    
    fig2.set_xticks(np.arange(0,len(q_angs_off),10/q_resol))
    fig2.set_yticks(np.arange(0,len(h_angs),10/h_resol))
    fig2.set_xticklabels(qlb)
    fig2.set_yticklabels(hlb)
    fig2.set_xlabel('Q offset (deg)')
    fig2.set_ylabel('H (deg)')
    fig2.format_coord = format_coord
    
    fig3.set_xticks(np.arange(0,len(q_angs_off),10/q_resol))
    fig3.set_yticks(np.arange(0,len(h_angs),10/h_resol))
    fig3.set_xticklabels(qlb)
    fig3.set_yticklabels(hlb)
    fig3.set_xlabel('Q offset (deg)')
    fig3.set_ylabel('H (deg)')
    fig3.format_coord = format_coord
    
    fig4.set_xticks(np.arange(0,len(q_angs_off),10/q_resol))
    fig4.set_yticks(np.arange(0,len(h_angs),10/h_resol))
    fig4.set_xticklabels(qlb)
    fig4.set_yticklabels(hlb)
    fig4.set_xlabel('Q offset (deg)')
    fig4.set_ylabel('H (deg)')
    fig4.format_coord = format_coord
    
    maximize_current_plt_window()
    
    return result