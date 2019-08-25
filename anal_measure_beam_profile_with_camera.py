# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:25:13 2018

@author: Neo
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys
import time
import copy
import cv2
sys.path.append(r'D:/WMP_setup/Python_codes')
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
from neo_common_code import *
from fit_gaussian import *
camera_px_size_um = 5.2

def anal_scan_beam_profile(sample,show_img=False):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_beam_profile',sample)
    global sc_poss, toptica_poss
    with np.load(os.path.join(main_path,'sc_data.npz')) as data:
        sc_poss = list(data['y_poss'])
        sc_imgs = data['images']
    with np.load(os.path.join(main_path,'toptica_data.npz')) as data:
        toptica_poss = list(data['y_poss'])
        toptica_imgs = data['images']
    
    if show_img:
        _fig = plt.figure('%s images'%sample)
        sc_im_fig = _fig.add_subplot(121)
        toptica_im_fig = _fig.add_subplot(122)
    
    global Xs_sc,Ys_sc,fwhmsx_sc,fwhmsy_sc,intens_sc,bgs_sc, Xs_toptica,Ys_toptica,fwhmxs_toptica,fwhmys_toptica,intens_toptica,bgs_toptica
    global Xs_sc_e,Ys_sc_e,fwhmsx_sc_e,fwhmsy_sc_e,intens_sc_e,bgs_sc_e, Xs_toptica_e,Ys_toptica_e,fwhmxs_toptica_e,fwhmys_toptica_e,intens_toptica_e,bgs_toptica_e
    Xs_sc,Ys_sc,fwhmsx_sc,fwhmsy_sc,intens_sc,bgs_sc=[],[],[],[],[],[]
    Xs_sc_e,Ys_sc_e,fwhmsx_sc_e,fwhmsy_sc_e,intens_sc_e,bgs_sc_e=[],[],[],[],[],[]
    Xs_toptica,Ys_toptica,fwhmxs_toptica,fwhmys_toptica,intens_toptica,bgs_toptica=[],[],[],[],[],[]
    Xs_toptica_e,Ys_toptica_e,fwhmxs_toptica_e,fwhmys_toptica_e,intens_toptica_e,bgs_toptica_e=[],[],[],[],[],[]
    
    for idx,y_pos in enumerate(toptica_poss):
        curr_sc_pos = sc_poss[idx]
        curr_sc_img = sc_imgs[idx]
        curr_toptica_pos = y_pos
        curr_toptica_img = toptica_imgs[idx]

        crop_sc_img,minx_sc,miny_sc = crop_to_centroid(curr_sc_img)
        crop_toptica_img,minx_toptica,miny_toptica = crop_to_centroid(curr_toptica_img)
        
        proc_sc_img = fill_in_hole(crop_sc_img)
        proc_toptica_img = fill_in_hole(crop_toptica_img)
        
        popt,perr = gaussian2D_fitter(proc_sc_img,False,255)
        X_sc,Y_sc,fwhmx_sc,fwhmy_sc,inten_sc,bg_sc = popt
        X_sc_e,Y_sc_e,fwhmx_sc_e,fwhmy_sc_e,inten_sc_e,bg_sc_e = perr
        X_sc = (X_sc + minx_sc)*camera_px_size_um
        X_sc_e *= camera_px_size_um
        Y_sc = (Y_sc + miny_sc)*camera_px_size_um
        Y_sc_e *= camera_px_size_um
        fwhmx_sc *= camera_px_size_um
        fwhmy_sc *= camera_px_size_um
        fwhmx_sc_e *= camera_px_size_um
        fwhmy_sc_e *= camera_px_size_um
        Xs_sc.append(X_sc)
        Ys_sc.append(Y_sc)
        fwhmsx_sc.append(fwhmx_sc)
        fwhmsy_sc.append(fwhmy_sc)
        intens_sc.append(inten_sc)
        bgs_sc.append(bg_sc)
        Xs_sc_e.append(X_sc_e)
        Ys_sc_e.append(Y_sc_e)
        fwhmsx_sc_e.append(fwhmx_sc_e)
        fwhmsy_sc_e.append(fwhmy_sc_e)
        intens_sc_e.append(inten_sc_e)
        bgs_sc_e.append(bg_sc_e)
        
        popt, perr = gaussian2D_fitter(proc_toptica_img,False,255)
        X_toptica,Y_toptica,fwhmx_toptica,fwhmy_toptica,inten_toptica,bg_toptica = popt
        X_toptica_e,Y_toptica_e,fwhmx_toptica_e,fwhmy_toptica_e,inten_toptica_e,bg_toptica_e = perr
        X_toptica = (X_toptica + minx_toptica)*camera_px_size_um
        X_toptica_e *= camera_px_size_um
        Y_toptica = (Y_toptica + miny_toptica)*camera_px_size_um
        Y_toptica_e *= camera_px_size_um
        fwhmx_toptica *= camera_px_size_um
        fwhmy_toptica *= camera_px_size_um
        fwhmx_toptica_e *= camera_px_size_um
        fwhmy_toptica_e *= camera_px_size_um
        Xs_toptica.append(X_toptica)
        Ys_toptica.append(Y_toptica)
        fwhmxs_toptica.append(fwhmx_toptica)
        fwhmys_toptica.append(fwhmy_toptica)
        intens_toptica.append(inten_toptica)
        bgs_toptica.append(bg_toptica)
        Xs_toptica_e.append(X_toptica_e)
        Ys_toptica_e.append(Y_toptica_e)
        fwhmxs_toptica_e.append(fwhmx_toptica_e)
        fwhmys_toptica_e.append(fwhmy_toptica_e)
        intens_toptica_e.append(inten_toptica_e)
        bgs_toptica_e.append(bg_toptica_e)
        
        if show_img:
            if idx == 0:
                sc_im = sc_im_fig.imshow(proc_sc_img,norm=LogNorm(vmin=0.01, vmax=255),cmap='Blues_r')
                toptica_im = toptica_im_fig.imshow(proc_toptica_img,norm=LogNorm(vmin=0.01, vmax=255),cmap='Reds_r')
                _fig.suptitle(sample)
                plt.pause(0.01)
                plt.tight_layout(rect=(0,0,1,0.95))
            else:
                sc_im.set_data(proc_sc_img)
                toptica_im.set_data(proc_toptica_img)          
                
            sc_im_fig.set_title('SC %.3f $\mu$m'%curr_sc_pos)
            toptica_im_fig.set_title('Toptica %.3f $\mu$m'%curr_sc_pos)
            plt.pause(0.1)
            
    xy_th = 0.5
    fwhm_th = 0.5
    nil,nil,in_idx0,nil,nil,out_idx0=remove_outlier2(sc_poss,sc_poss,th=np.median(np.diff(sc_poss)*1.5),cyclic=False,return_idx=True)
    nil,nil,in_idx1,nil,nil,out_idx1=remove_outlier2(sc_poss,Xs_sc,th=xy_th,cyclic=False,return_idx=True)
    nil,nil,in_idx2,nil,nil,out_idx2=remove_outlier2(sc_poss,fwhmsx_sc,th=fwhm_th,cyclic=False,return_idx=True)
    nil,nil,in_idx3,nil,nil,out_idx3=remove_outlier2(sc_poss,Ys_sc,th=xy_th,cyclic=False,return_idx=True)
    nil,nil,in_idx4,nil,nil,out_idx4=remove_outlier2(sc_poss,fwhmsy_sc,th=fwhm_th,cyclic=False,return_idx=True)
    out_idx5 = np.where(np.diff(sc_poss)==0)[0]
    out_idxes = sorted(list(set(list(out_idx0)+list(out_idx1)+list(out_idx2)+list(out_idx3)+list(out_idx4)+list(out_idx5))),reverse=True)
    for out_idx in out_idxes:
        del sc_poss[out_idx], Xs_sc[out_idx], Ys_sc[out_idx], fwhmsx_sc[out_idx], fwhmsy_sc[out_idx], intens_sc[out_idx], bgs_sc[out_idx]
        del Xs_sc_e[out_idx],Ys_sc_e[out_idx],fwhmsx_sc_e[out_idx],fwhmsy_sc_e[out_idx],intens_sc_e[out_idx],bgs_sc_e[out_idx]
    
    nil,nil,in_idx0,nil,nil,out_idx0=remove_outlier2(toptica_poss,toptica_poss,th=np.median(np.diff(toptica_poss)*1.5),cyclic=False,return_idx=True)
    nil,nil,in_idx1,nil,nil,out_idx1=remove_outlier2(toptica_poss,Xs_toptica,th=xy_th,cyclic=False,return_idx=True)
    nil,nil,in_idx2,nil,nil,out_idx2=remove_outlier2(toptica_poss,fwhmxs_toptica,th=fwhm_th,cyclic=False,return_idx=True)
    nil,nil,in_idx3,nil,nil,out_idx3=remove_outlier2(toptica_poss,Ys_toptica,th=xy_th,cyclic=False,return_idx=True)
    nil,nil,in_idx4,nil,nil,out_idx4=remove_outlier2(toptica_poss,fwhmys_toptica,th=fwhm_th,cyclic=False,return_idx=True)
    out_idx5 = np.where(np.diff(toptica_poss)==0)[0]
    out_idxes = sorted(list(set(list(out_idx0)+list(out_idx1)+list(out_idx2)+list(out_idx3)+list(out_idx4)+list(out_idx5))),reverse=True)
    for out_idx in out_idxes:
        del toptica_poss[out_idx], Xs_toptica[out_idx], Ys_toptica[out_idx], fwhmxs_toptica[out_idx], fwhmys_toptica[out_idx], intens_toptica[out_idx], bgs_toptica[out_idx]
        del Xs_toptica_e[out_idx],Ys_toptica_e[out_idx],fwhmxs_toptica_e[out_idx],fwhmys_toptica_e[out_idx],intens_toptica_e[out_idx],bgs_toptica_e[out_idx]
    
    sc_poss=np.array(sc_poss)
    Xs_sc=np.array(Xs_sc)
    Ys_sc=np.array(Ys_sc)
    fwhmsx_sc=np.array(fwhmsx_sc)
    fwhmsy_sc=np.array(fwhmsy_sc)
    intens_sc=np.array(intens_sc)
    bgs_sc=np.array(bgs_sc)
    toptica_poss=np.array(toptica_poss)
    Xs_toptica=np.array(Xs_toptica)
    Ys_toptica=np.array(Ys_toptica)
    fwhmxs_toptica=np.array(fwhmxs_toptica)
    fwhmys_toptica=np.array(fwhmys_toptica)
    intens_toptica=np.array(intens_toptica)
    bgs_toptica=np.array(bgs_toptica)
    
    poss_interp = np.linspace(max(min(sc_poss),min(toptica_poss)),min(max(sc_poss),max(toptica_poss)),100)
    sc_sort_idx = sc_poss.argsort()
    Xs_sc_fx = sp.interpolate.interp1d(sc_poss[sc_sort_idx],Xs_sc[sc_sort_idx],'cubic')
    Ys_sc_fx = sp.interpolate.interp1d(sc_poss[sc_sort_idx],Ys_sc[sc_sort_idx],'cubic')
    Xs_sc_interp = Xs_sc_fx(poss_interp)
    Ys_sc_interp = Ys_sc_fx(poss_interp)
    toptica_sort_idx = toptica_poss.argsort()
    Xs_toptica_fx = sp.interpolate.interp1d(toptica_poss[toptica_sort_idx],Xs_toptica[toptica_sort_idx],'cubic')
    Ys_toptica_fx = sp.interpolate.interp1d(toptica_poss[toptica_sort_idx],Ys_toptica[toptica_sort_idx],'cubic')
    Xs_toptica_interp = Xs_toptica_fx(poss_interp)
    Ys_toptica_interp = Ys_toptica_fx(poss_interp)
    matching_x_plane = poss_interp[get_nearest_idx_from_list(0,Xs_sc_interp-Xs_toptica_interp)]
    matching_y_plane = poss_interp[get_nearest_idx_from_list(0,Ys_sc_interp-Ys_toptica_interp)]    
    
    _fig = plt.figure('%s'%sample)
    _fig.suptitle(sample)
    x_fig = _fig.add_subplot(221)
    x_fig.set_title('horizontal')
    x_fig.set_ylabel('beam position, $\mu$m')
    x_fig.errorbar(sc_poss,Xs_sc,yerr=Xs_sc_e,ls='None',marker='o',capsize=5,C='C0',ecolor='C0',elinewidth=1,label='SC')
    x_fig.fill_between(sc_poss, Xs_sc-fwhmsx_sc/2, Xs_sc+fwhmsx_sc/2,alpha=0.5, edgecolor='C0', facecolor='C0')
    x_fig.errorbar(toptica_poss,Xs_toptica,yerr=Xs_toptica_e,ls='None',marker='o',capsize=5,C='C1',ecolor='C1',elinewidth=1,label='Toptica')
    x_fig.fill_between(toptica_poss, Xs_toptica-fwhmxs_toptica/2, Xs_toptica+fwhmxs_toptica/2,alpha=0.5, edgecolor='C1', facecolor='C1')
    x_fig.axvline(matching_x_plane,ls='--',color='Grey')
    x_fig.text(matching_x_plane,x_fig.get_ylim()[0],'%.2f'%matching_x_plane,horizontalalignment='left', verticalalignment='bottom')
    plt.grid()
    x_fig.legend()
    
    y_fig = _fig.add_subplot(222)
    y_fig.set_title('vertical')
    y_fig.set_ylabel('beam position, $\mu$m')
    y_fig.errorbar(sc_poss,Ys_sc,yerr=Ys_sc_e,ls='None',marker='o',capsize=5,C='C0',ecolor='C0',elinewidth=1)
    y_fig.fill_between(sc_poss, Ys_sc-fwhmsy_sc/2, Ys_sc+fwhmsy_sc/2,alpha=0.5, edgecolor='C0', facecolor='C0')
    y_fig.errorbar(toptica_poss,Ys_toptica,yerr=Ys_toptica_e,ls='None',marker='o',capsize=5,C='C1',ecolor='C1',elinewidth=1)
    y_fig.fill_between(toptica_poss, Ys_toptica-fwhmys_toptica/2, Ys_toptica+fwhmys_toptica/2,alpha=0.5, edgecolor='C1', facecolor='C1')
    y_fig.axvline(matching_y_plane,ls='--',color='Grey')
    y_fig.text(matching_y_plane,y_fig.get_ylim()[0],'%.2f'%matching_y_plane,horizontalalignment='left', verticalalignment='bottom')
    plt.grid()
    
    fwhmx_fig = _fig.add_subplot(425)
    fwhmx_fig.set_title('horizontal FWHM')
    fwhmx_fig.set_ylabel('FWHM, $\mu$m')
    fwhmx_fig.errorbar(sc_poss, fwhmsx_sc, yerr=fwhmsx_sc_e,ls='None',marker='o',capsize=5,C='C0',ecolor='C0',elinewidth=1)
    fwhmx_fig.errorbar(toptica_poss, fwhmxs_toptica, yerr=fwhmxs_toptica_e,ls='None',marker='o',capsize=5,C='C1',ecolor='C1',elinewidth=1)
    plt.grid()
    
    fwhmy_fig = _fig.add_subplot(426)
    fwhmy_fig.set_title('vertical FWHM')
    fwhmy_fig.set_ylabel('FWHM, $\mu$m')
    fwhmy_fig.errorbar(sc_poss, fwhmsy_sc, yerr=fwhmsy_sc_e,ls='None',marker='o',capsize=5,C='C0',ecolor='C0',elinewidth=1)
    fwhmy_fig.errorbar(toptica_poss, fwhmys_toptica, yerr=fwhmys_toptica_e,ls='None',marker='o',capsize=5,C='C1',ecolor='C1',elinewidth=1)
    plt.grid()
    
    inten_fig = _fig.add_subplot(427)
    inten_fig.set_title('beam power')
    inten_fig.set_ylabel('power, au')
    inten_fig.set_xlabel('focal plane position, $\mu$m')
    inten_fig.errorbar(sc_poss, intens_sc, yerr=intens_sc_e,ls='None',marker='o',capsize=5,C='C0',ecolor='C0',elinewidth=1)
    inten_fig.errorbar(toptica_poss, intens_toptica, yerr=intens_toptica_e,ls='None',marker='o',capsize=2,C='C1',ecolor='C1',elinewidth=1)
    plt.grid()
    
    bg_fig = _fig.add_subplot(428)
    bg_fig.set_title('background')
    bg_fig.set_ylabel('background, au')
    bg_fig.set_xlabel('focal plane position, $\mu$m')
    bg_fig.errorbar(sc_poss, bgs_sc, yerr=bgs_sc_e,ls='None',marker='o',capsize=2,C='C0',ecolor='C0',elinewidth=1)
    bg_fig.errorbar(toptica_poss, bgs_toptica, yerr=bgs_toptica_e,ls='None',marker='o',capsize=5,C='C1',ecolor='C1',elinewidth=1)
    plt.grid()
    
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.1)
    _fig.tight_layout(rect=(0,0,1,0.95))
    _fig.savefig(os.path.join(main_path,'%s.png'%sample))
    
    return matching_x_plane,matching_y_plane

def get_image_centroid_and_bounding_rect(img):
#    m = cv2.moments(img)
#    cx = int(m['m10']/m['m00']) #centre of image in x
#    cy = int(m['m01']/m['m00'])
    br = cv2.boundingRect(img.clip(254)-254)
    cx = int(br[0]+1.*br[2]/2)-1
    cy = int(br[1]+1.*br[3]/2)-1
    lx = br[2]*2 #length of x to be bounded
    ly = br[3]*2
    minx = cx - lx/2 #after cropped, local origin location in the global one
    miny = cy - ly/2
    return cx, cy, lx, ly, minx, miny

def crop_img(img,cx,cy,lx,ly):
    return img[int(cy-ly/2):int(cy+ly/2),int(cx-lx/2):int(cx+lx/2)]

def crop_to_centroid(img):
    cx, cy, lx, ly, minx, miny = get_image_centroid_and_bounding_rect(img)
    new_img = crop_img(img,cx,cy,lx,ly)
    return new_img, minx, miny

def fill_in_hole(img,sat=255):
    new_img = np.zeros(img.shape)
    for r,row in enumerate(img):
        new_row = copy.copy(img[r])
        sat_idxes = np.where(row == sat)[0]
#        idx_diff = np.diff(sat_idxes)
#        if (np.diff(idx_diff) > 1).any():
        if len(sat_idxes) > 1:
            for i in range(sat_idxes[0],sat_idxes[-1]+1):
                new_row[i] = sat
        new_img[r] = new_row
    return new_img