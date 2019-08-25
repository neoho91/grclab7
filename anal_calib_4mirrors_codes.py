# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 13:52:15 2018

@author: Neo
"""

import numpy as np
import scipy as sp
import os
import sys
import time
import matplotlib.pyplot as plt
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from anal_calibrate_circular_pol3_1560 import *
from fit_calib_4mirrors import *
from read_TH_temperature import *
from fit_mallus import *

def anal_calib_4mirrors(sample,get_temperature=True,plot=True):
    main_path=os.path.join(r'D:\Nonlinear_setup\Experimental_data\calibrate_4mirrors',sample)
    try:
        if get_temperature:
            timestamps = np.load(os.path.join(main_path,'timestamps.npy'))
            start_time_nopol,end_time_nopol,start_time_para,end_time_para,start_time_apara,end_time_apara = timestamps
            curr_temp_nopol = get_temp_in_range(start_time_nopol,end_time_nopol)
            curr_temp_para = get_temp_in_range(start_time_para,end_time_para)
            curr_temp_apara = get_temp_in_range(start_time_apara,end_time_apara)
            try:
                old_temp_nopol = np.load(os.path.join(main_path,'temp_nopol.npy'))
                old_temp_para = np.load(os.path.join(main_path,'temp_para.npy'))
                old_temp_apara = np.load(os.path.join(main_path,'temp_apara.npy'))
                if len(old_temp_nopol) < len(curr_temp_nopol):
                    raise IOError
                if len(old_temp_para) < len(curr_temp_para):
                    raise IOError
                if len(old_temp_apara) < len(curr_temp_apara):
                    raise IOError
            except IOError:
                np.save(os.path.join(main_path,'temp_nopol'),curr_temp_nopol)
                np.save(os.path.join(main_path,'temp_para'),curr_temp_para)
                np.save(os.path.join(main_path,'temp_apara'),curr_temp_apara)
    except:
        print('Unable to read TH+ logger temperature.')
    ana_angs = np.load(os.path.join(main_path,'ana_angs.npy'))
    powers_nopol = np.load(os.path.join(main_path,'powers_nopol.npy'))
    powers_para = np.load(os.path.join(main_path,'powers_para.npy'))
    powers_apara = np.load(os.path.join(main_path,'powers_apara.npy'))
    try:
        temps_nopol = np.load(os.path.join(main_path,'temp_nopol.npy'))
        temps_para = np.load(os.path.join(main_path,'temp_para.npy'))
        temps_apara = np.load(os.path.join(main_path,'temp_apara.npy'))
        temp_nopol = np.mean(get_temp_only(temps_nopol))
        temp_nopol_e = np.std(get_temp_only(temps_nopol))
        temp_para = np.mean(get_temp_only(temps_para))
        temp_para_e = np.std(get_temp_only(temps_para))
        temp_apara = np.mean(get_temp_only(temps_apara))
        temp_apara_e = np.std(get_temp_only(temps_apara))
        title = '%s\nnopol = %.2f $\pm$ %.2f C, para = %.2f $\pm$ %.2f C, apara = %.2f $\pm$ %.2f C'%(sample,temp_nopol,temp_nopol_e,temp_para,temp_para_e,temp_apara,temp_apara_e)
    except:
        title = sample
    popt, perr = calib_4mirror_fitter(ana_angs,powers_nopol,powers_para,powers_apara,title=title,plot=plot)
    tilt = popt[3]
    tilt_e = perr[3]
    retard = popt[4]
    retard_e = perr[4]
    
    return (tilt,tilt_e,retard,retard_e,temp_nopol,temp_nopol_e,temp_para,temp_para_e,temp_apara,temp_apara_e)

def anal_all_calib_4mirrors(plot=False):
    main_path = r'D:\Nonlinear_setup\Experimental_data\calibrate_4mirrors'
    samples = os.listdir(main_path)
    samples.remove('old')
    global tilts,tilts_e,retards,retards_e,temps,temps_e
    tilts,tilts_e,retards,retards_e,temps,temps_e = [],[],[],[],[],[]
    for sample in samples:
        tilt,tilt_e,retard,retard_e,temp_nopol,temp_nopol_e,temp_para,temp_para_e,temp_apara,temp_apara_e = anal_calib_4mirrors(sample,get_temperature=False,plot=plot)
        temp = np.mean([temp_nopol,temp_para,temp_apara])
        temp_e = np.std([temp_nopol,temp_para,temp_apara]) + np.sqrt(np.sum(np.square([temp_nopol_e,temp_para_e,temp_apara_e])))
        temps.append(temp)
        temps_e.append(temp_e)
        tilts.append(tilt)
        tilts_e.append(tilt_e)
        retards.append(retard)
        retards_e.append(retard_e)
    temps = np.array(temps)
    temps_e = np.array(temps_e)
    tilts = np.array(tilts)
    tilts_e = np.array(tilts_e)
    retards = np.array(retards)
    retards_e = np.array(retards_e)
    
    _fig = plt.figure('all calib_4mirrors')
    fig = _fig.add_subplot(111)
    fig.errorbar(temps,tilts,xerr=temps_e,yerr=tilts_e,capsize=2,elinewidth=1,lw=0,marker='o',color='C0')
    fig.set_xlabel('mean temperature, degC')
    fig.set_ylabel('tilt, deg',color='C0')
    fig.grid()
    fig2 = fig.twinx()
    fig2.errorbar(temps,retards,xerr=temps_e,yerr=retards_e,capsize=2,elinewidth=1,lw=0,marker='o',color='C1')
    fig2.set_ylabel('retard, wave',color='C1')
    _fig.suptitle('all calib_4mirrors')
    fig2.format_coord = make_format(fig2, fig)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.pause(1e-3)
    return sp.interpolate.interp1d(temps,tilts,'cubic'),sp.interpolate.interp1d(temps,retards,'cubic')

def anal_single_background_check_sc_circular_polarization(npz_path,get_temperature=True,plot=True):
    with np.load(npz_path) as data:
        start_time = data['start_time']
        end_time = data['end_time']
        gains = data['gains']
        theta_errs = data['theta_errs']
        thetas = data['thetas']
        freqs = data['freqs']
        Xerrs = data['Xerrs']*1e6
        Xs = data['Xs']*1e6
    try:
        if get_temperature:
            curr_temp = get_temp_in_range(start_time.item(),end_time.item())
            try:
#                raise IOError
                with np.load(npz_path) as data:
                    old_temp = data['temp']
                if len(old_temp) < len(curr_temp):
                    raise IOError
            except (IOError, KeyError):
                np.savez(npz_path,
                 start_time=start_time,
                 end_time=end_time,
                 Xs=Xs*1e-6,
                 Xerrs=Xerrs*1e-6,
                 thetas=thetas,
                 theta_errs=theta_errs,
                 gains=gains,
                 freqs=freqs,
                 temp=curr_temp)
            curr_temp = get_temp_only(curr_temp)
        else:
            with np.load(npz_path) as data:
                curr_temp = data['temp']
            curr_temp = get_temp_only(curr_temp)
    except:
        curr_temp=[0]
    temp = np.mean(curr_temp)
    temp_e = np.std(curr_temp)
    curr_time = np.mean([start_time.item(),end_time.item()])
    return curr_time,temp,temp_e, mallus_fitter(np.arange(0,360.01,10),Xs,plot=plot,title='%.2f $\pm$ %.2f $^o$C'%(temp,temp_e))

def anal_background_check_sc_circular_polarization(sample,get_temperature=True):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\background_check_sc_circular_polarization',sample)
    npz_files = list(filter(lambda name: name.endswith('.npz'),os.listdir(main_path)))
    def get_num_from_name(name):
        try:
            return float(name.split('_')[1].split('.npz')[0])/10
        except:
            return float(name.split('_')[1].split('.npy')[0])/10
    npz_files.sort(key=get_num_from_name)
    global As, As_e, phis, phis_e, Rs, Rs_e, temps, temps_e, times, valid_temps, valid_temps_e, valid_times
    As, As_e, phis, phis_e, Rs, Rs_e, temps, temps_e, times = [],[],[],[],[],[],[],[],[]
    for npz_file in npz_files:
        curr_time,temp,temp_e,((A,phi,R),(A_e,phi_e,R_e)) = anal_single_background_check_sc_circular_polarization(os.path.join(main_path,npz_file),get_temperature=get_temperature,plot=False)
        As.append(A)
        As_e.append(A_e)
        phis.append(phi)
        phis_e.append(phi_e)
        Rs.append(R)
        Rs_e.append(R_e)
        if temp == 0:
            temp = 999
        temps.append(temp)
        temps_e.append(temp_e)
        times.append(curr_time)
    As = np.array(As)
    As_e = np.array(As_e)
    phis = np.array(put_angles_to_same_range(phis,180))%180
    phis_e = np.array(phis_e)
    Rs = np.array(Rs)
    Rs_e = np.array(Rs_e)
    temps = np.array(temps)
    temps_e = np.array(temps_e)
    init_time = min(times)
    times = (np.array(times) - init_time)/60.
    
    valid_temps_e,valid_temps,nil,nil = remove_outlier(temps_e,temps,998)
    valid_times,valid_temps,nil,nil = remove_outlier(times,temps,998)
    
    _fig = plt.figure(sample)
    plt.get_current_fig_manager().window.showMaximized()
    fig = _fig.add_subplot(411)
    fig.errorbar(times,As,yerr=As_e,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C0')
    fig.set_ylabel('Fitted amplitude, $\mu$V')
    plt.grid()
    fig2 = _fig.add_subplot(412)
    fig2.errorbar(times,phis,yerr=phis_e,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C1')
    fig2.set_ylabel('Fitted max angle, deg')
    plt.grid()
    fig3 = _fig.add_subplot(413)
    fig3.errorbar(times,Rs,yerr=Rs_e,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C2')
    fig3.set_ylabel('Fitted min/max ratio')
    plt.grid()
    fig4 = _fig.add_subplot(414)
    fig4.errorbar(valid_times,valid_temps,yerr=valid_temps_e,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C3')
    fig4.set_ylabel('Temperature, $^o$C')
    fig4.set_xlabel('Time, min')
    fig4.text(0,fig4.get_ylim()[0],'%s'%(time.strftime("%d %b %Y %H:%M:%S", time.localtime(init_time))),horizontalalignment='left', verticalalignment='bottom')
    plt.grid()
    _fig.suptitle(sample)
    plt.pause(0.1)
    plt.tight_layout(rect=(0,0,1,0.95))
    plt.ion()
    
    def onclick(event):
        minute = event.xdata
        idx = get_nearest_idx_from_list(minute,times)
        path = os.path.join(main_path,'data_%i.npz'%(idx+1))
        print(anal_single_background_check_sc_circular_polarization(path,get_temperature=False,plot=True))
        print ''
        plt.suptitle('%.2f minutes'%minute)
        minute_x = times[idx]
        fig.axvline(minute_x,color='Grey')
        fig2.axvline(minute_x,color='grey')
        fig3.axvline(minute_x,color='grey')
        fig4.axvline(minute_x,color='grey')
        plt.draw_all()
    
    _fig.canvas.mpl_connect('button_press_event', onclick)
    