#%%
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
from neo_common_code import *
#from anal_calib_4mirrors_codes import *
from correction_factor_4mirrors import *
from fit_gaussian import *
hero_pixel_wavelengths = np.load(r'D:/WMP_setup/Python_codes/hero_pixel_wavelengths.npy')
try:
    reVP1test780coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test780coeff.npy')
    reVP1test390coeff = np.load(r'D:\Nonlinear_setup\Python_codes\reVP1test390coeff.npy')
except:
    print('reVP1test calibration file(s) not found.')
    reVP1test780coeff = np.array([1])
    reVP1test390coeff = np.array([1])
from NPBS_TR_ratio import *
from fit_poly import *
from cv2_rolling_ball import subtract_background_rolling_ball
import joblib
from joblib import Parallel,delayed
import threading

#%%
#--------------------------------------------------#
#Code for analysis of the results

def anal_pSHG_fix_alpha(sample,get_temperature=True,CC4m_param=(-2.9,0.252)):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\pSHG_fix_alpha',sample)
    with np.load(os.path.join(main_path,'data.npz')) as data:
        theta_errs = data['theta_errs']
        ana_angs = data['ana_angs']
        gains = data['gains']
        thetas = data['thetas']
        freqs = data['freqs']
        Xerrs = data['Xerrs']
        Xs = data['Xs']
    try:
        if get_temperature:
            timestamps = np.load(os.path.join(main_path,'timestamps.npy'))
            start_time,end_time = timestamps
            curr_temp = get_temp_in_range(start_time,end_time)
            try:
                old_temp = np.load(os.path.join(main_path,'temp.npy'))
                if len(old_temp) < len(curr_temp):
                    raise IOError
            except IOError:
                np.save(os.path.join(main_path,'temp'),curr_temp)
    except:
        print('Unable to read TH+ logger temperature.')
        
    temps = get_temp_only(np.load(os.path.join(main_path,'temp.npy')) )
    temp = np.mean(temps)
    temp_e = np.std(temps)
    global alphas,data780,max_alpha,d,alphas_spec,mean780s
    data780 = Xs
    alphas = ana_angs
    
    alphas_spec,mean780s,in_idxs,alphas_spec_out,mean780s_out,out_idxs = remove_outlier2(ana_angs,Xs,return_idx=True)
    error780s = []
    for idx in in_idxs:
        error780s.append(Xerrs[idx])
    if len(mean780s_out) > 0:
        print('Outlier(s) for lock-in removed:\n    ana_ang = %s\n    lock-in = %s'%(alphas_spec_out,mean780s_out))
         
    _fig = plt.figure(r'%s'%(sample))
    fig = _fig.add_subplot(141,projection='polar')
    fig.errorbar(alphas_spec[:len(mean780s)]/180.*np.pi,mean780s,yerr=error780s,marker='o',capsize=2,label='SHG',lw=0,elinewidth=0.5,zorder=0)
    fig.set_title(r'SHG @ 780 nm, 4mirrors @ %.3f $\pm$ %.3f degC'%(temp,temp_e))
    fig.set_xticks(np.arange(0,360,60)/180.*np.pi)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    try:
        fitted_X = np.linspace(0,360,300)
        fit_results_SHG = multiple_petals_n7_fitter(alphas_spec,mean780s) #fitting ignores the outliers
        fig.plot(fitted_X/180.*np.pi,multiple_petals_n7(fitted_X,*fit_results_SHG[0]),label='fit n7 SHG')
        
        corrected_SHGs = correcting_2petals(alphas_spec,mean780s)
        fit_results_SHG_2p = _two_petals_fitter(alphas_spec,corrected_SHGs)
        fig.plot(alphas_spec[:len(mean780s)]/180.*np.pi,corrected_SHGs,'v',label='2p SHG')
        fig.plot(fitted_X/180.*np.pi,two_petals(fitted_X,*fit_results_SHG_2p[0]),label='fit 2p SHG')
        SHG_max_alpha = fit_results_SHG_2p[0][1]
        SHG_max_alpha_e = fit_results_SHG_2p[1][1]
        t = 'raw data\nA = %.2e $\pm$ %.2e\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg\nR = %.2e $\pm$ %.2e'%tuple(np.array(tuple(zip(fit_results_SHG_2p[0],fit_results_SHG_2p[1]))).flatten())
        fig.set_title(t)
        handles,labels = fig.get_legend_handles_labels()
        handles = [handles[3],handles[0],handles[1],handles[2]]
        labels = [labels[3],labels[0],labels[1],labels[2]]
        fig.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=2)
        
        fig3 = _fig.add_subplot(142)
        fig3.bar(range(8),fit_results_SHG[0][:8],yerr=fit_results_SHG[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
        fig3.axhline(0,color='k')
        fig3.set_ylabel('Amplitude, au',color='C0')
        fig3.set_title('Fitting summary')
        fig3.set_xlabel('Harmonics')
        fig3_2 = fig3.twinx()
        fig3_2.errorbar(range(1,8),fit_results_SHG[0][8:],ls='None',marker='o',yerr=fit_results_SHG[1][8:],capsize=5,C='C1',ecolor='k')
        fig3_2.grid()
        fig3_2.set_ylabel('Angle, deg',color='C1')
        fig3_2.format_coord = make_format(fig3_2, fig3)        
        
        if CC4m_param == 'auto':
            pass
        else:
            phi,ret=CC4m_param
        if fit_results_SHG_2p[0][2] < 0:
            _R = 0
        else:
            _R = fit_results_SHG_2p[0][2]
        CCs = CC_4mirrors(alphas_spec,fit_results_SHG_2p[0][1],_R,phi,ret)
        data780_bef4m = CCs*mean780s
        alphas_bef,data780_bef,in_idxs,alphas_bef_out,data780_bef_out,out_idxs = remove_outlier2(alphas_spec,data780_bef4m,return_idx=True)
        error780s_bef = []
        for idx in in_idxs:
            error780s_bef.append(error780s[idx])
        if len(alphas_bef_out) > 0:
            print('Outlier(s) for before4m removed:\n    ana_ang = %s\n    lock-in = %s'%(alphas_bef_out,data780_bef_out))
        fig4 = _fig.add_subplot(143,projection='polar')
        fig4.errorbar(alphas_bef[:len(data780_bef)]/180.*np.pi,data780_bef,yerr=error780s_bef,marker='o',capsize=2,label='bef 4mirrors',lw=0,elinewidth=0.5,zorder=0)
        fitted_X = np.linspace(0,360,300)
        fit_results_SHG_bef4m = multiple_petals_n7_fitter(alphas_bef,data780_bef)
        fig4.plot(fitted_X/180.*np.pi,multiple_petals_n7(fitted_X,*fit_results_SHG_bef4m[0]),label='fit n7 bef 4mirrors')
        global corrected_SHGs_bef4m
        corrected_SHGs_bef4m = correcting_2petals(alphas_bef,data780_bef)
        fig4.plot(alphas_bef[:len(data780_bef)]/180.*np.pi,corrected_SHGs_bef4m,'v',label='2p bef 4mirrors')
        fit_results_SHG_2p_bef4m = _two_petals_fitter(alphas_bef,corrected_SHGs_bef4m)
        fig4.plot(fitted_X/180.*np.pi,two_petals(fitted_X,*fit_results_SHG_2p_bef4m[0]),label='fit 2p bef 4mirrors')
        t4 = 'A = %.2e $\pm$ %.2e\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg\nR = %.2e $\pm$ %.2e'%tuple(np.array(tuple(zip(fit_results_SHG_2p_bef4m[0],fit_results_SHG_2p_bef4m[1]))).flatten())
        t4 = 'before 4 mirrors ($\phi$ = %.3f deg, ret = %.4f wave)\n%s'%(phi,ret,t4)
        fig4.set_title(t4)
        fig4.set_xticks(np.arange(0,360,60)/180.*np.pi)
        handles,labels = fig4.get_legend_handles_labels()
        handles = [handles[3],handles[0],handles[1],handles[2]]
        labels = [labels[3],labels[0],labels[1],labels[2]]
        fig4.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=2)
        
        fig5 = _fig.add_subplot(144)
        fig5.bar(range(8),fit_results_SHG_bef4m[0][:8],yerr=fit_results_SHG_bef4m[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
        fig5.axhline(0,color='k')
        fig5.set_ylabel('Amplitude, au',color='C0')
        fig5.set_title('Before 4 mirrors fitting summary')
        fig5.set_xlabel('Harmonics')
        fig5_2 = fig5.twinx()
        fig5_2.errorbar(range(1,8),fit_results_SHG_bef4m[0][8:],ls='None',marker='o',yerr=fit_results_SHG_bef4m[1][8:],capsize=5,C='C1',ecolor='k')
        fig5_2.grid()
        fig5_2.set_ylabel('Angle, deg',color='C1')
        fig5_2.format_coord = make_format(fig5_2, fig5)
        
        _fig.suptitle('%s 4mirrors @ %.3f $\pm$ %.3f degC '%(sample,temp,temp_e))
        plt.pause(1e-2)
        _fig.tight_layout(rect=(0,0,1,0.95))
        _fig.savefig(os.path.join(main_path,'%s.png'%sample))
    
        d = fit_results_SHG_2p_bef4m[0][1]  
        max_alpha.append(d)
        
        return SHG_max_alpha, SHG_max_alpha_e, fit_results_SHG, fit_results_SHG_2p, fit_results_SHG_bef4m, fit_results_SHG_2p_bef4m
    
    except Exception as e:
        print(e)
        print('Failed to fit %s'%sample)
#%%

def anal_rotA_meas_lockin(sample):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\rotA_meas_lockin',sample)
    with np.load(os.path.join(main_path,'data.npz')) as data:
        theta_errs = data['theta_errs']
        ana_angs = data['ana_angs']
        gains = data['gains']
        thetas = data['thetas']
        freqs = data['freqs']
        Xerrs = data['Xerrs']
        Xs = data['Xs']

    global alphas,data780,max_alpha,d,alphas_spec,mean780s
    data780 = Xs
    alphas = ana_angs
    
    alphas_spec,mean780s,in_idxs,alphas_spec_out,mean780s_out,out_idxs = remove_outlier2(ana_angs,Xs,return_idx=True)
    error780s = []
    for idx in in_idxs:
        error780s.append(Xerrs[idx])
    if len(mean780s_out) > 0:
        print('Outlier(s) for lock-in removed:\n    ana_ang = %s\n    lock-in = %s'%(alphas_spec_out,mean780s_out))
         
    _fig = plt.figure(r'%s'%(sample))
    fig = _fig.add_subplot(121,projection='polar')
    fig.errorbar(alphas_spec[:len(mean780s)]/180.*np.pi,mean780s,yerr=error780s,marker='o',capsize=2,label='SHG',lw=0,elinewidth=0.5,zorder=0)
    fig.set_title(r'SHG @ 780 nm')
    fig.set_xticks(np.arange(0,360,60)/180.*np.pi)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    try:
        fitted_X = np.linspace(0,360,300)
        fit_results_SHG = multiple_petals_n7_fitter(alphas_spec,mean780s) #fitting ignores the outliers
        fig.plot(fitted_X/180.*np.pi,multiple_petals_n7(fitted_X,*fit_results_SHG[0]),label='fit n7 SHG')
        
        corrected_SHGs = correcting_2petals(alphas_spec,mean780s)
        fit_results_SHG_2p = _two_petals_fitter(alphas_spec,corrected_SHGs)
        fig.plot(alphas_spec[:len(mean780s)]/180.*np.pi,corrected_SHGs,'v',label='2p SHG')
        fig.plot(fitted_X/180.*np.pi,two_petals(fitted_X,*fit_results_SHG_2p[0]),label='fit 2p SHG')
        SHG_max_alpha = fit_results_SHG_2p[0][1]
        SHG_max_alpha_e = fit_results_SHG_2p[1][1]
        t = 'raw data\nA = %.2e $\pm$ %.2e\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg\nR = %.2e $\pm$ %.2e'%tuple(np.array(tuple(zip(fit_results_SHG_2p[0],fit_results_SHG_2p[1]))).flatten())
        fig.set_title(t)
        handles,labels = fig.get_legend_handles_labels()
        handles = [handles[3],handles[0],handles[1],handles[2]]
        labels = [labels[3],labels[0],labels[1],labels[2]]
        fig.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=2)
        
        fig3 = _fig.add_subplot(122)
        fig3.bar(range(8),fit_results_SHG[0][:8],yerr=fit_results_SHG[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
        fig3.axhline(0,color='k')
        fig3.set_ylabel('Amplitude, au',color='C0')
        fig3.set_title('Fitting summary')
        fig3.set_xlabel('Harmonics')
        fig3_2 = fig3.twinx()
        fig3_2.errorbar(range(1,8),fit_results_SHG[0][8:],ls='None',marker='o',yerr=fit_results_SHG[1][8:],capsize=5,C='C1',ecolor='k')
        fig3_2.grid()
        fig3_2.set_ylabel('Angle, deg',color='C1')
        fig3_2.format_coord = make_format(fig3_2, fig3)        
        
        return SHG_max_alpha, SHG_max_alpha_e, fit_results_SHG, fit_results_SHG_2p
    
    except Exception as e:
        print(e)
        print('Failed to fit %s'%sample)
    
    finally:
        plt.pause(0.5)
        _fig.savefig(os.path.join(main_path,'%s.png'%sample))
        
#%%

def anal_pSHG_with_andor(sample,min_wl=760,max_wl=790):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\pSHG_with_andor',sample)
    global wls,ana_angs,specs,d,bg_specs,alphas_spec,mean780s,error780s
    with np.load(os.path.join(main_path,'data.npz')) as data:
        wls = data['wls']
        ana_angs = data['ana_angs']
        specs = data['specs']
        try:
            bg_specs = data['bg_specs']
        except KeyError:
            specs_dev = data['specs_dev']
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)

    alphas_spec = ana_angs
    try:
        mean780s = []
        error780s = []
        for i,spec in enumerate(specs):
            curr_specs = np.array(list(map(lambda x: x-bg_specs[i],spec))) #each spectrum - spectrum without 1550
            curr_sums = np.array(list(map(lambda x: np.sum(x[min_idx:max_idx]), curr_specs))) #integrate in ROI
            mean780s.append(np.mean(curr_sums))
            error780s.append(np.std(curr_sums)/np.sqrt(len(curr_sums)))
        mean780s = np.array(mean780s)
        error780s = np.array(error780s)
    except:
        mean780s = np.array(list(map(lambda spec:           np.sum(spec[min_idx:max_idx] - np.median(np.append(spec[:min_idx],spec[max_idx:])))        ,specs)))
        error780s = np.array(list(map(lambda spec_dev:           np.sum(spec_dev)                    ,specs_dev)))
    
    _fig = plt.figure(r'%s'%(sample))
    fig = _fig.add_subplot(121,projection='polar')
    fig.errorbar(alphas_spec[:len(mean780s)]/180.*np.pi,mean780s,yerr=error780s,marker='o',capsize=2,label='SHG',lw=0,elinewidth=0.5,zorder=0)
    fig.set_title(r'SHG @ 780 nm')
    fig.set_xticks(np.arange(0,360,60)/180.*np.pi)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    try:
        fitted_X = np.linspace(0,360,300)
        fit_results_SHG = multiple_petals_n7_fitter(alphas_spec[:len(mean780s)],mean780s) #fitting ignores the outliers
        fig.plot(fitted_X/180.*np.pi,multiple_petals_n7(fitted_X,*fit_results_SHG[0]),label='fit n7 SHG')
        
        corrected_SHGs = correcting_2petals(alphas_spec[:len(mean780s)],mean780s)
        fit_results_SHG_2p = _two_petals_fitter(alphas_spec[:len(mean780s)],corrected_SHGs)
        fig.plot(alphas_spec[:len(mean780s)]/180.*np.pi,corrected_SHGs,'v',label='2p SHG')
        fig.plot(fitted_X/180.*np.pi,two_petals(fitted_X,*fit_results_SHG_2p[0]),label='fit 2p SHG')
        SHG_max_alpha = fit_results_SHG_2p[0][1]
        SHG_max_alpha_e = fit_results_SHG_2p[1][1]
        t = 'raw data\nA = %.2e $\pm$ %.2e\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg\nR = %.2e $\pm$ %.2e'%tuple(np.array(tuple(zip(fit_results_SHG_2p[0],fit_results_SHG_2p[1]))).flatten())
        fig.set_title(t)
        handles,labels = fig.get_legend_handles_labels()
        handles = [handles[3],handles[0],handles[1],handles[2]]
        labels = [labels[3],labels[0],labels[1],labels[2]]
        fig.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=2)
        
        fig3 = _fig.add_subplot(122)
        fig3.bar(range(8),fit_results_SHG[0][:8],yerr=fit_results_SHG[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
        fig3.axhline(0,color='k')
        fig3.set_ylabel('Amplitude, au',color='C0')
        fig3.set_title('Fitting summary')
        fig3.set_xlabel('Harmonics')
        fig3_2 = fig3.twinx()
        fig3_2.errorbar(range(1,8),fit_results_SHG[0][8:],ls='None',marker='o',yerr=fit_results_SHG[1][8:],capsize=5,C='C1',ecolor='k')
        fig3_2.grid()
        fig3_2.set_ylabel('Angle, deg',color='C1')
        fig3_2.format_coord = make_format(fig3_2, fig3)        
        
        return SHG_max_alpha, SHG_max_alpha_e, fit_results_SHG, fit_results_SHG_2p
    
    except Exception as e:
        print(e)
        print('Failed to fit %s'%sample)
    
    finally:
        plt.pause(0.5)
        _fig.savefig(os.path.join(main_path,'%s.png'%sample))

#%%

def anal_rotA_meas_lockin_XY(sample):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\rotA_meas_lockin',sample)
    with np.load(os.path.join(main_path,'data.npz')) as data:
        ana_angs = data['ana_angs']
        Xs = data['Xs']
        Xerrs = data['Xerrs']
        Ys = data['Ys']
        Yerrs = data['Yerrs']

    global alphas,data780,max_alpha,d,alphas_spec,mean780s
    alphas = ana_angs
    data780 = Xs

    alphas_spec,mean780s,in_idxs,alphas_spec_out,mean780s_out,out_idxs = remove_outlier2(ana_angs,Xs,return_idx=True)
    error780s = []
    for idx in in_idxs:
        error780s.append(Xerrs[idx])
    if len(mean780s_out) > 0:
        print('Outlier(s) for lock-in removed:\n    ana_ang = %s\n    lock-in = %s'%(alphas_spec_out,mean780s_out))
         
    _fig = plt.figure(r'%s'%(sample))
    fig = _fig.add_subplot(121,projection='polar')
    fig.errorbar(alphas_spec[:len(mean780s)]/180.*np.pi,mean780s,yerr=error780s,marker='o',capsize=2,label='SHG',lw=0,elinewidth=0.5,zorder=0)
    fig.set_title(r'SHG @ 780 nm')
    fig.set_xticks(np.arange(0,360,60)/180.*np.pi)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    try:
        fitted_X = np.linspace(0,360,300)
        fit_results_SHG = multiple_petals_n7_fitter(alphas_spec,mean780s) #fitting ignores the outliers
        fig.plot(fitted_X/180.*np.pi,multiple_petals_n7(fitted_X,*fit_results_SHG[0]),label='fit n7 SHG')
        
        corrected_SHGs = correcting_2petals(alphas_spec,mean780s)
        fit_results_SHG_2p = _two_petals_fitter(alphas_spec,corrected_SHGs)
        fig.plot(alphas_spec[:len(mean780s)]/180.*np.pi,corrected_SHGs,'v',label='2p SHG')
        fig.plot(fitted_X/180.*np.pi,two_petals(fitted_X,*fit_results_SHG_2p[0]),label='fit 2p SHG')
        SHG_max_alpha = fit_results_SHG_2p[0][1]
        SHG_max_alpha_e = fit_results_SHG_2p[1][1]
        t = 'raw data\nA = %.2e $\pm$ %.2e\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg\nR = %.2e $\pm$ %.2e'%tuple(np.array(tuple(zip(fit_results_SHG_2p[0],fit_results_SHG_2p[1]))).flatten())
        fig.set_title(t)
        handles,labels = fig.get_legend_handles_labels()
        handles = [handles[3],handles[0],handles[1],handles[2]]
        labels = [labels[3],labels[0],labels[1],labels[2]]
        fig.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=2)
        
        fig3 = _fig.add_subplot(122)
        fig3.bar(range(8),fit_results_SHG[0][:8],yerr=fit_results_SHG[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
        fig3.axhline(0,color='k')
        fig3.set_ylabel('Amplitude, au',color='C0')
        fig3.set_title('Fitting summary')
        fig3.set_xlabel('Harmonics')
        fig3_2 = fig3.twinx()
        fig3_2.errorbar(range(1,8),fit_results_SHG[0][8:],ls='None',marker='o',yerr=fit_results_SHG[1][8:],capsize=5,C='C1',ecolor='k')
        fig3_2.grid()
        fig3_2.set_ylabel('Angle, deg',color='C1')
        fig3_2.format_coord = make_format(fig3_2, fig3)        
        
        return SHG_max_alpha, SHG_max_alpha_e, fit_results_SHG, fit_results_SHG_2p
    
    except Exception as e:
        print(e)
        print('Failed to fit %s'%sample)
    
    finally:
        plt.pause(0.5)
        _fig.savefig(os.path.join(main_path,'%s.png'%sample))

#%%
def VP2_mapping_SHG_anal(sample,min_wl=757.9,max_wl=814.4,min_wl_bg=814.4,max_wl_bg=870.5,show_me_all_specs_to_choose_wl_range=False,normalize_SHG_by_pm_with_power=0):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\VP2_mapping_SHG',sample)
    data = np.load(os.path.join(main_path,'data.npy'))
    spec_wl, BCKGND_SPEC = data[0]
    data = data[1:]
    def get_spec(datum):
        return datum[2]
    def get_pm_data(datum):
        return datum[1]
    def get_x(datum):
        return datum[0][0]
    def get_z(datum):
        return datum[0][1]
    all_Xs = np.load(os.path.join(main_path,'Xs.npy'))
    all_Zs = np.load(os.path.join(main_path,'Zs.npy'))
    resol=all_Xs[1]-all_Xs[0]
    global SHG_data, pm_data, all_specs_for_alpha
    
    if show_me_all_specs_to_choose_wl_range:
        plt.ion()
        all_specs = list(map(get_spec,data))        
        sums = list(np.sum(all_specs,axis=1))
        fig = plt.figure('%s choose wavelength ranges'%sample)
        ax=fig.add_subplot(111)
        ax.set_xlabel('Wavelength, nm')
        for spec in all_specs:
            ax.plot(spec_wl,spec)
#        ax.plot(spec_wl,all_specs[sums.index(np.max(sums))])
        fig.canvas.mpl_connect('key_press_event', onclick)
        ax.set_title('select min wavelength to sum\nmouse over and press enter to choose')
        plt.pause(1e-6)
        while not plt.waitforbuttonpress():
            pass
        min_wl = _wl_chosen[0]
        ax.axvline(min_wl)
        
        ax.set_title('min = %.2f nm. Now select max wavelength to sum\nmouse over and press enter to choose'%min_wl)
        plt.pause(1e-6)
        while not plt.waitforbuttonpress():
            pass
        max_wl = _wl_chosen[0]
        ax.axvline(max_wl)
        
        ax.set_title('min = %.2f nm, max = %.2f nm. Now go to console'%(min_wl,max_wl))
        bg_min_plot = ax.axvline(min_wl_bg)
        bg_max_plot = ax.axvline(max_wl_bg)
        plt.pause(1e-6)
        ans = raw_input('Use default background (814 - 870 nm)? Enter "n" to choose. ')
        if ans == 'n':
            ax.set_title('select min background wavelength to sum\nmouse over and press enter to choose')
            plt.pause(1e-6)
            while not plt.waitforbuttonpress():
                pass
            min_wl_bg = _wl_chosen[0]
            bg_min_plot.set_xdata(min_wl_bg)
            
            ax.set_title('min = %.2f nm. Now select max background wavelength to sum\nmouse over and press enter to choose'%min_wl_bg)
            plt.pause(1e-6)
            while not plt.waitforbuttonpress():
                pass
            max_wl_bg = _wl_chosen[0]
            bg_max_plot.set_xdata(max_wl_bg)
            ax.set_title('min = %.2f nm, max = %.2f nm. Processing...'%(min_wl_bg,max_wl_bg))
            plt.pause(1e-6)
    
    print('min, max wavelength = %.2f, %.2f nm'%(min_wl,max_wl))
    print('min, max background wavelength = %.2f, %.2f nm'%(min_wl_bg,max_wl_bg))
    print('Processing... please wait...')
    
    min_idx=get_nearest_idx_from_list(min_wl,spec_wl)
    max_idx=get_nearest_idx_from_list(max_wl,spec_wl)
    min_idx_bg=get_nearest_idx_from_list(min_wl_bg,spec_wl)
    max_idx_bg=get_nearest_idx_from_list(max_wl_bg,spec_wl)
    
    SHG_data = []
    pm_data = []
    Zs = list(map(get_z,data))
    Zs = list(set(Zs))
    Zs.sort()
    Xs = list(map(get_x,data))
    Xs = list(set(Xs))
    Xs.sort()
    curr_data = np.zeros((len(all_Zs),len(all_Xs)))
    
    powers = list(map(get_pm_data,data))
    curr_power_data = np.zeros((len(all_Zs),len(all_Xs)))
    
#    all_specs_for_alpha_path = os.path.join(main_path,'all_specs_for_alpha.npy')
#    if os.path.isfile(all_specs_for_alpha_path):
#        all_specs_for_alpha = list(np.load(all_specs_for_alpha_path))
#    else:
#        all_specs_path_for_alpha = [os.path.join(main_path,r'SPECS_a%i_x%i_z%i.npy'%(alpha*100,X*100,Z*100)) for Z in Zs for X in Xs]
#        all_specs_for_alpha = []
#        for p in all_specs_path_for_alpha:
#            if not os.path.isfile(p):
#                break
#            all_specs_for_alpha.append(np.load(p))
#        np.save(all_specs_for_alpha_path,np.array(all_specs_for_alpha))
    for j,datum in enumerate(data):
        curr_x = get_x(datum)
        i_x = get_nearest_idx_from_list(curr_x,Xs)
        curr_z = get_z(datum)
        i_z = get_nearest_idx_from_list(curr_z,Zs)
        specs = get_spec(datum) - BCKGND_SPEC
            
        try:
            specs780 = specs[min_idx:max_idx]
            specs780_bg = specs[min_idx_bg:max_idx_bg]
#                sums780 = np.sum(specs780,axis=1) - np.sum(specs780_bg,axis=1)
            sums780 = np.sum(specs780) - np.median(specs780_bg)*np.abs(max_idx-min_idx)
            mean780 = np.average(sums780)
            curr_data[i_z][i_x]=mean780
            curr_power_data[i_z][i_x]=powers[j]*1e3#uW
        except IndexError:
            pass
    
    curr_data[curr_data == 0] = np.median(curr_data[curr_data > 0])
    curr_power_data[curr_power_data == 0] = np.median(curr_power_data[curr_power_data > 0])
    
    curr_data = curr_data/(curr_power_data/100.)**normalize_SHG_by_pm_with_power
    SHG_data.append(curr_data)
    pm_data.append(curr_power_data)
#    np.save(all_specs_for_alpha_path,np.array(all_specs_for_alpha))

    def format_coord(x, y):
        x += 1
        y += 1
        x = round(x*resol/resol)*resol-resol + all_Xs[0]
        y = round(y*resol/resol)*resol-resol + all_Zs[0]
        return 'x=%1.1f, z=%1.1f'%(x, y)
    xlb=list(np.arange(all_Xs[0],all_Xs[-1]+0.01,10,dtype=int))
    ylb=list(np.arange(all_Zs[0],all_Zs[-1]+0.01,10,dtype=int))

    _fig = plt.figure(r'%s: sum %.2f to %.2f nm'%(sample,min_wl,max_wl))
    plt.clf()
    ax = _fig.add_subplot(121)
    cax = ax.imshow(curr_data, cmap='Reds_r',interpolation='none')
    _fig.colorbar(cax)
    ax.set_xticks(np.arange(0,len(all_Xs),10/resol))
    ax.set_yticks(np.arange(0,len(all_Zs),10/resol))
    ax.set_xticklabels(xlb)
    ax.set_yticklabels(ylb)
    ax.set_xlabel(r'$\mu$m')
    ax.set_ylabel(r'$\mu$m')
    ax.set_title('SHG, max/min = %.2f/%.2f'%(np.nanmax(curr_data),np.nanmin(curr_data)))
    ax.format_coord = format_coord
    plt.pause(1e-6)
    
    ax2 = _fig.add_subplot(122)
    cax2 = ax2.imshow(curr_power_data, cmap='Greys_r',interpolation='none')
    _fig.colorbar(cax2)
    ax2.set_xticks(np.arange(0,len(all_Xs),10/resol))
    ax2.set_yticks(np.arange(0,len(all_Zs),10/resol))
    ax2.set_xticklabels(xlb)
    ax2.set_yticklabels(ylb)
    ax2.set_xlabel(r'$\mu$m')
    ax2.set_ylabel(r'$\mu$m')
    ax2.set_title('Pump, max/min = %.2f/%.2f'%(np.nanmax(curr_power_data),np.nanmin(curr_power_data)))
    ax2.format_coord = format_coord
    plt.pause(1e-6)
    
    _fig.suptitle(r'%s:sum %.2f to %.2f nm'%(sample,min_wl,max_wl))
    _fig.tight_layout(rect=(0,0,1,0.95))
    plt.pause(1e-6)
    print('Done.')
#%%
def VP2_pol_SHG_rotA_anal(sample,pm_as_ref=True,verbose=False):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\VP2_pol_SHG_rotA',sample)
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    spec_files = list(filter(lambda name: 'SPECS' in name,npy_files))
    BCKGND_SPEC = np.load(os.path.join(main_path,'BCKGND_SPEC.npy'))
    global alphas,betas,data780,datapm,datasc,max_alpha,d
    
    def get_a_from_name(name):
        try:
            return float(name.split('a')[1].split('.npy')[0])/100
        except:
            return float(name.split('a')[1].split('_')[0])/100
    def get_b_from_name(name):
        try:
            return float(name.split('b')[1].split('.npy')[0])/100
        except:
            return float(name.split('b')[1].split('_')[0])/100
    
    alphas = list(map(lambda name: get_a_from_name(name),spec_files))
    alphas = list(set(alphas))
    alphas.sort()
    alphas = np.array(alphas)
           
    betas = list(map(lambda name: get_b_from_name(name),spec_files))
    betas = list(set(betas))
    betas.sort()
    betas = np.array(betas)
    
    for alpha in alphas:
        print alpha
        mean780s = []
        max_alpha=[]
        spec_files_of_alpha = list(filter(lambda name: alpha == get_a_from_name(name),spec_files))
        powers = np.load(os.path.join(main_path,'powers_a%i.npy'%(alpha*100)))*1e6 #uW
        for beta in betas: 
            curr_spec_file = list(filter(lambda name: beta == get_b_from_name(name),spec_files_of_alpha))
            curr_spec_file = curr_spec_file[0]
            specs = np.load(os.path.join(main_path,curr_spec_file))# - BCKGND_SPEC
            specs780 = specs[:,550:625]
            specs780_bg = specs[:,625:700]
    #        sums780 = np.sum(specs780,axis=1) - np.sum(specs780_bg,axis=1)
            sums780 = np.sum(specs780,axis=1) - np.median(specs780_bg,axis=1)*(700-625)
            mean780 = np.average(sums780)
    
            mean780s.append(mean780)
        
        mean780s = np.array(mean780s)
        if pm_as_ref:
            mean780s = mean780s/np.square(powers)
             
        _fig = plt.figure(r'%s @ a = %ideg'%(sample,alpha))
        _fig.clf()
        fig = _fig.add_subplot(121,projection='polar')
        fig.plot(betas[:len(mean780s)]/180.*np.pi,mean780s,'o',label='SHG')
        fig.set_title(r'SHG @ 780 nm')
        fig.set_xticks(np.arange(0,360,60)/180.*np.pi)
        
        fig2 = _fig.add_subplot(122,projection='polar')
        fig2.plot(betas[:len(powers)]/180.*np.pi,powers,'o',label='pump, uW')
        if pm_as_ref:
            fig2.set_title('Pump as ref @ 1560 nm')
        else:
            fig2.set_title('Pump @ 1560 nm')
        fig2.set_xticks(np.arange(0,360,60)/180.*np.pi)
        
        if pm_as_ref:
            try:
                sc_powers = np.load(os.path.join(main_path,'powers_sc.npy'))*1e6 #uW
                datasc = np.array(sc_powers)
            except:
                pass
       
        _fig.suptitle(r'%s @ a = %ideg'%(sample,alpha))
        
        data780 = np.array(mean780s)
        datapm = np.array(powers)
        betas = np.array(betas)
        
        try:
            fit_results_SHG = multiple_petals_n7_fitter(betas,mean780s)
            fig.plot(betas[:len(mean780s)]/180.*np.pi,multiple_petals_n7(betas,*fit_results_SHG[0]),label='fit n7 SHG')
            SHG_max_alpha = fit_results_SHG[0][9]
            SHG_max_alpha_e = fit_results_SHG[1][9]
            t = 'A0 + A2 = %.2f $\pm$ %.2f cts\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg'%(fit_results_SHG[0][0]+fit_results_SHG[0][2],np.sqrt(fit_results_SHG[1][0]**2+fit_results_SHG[1][2]**2),SHG_max_alpha,SHG_max_alpha_e)
            fig.set_title('SHG @ 780 nm\n%s'%(t))
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01),ncol=2)
            _fig.tight_layout(rect=(0,0,1,0.95))
            
            fit_results_pump = multiple_petals_n7_fitter(betas,powers)
            pump_max_alpha = fit_results_pump[0][9]
            pump_max_alpha_e = fit_results_pump[1][9]
            fig2.plot(betas[:len(mean780s)]/180.*np.pi,multiple_petals_n7(betas,*fit_results_pump[0]),label='fit n7 pump, uW')
            t = 'A0 + A2 = %.2f $\pm$ %.2f uW\n$\\alpha_{max}$ = %.2f $\pm$ %.2f deg'%(fit_results_pump[0][0]+fit_results_pump[0][2],np.sqrt(fit_results_pump[1][0]**2+fit_results_pump[1][2]**2),pump_max_alpha,pump_max_alpha_e)
            fig2.set_title('Pump @ 1560 nm\n%s'%(t))
            fig2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01))
            plt.pause(0.1)
            
            _fig2 = plt.figure('Fitting summary of %s at a = %ideg'%(sample,alpha))
            _fig2.clf()
            _fig2.suptitle('Fitting summary of %s at a = %ideg'%(sample,alpha))
            fig3 = _fig2.add_subplot(121)
            fig3.bar(range(8),fit_results_SHG[0][:8],yerr=fit_results_SHG[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
            fig3.axhline(0,color='k')
            fig3.set_ylabel('Amplitude, au',color='C0')
            fig3.set_title('SHG @ 780 nm')
            fig3.set_xlabel('Harmonics')
            fig3_2 = fig3.twinx()
            fig3_2.errorbar(range(1,8),fit_results_SHG[0][8:],ls='None',marker='o',yerr=fit_results_SHG[1][8:],capsize=5,C='C1',ecolor='k')
            fig3_2.grid()
            fig3_2.set_ylabel('Angle, deg',color='C1')
            
            fig4 = _fig2.add_subplot(122)
            fig4.bar(range(8),fit_results_pump[0][:8],yerr=fit_results_pump[1][:8],tick_label=list(range(8)),capsize=5,color='C0')
            fig4.axhline(0,color='k')
            fig4.set_ylabel('Amplitude, uW',color='C0')
            fig4.set_title('Pump @ 1560 nm')
            fig4.set_xlabel('Harmonics')
            fig4_2 = fig4.twinx()
            fig4_2.errorbar(range(1,8),fit_results_pump[0][8:],ls='None',marker='o',yerr=fit_results_pump[1][8:],capsize=5,C='C1',ecolor='k')
            fig4_2.grid()
            fig4_2.set_ylabel('Angle, deg',color='C1')
            _fig2.tight_layout(rect=(0,0,1,0.95))
            
    
            
#            return SHG_max_alpha, SHG_max_alpha_e, pump_max_alpha, pump_max_alpha_e, fit_results_SHG, fit_results_pump
        
            d = SHG_max_alpha  
            max_alpha.append(d)
        
        except Exception as e:
            if verbose:
                print(e)
                print('Failed to fit %s'%sample)

#%%
def anal_scan_delayline(npy_file,min_wl=500,max_wl=515):
    global poss, spec_sums, lockins
    data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
    poss = list(map(lambda x: x[0][0],data))
    lockins = np.array(map(lambda x: x[0][1],data))*1e3
    specs = list(map(lambda x: x[1],data))
    min_idx = get_nearest_idx_from_list(min_wl,hero_pixel_wavelengths)
    max_idx = get_nearest_idx_from_list(max_wl,hero_pixel_wavelengths)
    spec_sums = list(map(lambda x: np.sum(x[min_idx:max_idx]),specs))
    min_y = np.min(specs[0][min_idx-5:max_idx+5])
    max_y = np.max(specs[0][min_idx-5:max_idx+5])
    _figall = plt.figure(npy_file+' all specs')
    figall = _figall.add_subplot(111)
    for spec in specs:
        figall.plot(hero_pixel_wavelengths,spec)
    figall.set_xlim(min_wl-5,max_wl+5)
    figall.set_ylim(min_y-10,max_y)
    figall.set_xlabel('Wavelength, nm')
    _fig = plt.figure(npy_file)
    fig = _fig.add_subplot(121)
    fig2 = _fig.add_subplot(122)
    line, = fig.plot(poss,spec_sums,'o')
    line2, = fig2.plot(poss,lockins,'o')
    _fig.suptitle(npy_file)
    fig.set_title('spec sum of %i to %i nm'%(min_wl,max_wl))
    fig2.set_title('Lock-in amplitude')
    fig.set_xlabel('fdl pos, mm')
    fig.set_ylabel('spec sum, au')
    fig2.set_xlabel('fdl pos, mm')
    fig2.set_ylabel('lock-in amp, mV')
    poss = np.array(poss)
    lockins = np.array(lockins)
    spec_sums = np.array(spec_sums)
    return (_fig,fig,line,figall,fig2,line2)

#%%

def anal_scan_delayline_with_andor(npy_file,min_wl=500,max_wl=515,fit=False,fit_poly=False,manual_bg=0):
    global poss, spec_sums, lockins,specs
    data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
#    wls = data[1][5:-5]
    wls = data[1]
    data = data[2:]
    poss = list(map(lambda x: x[0],data))
#    specs = list(map(lambda x: np.convolve(x[1],np.ones(10)/10.,'same')[5:-5],data))
    specs = np.array(list(map(lambda x: x[1],data)))
    
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)
    
    # remove cosmic rays
#    for spec in specs:
#        spec = remove_outlier3(wls,spec,th=0.01)
    if not fit and not fit_poly:
        try:
            spec_sums = np.array(list(map(lambda spec:           np.sum(list(map(lambda x: x[min_idx:max_idx] - np.median(np.append(x[:min_idx],x[max_idx:])),spec)))        ,specs)))
        except:
            spec_sums = np.array(list(map(lambda spec:           np.sum(spec[min_idx:max_idx] - np.median(np.append(spec[:min_idx],spec[max_idx:])))        ,specs)))
    else:
        try:
            crop_specs = np.array(list(map(lambda x: np.median(x,0)[min_idx:max_idx],specs)))
        except:
            crop_specs = np.array(list(map(lambda x: x[min_idx:max_idx],specs)))
        if fit:
            fitted_data = np.array(list(map(lambda x: gaussian_fitter(wls[min_idx:max_idx],x,plot=False),crop_specs)))
            spec_sums = np.array(list(map(lambda x: x[0][2],fitted_data)))
        else:
            fitted_data = np.array(list(map(lambda x: np.poly1d(poly_fitter(wls[min_idx:max_idx],x,plot=False))(wls[min_idx:max_idx]),crop_specs)))
            spec_sums = np.sum(fitted_data-manual_bg,1)
#    min_y = np.min(specs[min_idx-5:max_idx+5])
#    max_y = np.max(specs[min_idx-5:max_idx+5])
    _figall = plt.figure(npy_file+' all specs')
    figall = _figall.add_subplot(111)
    for j,spec in enumerate(specs):
        if len(specs.shape) > 2:
            figall.plot(wls,np.median(spec,0),c=plt.cm.RdYlGn(255*j/(len(specs)-1)))
        else:
            figall.plot(wls,spec,c=plt.cm.RdYlGn(255*j/(len(specs)-1)))
#    figall.set_xlim(min_wl-5,max_wl+5)
#    figall.set_ylim(min_y-10,max_y)
    figall.set_xlabel('Wavelength, nm')
    _fig = plt.figure(npy_file)
    fig = _fig.add_subplot(111)
    line, = fig.plot(poss,spec_sums,'o')
    _fig.suptitle(npy_file)
    fig.set_title('spec sum of %i to %i nm'%(min_wl,max_wl))
    fig.set_xlabel('fdl pos, mm')
    fig.set_ylabel('spec sum, au')
    poss = np.array(poss)
    spec_sums = np.array(spec_sums)
    return (_fig,fig,line,figall)

#%%

def real_time_anal_scan_delayline_with_andor(npy_file,min_wl=500,max_wl=510,refresh_delay=1):
    data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
#    wls = data[1][5:-5]
    wls = data[1]
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)
    _fig, fig, line, figall= anal_scan_delayline_with_andor(npy_file,min_wl=min_wl,max_wl=max_wl)
    prev_data_len = 0
    num = 0
    global spec_sums,poss
    while True:
        try:
            data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
        except:
            plt.pause(0.1)
            data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
        data = data[2:]
        curr_data_len = len(data)
        if curr_data_len == prev_data_len:
            if num > 10:
                return
            plt.pause(refresh_delay)
            num += 1
            continue
        else:
            num = 0
        poss = list(map(lambda x: x[0],data))
#        specs = list(map(lambda x: np.convolve(x[1],np.ones(10)/10.,'same')[5:-5],data))
        specs = np.array(list(map(lambda x: x[1],data)))
        try:
            spec_sums = np.array(list(map(lambda spec:           np.sum(list(map(lambda x: x[min_idx:max_idx] - np.median(np.append(x[:min_idx],x[max_idx:])),spec)))        ,specs)))
        except:
            spec_sums = np.array(list(map(lambda spec:           np.sum(spec[min_idx:max_idx] - np.median(np.append(spec[:min_idx],spec[max_idx:])))        ,specs)))
#        spec_sums = np.array(list(map(lambda spec:           np.sum(spec[min_idx:max_idx] - np.median(np.append(spec[:min_idx],spec[max_idx:])))        ,specs)))
        
        line.set_xdata(poss)
        line.set_ydata(spec_sums)
        fig.relim(True)
        fig.autoscale_view(True,True,True)
        
        while curr_data_len > len(figall.lines):
            if len(specs.shape) > 2:
                figall.plot(wls,np.mean(specs[len(figall.lines)],0))
            else:
                figall.plot(wls,specs[len(figall.lines)])
        
        plt.draw_all()
        prev_data_len = curr_data_len
        plt.pause(refresh_delay)


#%%

def anal_scan_delayline_with_andor3(npy_file,min_wl=760,max_wl=790,fit=False,sc_on=True,normalize=True):
    global poss, spec_sums,specs,sc_powers,laser_powers, raw_spec_sums
    data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor3',npy_file+'.npy'))
    wls = data[1]
    data = data[2:]
    poss = list(map(lambda x: x[0],data))
    specs = np.array(list(map(lambda x: x[1],data)))
    sc_powers = np.array(list(map(lambda x: x[2],data)))
    laser_powers = np.array(list(map(lambda x: x[3],data)))
    
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)
    
    # remove cosmic rays
#    for spec in specs:
#        spec = remove_outlier3(wls,spec,th=0.01)
    if not fit:
        try:
            spec_sums = np.array(list(map(lambda spec:           np.sum(list(map(lambda x: remove_outlier3(range(len(x[min_idx:max_idx])),x[min_idx:max_idx],0.1)[1] - np.median(np.append(x[:min_idx],x[max_idx:])),spec)))        ,specs)))
        except:
            spec_sums = np.array(list(map(lambda spec:           np.sum(remove_outlier3(range(len(spec[min_idx:max_idx])),spec[min_idx:max_idx],0.1)[1] - np.median(np.append(spec[:min_idx],spec[max_idx:])))        ,specs)))
    else:
        try:
            crop_specs = np.array(list(map(lambda x: np.median(x,0)[min_idx:max_idx],specs)))
        except:
            crop_specs = np.array(list(map(lambda x: x[min_idx:max_idx],specs)))
        fitted_data = np.array(list(map(lambda x: gaussian_fitter(wls[min_idx:max_idx],x,plot=False),crop_specs)))
        spec_sums = np.array(list(map(lambda x: x[0][2],fitted_data)))
    raw_spec_sums = spec_sums
    if normalize:
        if sc_on:
            spec_sums = spec_sums/sc_powers/np.power(laser_powers,2)
        else:
            spec_sums = spec_sums/np.power(laser_powers,2)
    if len(specs.shape) > 2:
        min_y = np.min(specs[0][0][min_idx-5:max_idx+5])
        max_y = np.max(specs[0][0][min_idx-5:max_idx+5])
    else:
        min_y = np.min(specs[0][min_idx-5:max_idx+5])
        max_y = np.max(specs[0][min_idx-5:max_idx+5])
    _figall = plt.figure(npy_file+' all specs')
    figall = _figall.add_subplot(111)
    for j,spec in enumerate(specs):
        if len(specs.shape) > 2:
            figall.plot(wls,np.median(spec,0),c=plt.cm.RdYlGn(255*j/(len(specs)-1)))
        else:
            figall.plot(wls,spec,c=plt.cm.RdYlGn(255*j/(len(specs)-1)))
    figall.set_xlim(min_wl-5,max_wl+5)
    figall.set_ylim(min_y-10,max_y)
    figall.set_xlabel('Wavelength, nm')
    _fig = plt.figure(npy_file)
    fig = _fig.add_subplot(211)
    line, = fig.plot(poss,spec_sums,'o')
    _fig.suptitle(npy_file)
    fig.set_title('normed spec sum of %i to %i nm'%(min_wl,max_wl))
    if sc_on:
        fig.set_ylabel('spec sum/sc power/laser power$^2$, au')
    else:
        fig.set_ylabel('spec sum/laser power$^2$, au')
    figscpower = _fig.add_subplot(212)
    figlaserpower = figscpower.twinx()
    linesc, = figscpower.plot(poss,sc_powers,'o',color='C1')
    linelaser, = figlaserpower.plot(poss,laser_powers,'o',color='C2')
    figscpower.set_title('sc and laser powers monitor')
    figscpower.set_xlabel('fdl pos, mm')
    figscpower.set_ylabel('sc power, au',color='C1')
    figlaserpower.set_ylabel('laser power, au',color='C2')
    _fig.tight_layout()
    poss = np.array(poss)
    spec_sums = np.array(spec_sums)
    return (_fig,fig,line,figall,figscpower,figlaserpower,linesc,linelaser)

#%%

def anal_scan_delayline_with_andor4(npy_file,min_wl=765,max_wl=795,fit=False,normalize=True,sc_wl=None,minus_bg=True,minus_iSHG=False,show_all_raw=False):
    global poss, spec_sums,specs,sc_powers,laser_powers, lonly_specs, sconly_specs,processed_specs, _figall, _fig, sconly_sc_powers
    data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor4',npy_file+'.npy'))
    try:
        sc_wl = data[0][0]
    except:
        pass
    wls = data[1]
    both_beams = data[2]
    laser_only = data[3]
    sc_only = data[4]
    
    # data that will be analysed
    poss = list(map(lambda x: x[0],both_beams))
    specs = np.array(list(map(lambda x: x[1],both_beams)))
    sc_powers = np.array(list(map(lambda x: x[2],both_beams)))*1e9
    if sc_wl != None:
        sc_powers *= NPBS_TR_c_fx(sc_wl)
    laser_powers = np.array(list(map(lambda x: x[3],both_beams)))
    
    #other data
    lonly_poss = list(map(lambda x: x[0],laser_only))
    lonly_specs = np.array(list(map(lambda x: x[1],laser_only)))
    lonly_laser_powers = np.array(list(map(lambda x: x[3],laser_only)))
    
    sconly_poss = list(map(lambda x: x[0], sc_only))
    sconly_specs = np.array(list(map(lambda x: x[1],sc_only)))
    sconly_sc_powers = np.array(list(map(lambda x: x[2],sc_only)))*1e9
    if sc_wl != None:
        sconly_sc_powers *= NPBS_TR_c_fx(sc_wl)
    
    # Don't call with normalize True if there is not enough data (missing powermeter)
    
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)
    
    spec_sums = []
    processed_specs = []
    raw_both_specs, raw_sc_specs, raw_laser_specs = [], [], []
    raw_both_sums, raw_sc_sums, raw_laser_sums = [], [], []
    for i in range(len(specs)):
        curr_spec = np.median(specs[i],0)
#        curr_spec = remove_outlier3(range(len(curr_spec)),curr_spec,0.01)[1]
        if show_all_raw:
            curr_spec = np.median(specs[i],0)
            raw_both_specs.append(copy.copy(curr_spec))
            raw_both_sums.append(np.sum(curr_spec[min_idx:max_idx]))
            
            curr_sconly_spec = np.median(sconly_specs[i],0)
            raw_sc_specs.append(curr_sconly_spec)
            raw_sc_sums.append(np.sum(curr_sconly_spec[min_idx:max_idx]))
            
            curr_lonly_spec = np.median(lonly_specs[i],0)
            raw_laser_specs.append(curr_lonly_spec)
            raw_laser_sums.append(np.sum(curr_lonly_spec[min_idx:max_idx]))
        if fit:
            curr_popt = poly_fitter(wls[min_idx:max_idx],curr_spec[min_idx:max_idx],plot=False)
            curr_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
        else:
            curr_spec = curr_spec[min_idx:max_idx]
        if normalize:
            curr_spec /= sconly_sc_powers[i]
#                    curr_spec /= np.power(laser_powers[i],2)
        if minus_bg:
            curr_lonly_spec = np.median(lonly_specs[i],0)
#            curr_lonly_spec = remove_outlier3(range(len(curr_lonly_spec)),curr_lonly_spec,0.01)[1]
            if fit:
                curr_popt = poly_fitter(wls[min_idx:max_idx],curr_lonly_spec[min_idx:max_idx],plot=False)
                curr_lonly_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
            else:
                curr_lonly_spec = curr_lonly_spec[min_idx:max_idx]
            if normalize:
    #                    curr_lonly_spec /= np.power(laser_powers[i],2)
                pass
            curr_sconly_spec = np.median(sconly_specs[i],0)
#            curr_sconly_spec = remove_outlier3(range(len(curr_sconly_spec)),curr_sconly_spec,0.01)[1]
            if fit:
                curr_popt = poly_fitter(wls[min_idx:max_idx],curr_sconly_spec[min_idx:max_idx],plot=False)
                curr_sconly_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
            else:
                curr_sconly_spec = curr_sconly_spec[min_idx:max_idx]
            if normalize:
                curr_sconly_spec /= sconly_sc_powers[i]
            curr_signal_spec = curr_spec - curr_sconly_spec
        else:
            curr_signal_spec = curr_spec
        if minus_iSHG:
            curr_lonly_spec = (curr_lonly_spec - 544.5) #542.3 for 33kHz, 335.1 for 50kHz, 146.1 for 100 kHz
            if normalize:
                curr_lonly_spec /= sconly_sc_powers[i]
            curr_signal_spec = curr_signal_spec - curr_lonly_spec
        curr_spec_sum = np.sum(curr_signal_spec)
        spec_sums.append(curr_spec_sum)
        processed_specs.append(curr_signal_spec)
       
#    try:
#        min_y = np.min(processed_specs[min_idx-5:max_idx+5])
#        max_y = np.max(processed_specs[min_idx-5:max_idx+5])
#    except:
#        min_y = np.min(processed_specs[min_idx:max_idx])
#        max_y = np.max(processed_specs[min_idx:max_idx])
    _figall = plt.figure(npy_file)
    figall = _figall.add_subplot(121)
    
    for j,spec in enumerate(processed_specs):
        if fit:
            figall.plot(wls[min_idx:max_idx],spec,c=plt.cm.RdYlGn(255*j/(len(processed_specs)-1)))
        else:
            figall.plot(wls[min_idx:max_idx],spec,c=plt.cm.RdYlGn(255*j/(len(processed_specs)-1)))
#    figall.set_xlim(min_wl-5,max_wl+5)
#    figall.set_ylim(min_y-10,max_y)
    figall.set_xlabel('Wavelength, nm')
    figall.set_title(npy_file)
    fig = _figall.add_subplot(222)
    line, = fig.plot(poss,spec_sums,'o')
    _fig.suptitle(npy_file)
    fig.set_title('normed spec sum of %i to %i nm'%(min_wl,max_wl))
    fig.grid()
#    if sc_on:
#        fig.set_ylabel('spec sum/sc power/laser power$^2$, au')
#    else:
#        fig.set_ylabel('spec sum/laser power$^2$, au') #not true if not normalized?
    fig.set_ylabel('spec sum, au')
    
    figscpower = _figall.add_subplot(224)
    figlaserpower = figscpower.twinx()
    linesc, = figscpower.plot(poss,sc_powers,'o',color='C1')
    linelaser, = figlaserpower.plot(poss,laser_powers,'o',color='C2')
    figscpower.set_title('sc and laser powers monitor')
    figscpower.set_xlabel('fdl pos, mm')
    figscpower.set_ylabel('sc power, nW',color='C1')
    figscpower.grid()
    figlaserpower.set_ylabel('laser power, au',color='C2')
    plt.get_current_fig_manager().window.showMaximized()
#    _fig.tight_layout()
    poss = np.array(poss)
    spec_sums = np.array(spec_sums)
    
    if show_all_raw:
        _figall_raw = plt.figure(npy_file+' raw')
        figall_both = _figall_raw.add_subplot(321)
        figall_sc = _figall_raw.add_subplot(323)
        figall_laser = _figall_raw.add_subplot(325)
        
        for j,spec in enumerate(raw_both_specs):
            figall_both.plot(wls,raw_both_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            figall_sc.plot(wls,raw_sc_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            figall_laser.plot(wls,raw_laser_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            
        figall_both.set_ylabel('both beams on')
        figall_sc.set_ylabel('only sc')
        figall_laser.set_ylabel('only 1550')
        figall_laser.set_xlabel('Wavelength, nm')
        figall_both.set_title(npy_file+' raw')
        fig_both = _figall_raw.add_subplot(322)
        fig_sc = _figall_raw.add_subplot(324)
        fig_laser = _figall_raw.add_subplot(326)
        fig_both.plot(poss,raw_both_sums,'o')
        fig_sc.plot(poss,raw_sc_sums,'o')
        fig_laser.plot(poss,raw_laser_sums,'o')
        fig_both.set_title('raw spec sum of %i to %i nm'%(min_wl,max_wl))
        fig_both.grid()
        fig_sc.grid()
        fig_laser.grid()
        fig_sc.set_ylabel('raw spec sum, au')
        
        fig_laser.set_xlabel('fdl pos, mm')
        plt.get_current_fig_manager().window.showMaximized()

#%%

def anal_images_scan_delayline_with_andor5(npy_file,over_write=False,radius=150,plot=False,um_per_px=2./17):
    main_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor5', npy_file) 
    img_path = os.path.join(main_path,'images')
    all_npy_files = list(filter(lambda x: x.endswith('.npy'),os.listdir(img_path)))
    all_raw_imgs = list(filter(lambda x: not x.endswith('_noBG.npy'),all_npy_files))
    all_processed_imgs = list(filter(lambda x: x.endswith('_noBG.npy'),all_npy_files))
    def img_name_to_tag(name):
        return tuple(map(int,name.split('.')[0].split('_')[:2]))
    def tag_to_img_name(tag):
        return '%i_%i.npy'%tag
    def process_img(path):
        image = np.load(path)
        subtract_background_rolling_ball(image, radius, light_background=False, use_paraboloid=True, do_presmooth=True)
        np.save(path[:-4]+'_noBG.npy', image)
    all_raw_img_tags = list(map(img_name_to_tag,all_raw_imgs))
    all_processed_img_tags = list(map(img_name_to_tag,all_processed_imgs))
    all_unprocessed_img_tags = list(set(all_raw_img_tags)-set(all_processed_img_tags))
    
    if over_write:
        all_unprocessed_img_tags = all_raw_img_tags
    
    total_len = len(all_unprocessed_img_tags)+1
    if len(all_unprocessed_img_tags):
        start_time = time.time()
        
        process_img(os.path.join(main_path,'img_LED_only.npy'))
        elapsed_time = time.time() - start_time
        time_left = elapsed_time*(1.*total_len-1)
        completed = u'Processed %s (%.2f percent) %s left.'%('img_LED_only.npy',100.0*(float(1)/total_len),sec_to_hhmmss(time_left))
        prints(completed)
        prev = completed
        for j,tag in enumerate(all_unprocessed_img_tags):
            filename = tag_to_img_name(tag)
            path = os.path.join(img_path, filename)
            process_img(path)
            
            elapsed_time = time.time() - start_time
            time_left = elapsed_time*(1.*total_len/(j+1+1)-1)
            completed = u'Processed %s (%.2f percent) %s left.'%(filename,100.0*(float(j+1+1)/total_len),sec_to_hhmmss(time_left))
            prints(completed,prev)
            prev = completed
    else:
        print('All images had been processed.')
    
    print('Extracting coordinates of laser spot on sample...')
    img_info = np.load(os.path.join(main_path,'img_info.npy'))
    img_info_temp = list(map(lambda x: [x[0],int(x[1]),float(x[2]),float(x[3])],img_info.astype(np.object)))
    sample_ref_img = np.load(os.path.join(main_path,'img_LED_only_noBG.npy'))
    laser_ref_img = np.load(os.path.join(main_path,'img_laser_only.npy'))
    for i,img_info_ele in enumerate(img_info):
        img_name = img_info_ele[0]+'_noBG.npy'
        curr_data_img = np.load(os.path.join(img_path,img_name))
        curr_x, curr_y = extract_coords_of_laser_on_sample(curr_data_img,sample_ref_img,laser_ref_img,plot=plot,um_per_px=um_per_px)
        img_info_temp[i].append(curr_x)
        img_info_temp[i].append(curr_y)
    # img info will be [[img npy name, rep, fdl pos, time since start, laser x pos on sample, laser y pos on sample]]
    np.save(os.path.join(main_path,'img_info.npy'),np.array(img_info_temp))
    dl_poss = np.array(list(map(lambda x: x[2],img_info_temp)))
    laser_xs = np.array(list(map(lambda x: x[4],img_info_temp)))
    laser_ys = np.array(list(map(lambda x: x[5],img_info_temp)))
    
    fig = plt.figure(npy_file+' laser spot')
    plt.get_current_fig_manager().window.showMaximized()
    fig.suptitle(npy_file+' laser spot')
    figx = fig.add_subplot(211)
    figx.plot(dl_poss,laser_xs-laser_xs[0],'o')
    figx.set_ylabel('$\Delta$x, um')
    figx.axhline(0,c='k')
    figx.grid()
    figy = fig.add_subplot(212)
    figy.plot(dl_poss,laser_ys-laser_ys[0],'o')
    figy.set_ylabel('$\Delta$y, um')
    figy.set_xlabel('fdl pos, mm')
    figy.axhline(0,c='k')
    figy.grid()
    return dl_poss,laser_xs,laser_ys
    
        
def view_images_scan_delayline_with_andor5(npy_file, noBG=True):
    main_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor5', npy_file) 
    img_path = os.path.join(main_path,'images')
    keyword = ''
    if noBG:
        keyword = 'noBG'
    imshow_npy_files_with_word_in(img_path, keyword)

def extract_coords_of_laser_on_sample(data_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=2./17):# from photoshop = 5.7/155
    data_img_crop = data_img[100:-100,100:-100]
    img_LED_only = sample_ref_img
    min_val,max_val,min_loc,(x,y) = cv2.minMaxLoc(sp.ndimage.median_filter(laser_ref_img,3))
    img_laser_only_crop = laser_ref_img[y-15:y+15,x-15:x+15]
    
    conv_data_sample = sp.ndimage.median_filter(cv2.matchTemplate(data_img_crop,img_LED_only,5),2)
#    conv_data_sample = cv2.matchTemplate(data_img_crop,img_LED_only,5)
    min_val,max_val_sample,min_loc,max_loc_sample = cv2.minMaxLoc(conv_data_sample)
    x_shift,y_shift = np.array(max_loc_sample) - 100
    
    data_img_crop_cleaned = equalize_histogram_and_8bit(data_img_crop.astype(float) - sample_ref_img[100+y_shift:-100+y_shift,100+x_shift:-100+x_shift].astype(float))
    conv_data_laser = sp.ndimage.median_filter(cv2.matchTemplate(data_img_crop_cleaned,img_laser_only_crop,5),2)
#    conv_data_laser = cv2.matchTemplate(data_img_crop,img_laser_only_crop,5)
    min_val,max_val_laser,min_loc,max_loc_laser = cv2.minMaxLoc(conv_data_laser)
    
    laser_spot_from_sample = ( (np.array(max_loc_laser) - 160) - (100 - np.array(max_loc_sample)) )*um_per_px
    
    if plot:
        fig=plt.figure()
        fig_data_img=fig.add_subplot(121)
        fig_data_img.imshow(data_img_crop)
        fig_data_img.set_title('data img\nlaser spot on sample @ %s um'%(str(tuple(laser_spot_from_sample))))
        
        fig_LED_only=fig.add_subplot(243)
        fig_LED_only.imshow(img_LED_only)
        fig_LED_only.set_title('sample ref')
        fig_laser_only=fig.add_subplot(244)
        fig_laser_only.imshow(img_laser_only_crop)
        fig_laser_only.set_title('laser ref')
        
        fig_conv_LED = fig.add_subplot(247)
        fig_conv_LED.imshow(conv_data_sample)
        fig_conv_LED.set_title('Conv. with sample ref\n%.3f @ %s'%(max_val_sample,str(max_loc_sample)))
        
        fig_conv_laser = fig.add_subplot(248)
        fig_conv_laser.imshow(conv_data_laser)
        fig_conv_laser.set_title('Conv. with laser ref\n%.3f @ %s'%(max_val_laser,str(max_loc_laser)))
    
    return laser_spot_from_sample

def extract_coords_of_laser_on_sample_2(sample_img,laser_img,sample_ref_img,laser_ref_img,plot=False,um_per_px=2./17):# from photoshop = 5.7/155
    sample_img_crop = sample_img[100:-100,100:-100]
    laser_img_crop = laser_img[100:-100,100:-100]
    img_LED_only = sample_ref_img
    min_val,max_val,min_loc,(x,y) = cv2.minMaxLoc(sp.ndimage.median_filter(laser_ref_img,3))
    img_laser_only_crop = laser_ref_img[y-15:y+15,x-15:x+15]
    
    conv_data_sample = sp.ndimage.median_filter(cv2.matchTemplate(sample_img_crop,img_LED_only,5),2)
    min_val,max_val_sample,min_loc,max_loc_sample = cv2.minMaxLoc(conv_data_sample)
    x_shift,y_shift = np.array(max_loc_sample) - 100
    
    conv_data_laser = sp.ndimage.median_filter(cv2.matchTemplate(laser_img_crop,img_laser_only_crop,5),2)
    min_val,max_val_laser,min_loc,max_loc_laser = cv2.minMaxLoc(conv_data_laser)
    
    laser_spot_from_sample = ( (np.array(max_loc_laser) - 160) - (100 - np.array(max_loc_sample)) )*um_per_px
    
    if plot:
        fig=plt.figure()
        fig_data_img=fig.add_subplot(231)
        fig_data_img.imshow(sample_img_crop)
        plt.axis('off')
        fig_data_img.set_title('data img\nlaser spot on sample @ %s um'%(str(tuple(laser_spot_from_sample))))
        plt.figtext(0.365,0.7,r'$\ast$',va='center',ha='left')
        
        fig_LED_only=fig.add_subplot(232)
        fig_LED_only.imshow(img_LED_only)
        plt.axis('off')
        fig_LED_only.set_title('sample ref')
        plt.figtext(0.64,0.7,'=',va='center',ha='left')
        
        fig_conv_LED = fig.add_subplot(233)
        fig_conv_LED.imshow(conv_data_sample)
        plt.axis('off')
        fig_conv_LED.set_title('Conv. with sample ref\n%.3f @ %s'%(max_val_sample,str(max_loc_sample)))
        plt.figtext(0.365,0.29,r'$\ast$',va='center',ha='left')

        fig_laser_img=fig.add_subplot(234)        
        fig_laser_img.imshow(laser_img_crop)
        plt.axis('off')
        fig_laser_img.set_title('laser img')
        plt.figtext(0.64,0.29,'=',va='center',ha='left')
        
        fig_laser_only=fig.add_subplot(235)
        fig_laser_only.imshow(img_laser_only_crop)
        plt.axis('off')
        fig_laser_only.set_title('laser ref')
        fig_conv_laser = fig.add_subplot(236)
        fig_conv_laser.imshow(conv_data_laser)
        plt.axis('off')
        fig_conv_laser.set_title('Conv. with laser ref\n%.3f @ %s'%(max_val_laser,str(max_loc_laser)))
    
    return laser_spot_from_sample


def anal_scan_delayline_with_andor5(npy_file,min_wl=765,max_wl=795,fit=False,normalize=True,sc_wl=None,minus_bg=True,minus_iSHG=False,show_all_raw=False):
    global poss, spec_sums, specs, sc_powers, laser_powers, lonly_specs, sconly_specs, processed_specs, _figall, _fig, sconly_sc_powers
    main_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor5', npy_file)
    # data will be [[sc wavelength], wavelengths, [both on], [laser only], [sc only]]
    # each data point is appended in those last 3 lists, [fdl_pos, spectrum, sc_power, laser_power]
    data = np.load(os.path.join(main_path,npy_file + '.npy'))
    try:
        sc_wl = data[0][0]
    except:
        pass
    wls = data[1]
    both_beams = data[2]
    laser_only = data[3]
    sc_only = data[4]
    
    # data that will be analysed, regroup from data points into categories
    poss = list(map(lambda x: x[0],both_beams))
    specs = np.array(list(map(lambda x: x[1],both_beams)))
    sc_powers = np.array(list(map(lambda x: x[2],both_beams)))*1e9
    if sc_wl != None:
        sc_powers *= NPBS_TR_c_fx(sc_wl)
    laser_powers = np.array(list(map(lambda x: x[3],both_beams)))
    
    #other data
    lonly_poss = list(map(lambda x: x[0],laser_only))
    lonly_specs = np.array(list(map(lambda x: x[1],laser_only)))
    lonly_laser_powers = np.array(list(map(lambda x: x[3],laser_only)))
    
    sconly_poss = list(map(lambda x: x[0], sc_only))
    sconly_specs = np.array(list(map(lambda x: x[1],sc_only)))
    sconly_sc_powers = np.array(list(map(lambda x: x[2],sc_only)))*1e9
    if sc_wl != None:
        sconly_sc_powers *= NPBS_TR_c_fx(sc_wl)
    
    # Don't call with normalize True if there is not enough data (missing powermeter)
    
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)
    
    spec_sums = []
    processed_specs = []
    raw_both_specs, raw_sc_specs, raw_laser_specs = [], [], []
    raw_both_sums, raw_sc_sums, raw_laser_sums = [], [], []
    for i in range(len(specs)):
        curr_spec = np.median(specs[i],0)
#        curr_spec = remove_outlier3(range(len(curr_spec)),curr_spec,0.01)[1]
        if show_all_raw:
            curr_spec = np.median(specs[i],0)
            raw_both_specs.append(copy.copy(curr_spec))
            raw_both_sums.append(np.sum(curr_spec[min_idx:max_idx]))
            
            curr_sconly_spec = np.median(sconly_specs[i],0)
            raw_sc_specs.append(curr_sconly_spec)
            raw_sc_sums.append(np.sum(curr_sconly_spec[min_idx:max_idx]))
            
            curr_lonly_spec = np.median(lonly_specs[i],0)
            raw_laser_specs.append(curr_lonly_spec)
            raw_laser_sums.append(np.sum(curr_lonly_spec[min_idx:max_idx]))
        if fit:
            curr_popt = poly_fitter(wls[min_idx:max_idx],curr_spec[min_idx:max_idx],plot=False)
            curr_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
        else:
            curr_spec = curr_spec[min_idx:max_idx]
        if normalize:
            curr_spec /= sconly_sc_powers[i]
#                    curr_spec /= np.power(laser_powers[i],2)
        if minus_bg:
            curr_lonly_spec = np.median(lonly_specs[i],0)
#            curr_lonly_spec = remove_outlier3(range(len(curr_lonly_spec)),curr_lonly_spec,0.01)[1]
            if fit:
                curr_popt = poly_fitter(wls[min_idx:max_idx],curr_lonly_spec[min_idx:max_idx],plot=False)
                curr_lonly_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
            else:
                curr_lonly_spec = curr_lonly_spec[min_idx:max_idx]
            if normalize:
    #                    curr_lonly_spec /= np.power(laser_powers[i],2)
                pass
            curr_sconly_spec = np.median(sconly_specs[i],0)
#            curr_sconly_spec = remove_outlier3(range(len(curr_sconly_spec)),curr_sconly_spec,0.01)[1]
            if fit:
                curr_popt = poly_fitter(wls[min_idx:max_idx],curr_sconly_spec[min_idx:max_idx],plot=False)
                curr_sconly_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
            else:
                curr_sconly_spec = curr_sconly_spec[min_idx:max_idx]
            if normalize:
                curr_sconly_spec /= sconly_sc_powers[i]
            curr_signal_spec = curr_spec - curr_sconly_spec
        else:
            curr_signal_spec = curr_spec
        if minus_iSHG:
            curr_lonly_spec = (curr_lonly_spec - 544.5) #542.3 for 33kHz, 335.1 for 50kHz, 146.1 for 100 kHz
            if normalize:
                curr_lonly_spec /= sconly_sc_powers[i]
            curr_signal_spec = curr_signal_spec - curr_lonly_spec
        curr_spec_sum = np.sum(curr_signal_spec)
        spec_sums.append(curr_spec_sum)
        processed_specs.append(curr_signal_spec)
       
#    try:
#        min_y = np.min(processed_specs[min_idx-5:max_idx+5])
#        max_y = np.max(processed_specs[min_idx-5:max_idx+5])
#    except:
#        min_y = np.min(processed_specs[min_idx:max_idx])
#        max_y = np.max(processed_specs[min_idx:max_idx])
    _figall = plt.figure(npy_file)
    figall = _figall.add_subplot(121)
    
    for j,spec in enumerate(processed_specs):
        if fit:
            figall.plot(wls[min_idx:max_idx],spec,c=plt.cm.RdYlGn(255*j/(len(processed_specs)-1)))
        else:
            figall.plot(wls[min_idx:max_idx],spec,c=plt.cm.RdYlGn(255*j/(len(processed_specs)-1)))
#    figall.set_xlim(min_wl-5,max_wl+5)
#    figall.set_ylim(min_y-10,max_y)
    figall.set_xlabel('Wavelength, nm')
    figall.set_title(npy_file)
    fig = _figall.add_subplot(222)
    line, = fig.plot(poss,spec_sums,'o')
    _figall.suptitle(npy_file)
    fig.set_title('normed spec sum of %i to %i nm'%(min_wl,max_wl))
    fig.grid()
#    if sc_on:
#        fig.set_ylabel('spec sum/sc power/laser power$^2$, au')
#    else:
#        fig.set_ylabel('spec sum/laser power$^2$, au') #not true if not normalized?
    fig.set_ylabel('spec sum, au')
    
    figscpower = _figall.add_subplot(224)
    figlaserpower = figscpower.twinx()
    linesc, = figscpower.plot(poss,sc_powers,'o',color='C1')
    linelaser, = figlaserpower.plot(poss,laser_powers,'o',color='C2')
    figscpower.set_title('sc and laser powers monitor')
    figscpower.set_xlabel('fdl pos, mm')
    figscpower.set_ylabel('sc power, nW',color='C1')
    figscpower.grid()
    figlaserpower.set_ylabel('laser power, au',color='C2')
    plt.get_current_fig_manager().window.showMaximized()
#    _fig.tight_layout()
    poss = np.array(poss)
    spec_sums = np.array(spec_sums)
    
    if show_all_raw:
        _figall_raw = plt.figure(npy_file+' raw')
        figall_both = _figall_raw.add_subplot(321)
        figall_sc = _figall_raw.add_subplot(323)
        figall_laser = _figall_raw.add_subplot(325)
        
        for j,spec in enumerate(raw_both_specs):
            figall_both.plot(wls,raw_both_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            figall_sc.plot(wls,raw_sc_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            figall_laser.plot(wls,raw_laser_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            
        figall_both.set_ylabel('both beams on')
        figall_sc.set_ylabel('only sc')
        figall_laser.set_ylabel('only 1550')
        figall_laser.set_xlabel('Wavelength, nm')
        figall_both.set_title(npy_file+' raw')
        fig_both = _figall_raw.add_subplot(322)
        fig_sc = _figall_raw.add_subplot(324)
        fig_laser = _figall_raw.add_subplot(326)
        fig_both.plot(poss,raw_both_sums,'o')
        fig_sc.plot(poss,raw_sc_sums,'o')
        fig_laser.plot(poss,raw_laser_sums,'o')
        fig_both.set_title('raw spec sum of %i to %i nm'%(min_wl,max_wl))
        fig_both.grid()
        fig_sc.grid()
        fig_laser.grid()
        fig_sc.set_ylabel('raw spec sum, au')
        
        fig_laser.set_xlabel('fdl pos, mm')
        plt.get_current_fig_manager().window.showMaximized()
    
    plt.pause(1e-3)
    global img_dl_poss,laser_xs,laser_ys
    img_dl_poss,laser_xs,laser_ys = anal_images_scan_delayline_with_andor5(npy_file)

#%%
def anal_images_scan_delayline_with_andor6(npy_file,over_write=False,radius=150,plot=False,um_per_px=2./17,multiprocess=True,plot_each_run_with_different_color=False):
    main_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor6', npy_file) 
    img_path = os.path.join(main_path,'images')
    all_npy_files = list(filter(lambda x: x.endswith('.npy'),os.listdir(img_path)))
    all_raw_imgs = list(filter(lambda x: not x.endswith('_noBG.npy'),all_npy_files))
    all_laser_imgs = list(filter(lambda x: x.endswith('_laser.npy'),all_npy_files))
    all_processed_imgs = list(filter(lambda x: x.endswith('_noBG.npy'),all_npy_files))
    def img_name_to_tag(name):
        return tuple(map(int,name.split('.')[0].split('_')[:2]))
    def tag_to_img_name(tag):
        return '%i_%i.npy'%tag
    def process_img(path):
        image = np.load(path)
        subtract_background_rolling_ball(image, radius, light_background=False, use_paraboloid=True, do_presmooth=True)
        np.save(path[:-4]+'_noBG.npy', image)
    all_raw_img_tags = list(map(img_name_to_tag,all_raw_imgs))
    all_processed_img_tags = list(map(img_name_to_tag,all_processed_imgs))
    all_unprocessed_img_tags = list(set(all_raw_img_tags)-set(all_processed_img_tags)-set(all_laser_imgs))
    
    if over_write:
        all_unprocessed_img_tags = all_raw_img_tags
    
    total_len = len(all_unprocessed_img_tags)+1
    if len(all_unprocessed_img_tags):
        start_time = time.time()
        
        if multiprocess:
            print('Processing images, est. needs %s.'%sec_to_hhmmss(24.5*np.ceil(total_len/12.)))
            all_paths = list(map(lambda x: os.path.join(img_path,x), [ph for ph in all_raw_imgs if ph not in all_laser_imgs]))
            all_paths.append(os.path.join(main_path,'img_LED_only.npy'))
            Parallel(n_jobs=-1,verbose=50)(delayed(process_img)(path) for path in all_paths)
        else:
            process_img(os.path.join(main_path,'img_LED_only.npy'))
            elapsed_time = time.time() - start_time
            time_left = elapsed_time*(1.*total_len-1)
            completed = u'Processed %s (%.2f percent) %s left.'%('img_LED_only.npy',100.0*(float(1)/total_len),sec_to_hhmmss(time_left))
            prints(completed)
            prev = completed
            for j,tag in enumerate(all_unprocessed_img_tags):
                filename = tag_to_img_name(tag)
                path = os.path.join(img_path, filename)
                process_img(path)
                
                elapsed_time = time.time() - start_time
                time_left = elapsed_time*(1.*total_len/(j+1+1)-1)
                completed = u'Processed %s (%.2f percent) %s left.'%(filename,100.0*(float(j+1+1)/total_len),sec_to_hhmmss(time_left))
                prints(completed,prev)
                prev = completed
                plt.pause(1e-6)
    else:
        print('All images had been processed.')
    
    print('Extracting coordinates of laser spot on sample...')
    img_info = np.load(os.path.join(main_path,'img_info.npy'))
    img_info_temp = list(map(lambda x: [x[0],int(x[1]),float(x[2]),float(x[3])],img_info.astype(np.object)))
    sample_ref_img = np.load(os.path.join(main_path,'img_LED_only_noBG.npy'))
    laser_ref_img = np.load(os.path.join(main_path,'img_laser_only.npy'))
    for i,img_info_ele in enumerate(img_info):
        img_name = img_info_ele[0]+'_noBG.npy'
        curr_data_img = np.load(os.path.join(img_path,img_name))
        if all_laser_imgs:
            curr_laser_img = np.load(os.path.join(img_path, all_laser_imgs[i]))
            curr_x, curr_y = extract_coords_of_laser_on_sample_2(curr_data_img,curr_laser_img,sample_ref_img,laser_ref_img,plot=plot,um_per_px=um_per_px)
        else:
            curr_x, curr_y = extract_coords_of_laser_on_sample(curr_data_img,sample_ref_img,laser_ref_img,plot=plot,um_per_px=um_per_px)
        img_info_temp[i].append(curr_x)
        img_info_temp[i].append(curr_y)
    # img info will be [[img npy name, rep, fdl pos, time since start, laser x pos on sample, laser y pos on sample]]
    np.save(os.path.join(main_path,'img_info.npy'),np.array(img_info_temp))
    dl_poss = np.array(list(map(lambda x: x[2],img_info_temp)))
    laser_xs = np.array(list(map(lambda x: x[4],img_info_temp)))
    laser_ys = np.array(list(map(lambda x: x[5],img_info_temp)))
    
    fig = plt.figure(npy_file+' laser spot')
    plt.get_current_fig_manager().window.showMaximized()
    fig.suptitle(npy_file+' laser spot')
    figx = fig.add_subplot(211)
    if plot_each_run_with_different_color:
        rep_len = len(poss)/(np.argmin(np.diff(dl_poss)))
        pos_len = len(poss)/rep_len
        laser_xs_reshaped = laser_xs.reshape((rep_len,pos_len))
        for j in range(rep_len):
            figx.plot(dl_poss[:pos_len],laser_xs_reshaped[j]-laser_xs[0],'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
    else:
        figx.plot(dl_poss,laser_xs-laser_xs[0],'o')
    figx.set_ylabel('$\Delta$x, um')
    figx.axhline(0,c='k')
    figx.grid()
    figy = fig.add_subplot(212)
    if plot_each_run_with_different_color:
        laser_ys_reshaped = laser_ys.reshape((rep_len,pos_len))
        for j in range(rep_len):
            figy.plot(dl_poss[:pos_len],laser_ys_reshaped[j]-laser_ys[0],'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
    else:
        figy.plot(dl_poss,laser_ys-laser_ys[0],'o')
    figy.set_ylabel('$\Delta$y, um')
    figy.set_xlabel('fdl pos, mm')
    figy.axhline(0,c='k')
    figy.grid()
    plt.pause(0.1)
    fig.savefig(os.path.join(main_path,'laser spot shift.png'))
    return dl_poss,laser_xs,laser_ys
    
        
def view_images_scan_delayline_with_andor6(npy_file, noBG=True):
    main_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor6', npy_file) 
    img_path = os.path.join(main_path,'images')
    keyword = ''
    if noBG:
        keyword = 'noBG'
    imshow_npy_files_with_word_in(img_path, keyword)

def anal_scan_delayline_with_andor6(npy_file,min_wl=765,max_wl=795,fit=False,normalize=True,minus_bg=True,minus_iSHG=False,show_all_raw=False,plot_each_run_with_different_color=True):
    global poss, spec_sums, specs, sc_powers, laser_powers, lonly_specs, sconly_specs, processed_specs, _figall, _fig, sconly_sc_powers
    main_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor6', npy_file)
    # data will be [[sc wavelength], wavelengths, [both on], [laser only], [sc only]]
    # each data point is appended in those last 3 lists, [fdl_pos, spectrum, sc_power, laser_power, sc_gamma (if applicable)]
    # sc_gamma is a measure of the reference polarisation of the SC (after 1 reflection by NPBS) parametrised by alpha, gamma
    data = np.load(os.path.join(main_path,npy_file + '.npy'))
    
    wls = data[1]
    both_beams = data[2]
    laser_only = data[3]
    sc_only = data[4]
    
    # data that will be analysed, regroup from data points into categories
    poss = list(map(lambda x: x[0], both_beams))
    specs = np.array(list(map(lambda x: x[1], both_beams)))
    sc_powers = np.array(list(map(lambda x: x[2], both_beams)))
    laser_powers = np.array(list(map(lambda x: x[3], both_beams)))
    sc_gammas = np.array(list(map(lambda x: x[4], both_beams)))
    
    #other data
    lonly_poss = list(map(lambda x: x[0], laser_only))
    lonly_specs = np.array(list(map(lambda x: x[1], laser_only)))
    lonly_laser_powers = np.array(list(map(lambda x: x[3], laser_only)))
    
    sconly_poss = list(map(lambda x: x[0], sc_only))
    sconly_specs = np.array(list(map(lambda x: x[1], sc_only)))
    sconly_sc_powers = np.array(list(map(lambda x: x[2], sc_only)))
    sconly_sc_gammas = np.array(list(map(lambda x: x[4], sc_only)))
    
    # Don't call with normalize True if there is not enough data (missing powermeter)
    
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)
    
    spec_sums = []
    processed_specs = []
    raw_both_specs, raw_sc_specs, raw_laser_specs = [], [], []
    raw_both_sums, raw_sc_sums, raw_laser_sums = [], [], []
    for i in range(len(specs)):
        curr_spec = np.median(specs[i],0)
#        curr_spec = remove_outlier3(range(len(curr_spec)),curr_spec,0.01)[1]
        if show_all_raw:
            curr_spec = np.median(specs[i],0)
            raw_both_specs.append(copy.copy(curr_spec))
            raw_both_sums.append(np.sum(curr_spec[min_idx:max_idx]))
            
            curr_sconly_spec = np.median(sconly_specs[i],0)
            raw_sc_specs.append(curr_sconly_spec)
            raw_sc_sums.append(np.sum(curr_sconly_spec[min_idx:max_idx]))
            
            curr_lonly_spec = np.median(lonly_specs[i],0)
            raw_laser_specs.append(curr_lonly_spec)
            raw_laser_sums.append(np.sum(curr_lonly_spec[min_idx:max_idx]))
        if fit:
            curr_popt = poly_fitter(wls[min_idx:max_idx],curr_spec[min_idx:max_idx],plot=False)
            curr_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
        else:
            curr_spec = curr_spec[min_idx:max_idx]
        if normalize:
            curr_spec /= sconly_sc_powers[i]
#                    curr_spec /= np.power(laser_powers[i],2)
        if minus_bg:
            curr_lonly_spec = np.median(lonly_specs[i],0)
#            curr_lonly_spec = remove_outlier3(range(len(curr_lonly_spec)),curr_lonly_spec,0.01)[1]
            if fit:
                curr_popt = poly_fitter(wls[min_idx:max_idx],curr_lonly_spec[min_idx:max_idx],plot=False)
                curr_lonly_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
            else:
                curr_lonly_spec = curr_lonly_spec[min_idx:max_idx]
            if normalize:
    #                    curr_lonly_spec /= np.power(laser_powers[i],2)
                pass
            curr_sconly_spec = np.median(sconly_specs[i],0)
#            curr_sconly_spec = remove_outlier3(range(len(curr_sconly_spec)),curr_sconly_spec,0.01)[1]
            if fit:
                curr_popt = poly_fitter(wls[min_idx:max_idx],curr_sconly_spec[min_idx:max_idx],plot=False)
                curr_sconly_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
            else:
                curr_sconly_spec = curr_sconly_spec[min_idx:max_idx]
            if normalize:
                curr_sconly_spec /= sconly_sc_powers[i]
            curr_signal_spec = curr_spec - curr_sconly_spec
        else:
            curr_signal_spec = curr_spec
        if minus_iSHG:
            curr_lonly_spec = (curr_lonly_spec - 544.5) #542.3 for 33kHz, 335.1 for 50kHz, 146.1 for 100 kHz
            if normalize:
                curr_lonly_spec /= sconly_sc_powers[i]
            curr_signal_spec = curr_signal_spec - curr_lonly_spec
        curr_spec_sum = np.sum(curr_signal_spec)
        spec_sums.append(curr_spec_sum)
        processed_specs.append(curr_signal_spec)
       
#    try:
#        min_y = np.min(processed_specs[min_idx-5:max_idx+5])
#        max_y = np.max(processed_specs[min_idx-5:max_idx+5])
#    except:
#        min_y = np.min(processed_specs[min_idx:max_idx])
#        max_y = np.max(processed_specs[min_idx:max_idx])
    _figall = plt.figure(npy_file)
    figall = _figall.add_subplot(121)
    
    for j,spec in enumerate(processed_specs):
        if fit:
            figall.plot(wls[min_idx:max_idx],spec,c=plt.cm.RdYlGn(255*j/(len(processed_specs)-1)))
        else:
            figall.plot(wls[min_idx:max_idx],spec,c=plt.cm.RdYlGn(255*j/(len(processed_specs)-1)))
#    figall.set_xlim(min_wl-5,max_wl+5)
#    figall.set_ylim(min_y-10,max_y)
    figall.set_xlabel('Wavelength, nm')
    figall.set_title(npy_file)
    fig = _figall.add_subplot(222)
    if plot_each_run_with_different_color:
        rep_len = len(poss)/(np.argmin(np.diff(poss)))
        pos_len = len(poss)/rep_len
        spec_sums_reshaped = np.array(spec_sums).reshape((rep_len,pos_len))
        for j,spec_sum in enumerate(spec_sums_reshaped):
            line,= fig.plot(poss[:pos_len],spec_sum,'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
        m = np.mean(spec_sums_reshaped,0)
        e = np.std(spec_sums_reshaped,0)/np.sqrt(rep_len)
        fig.errorbar(poss[:pos_len],m,e,ms=3,lw=0,elinewidth=1,capsize=2,marker='o',color='k')
    else:
        line, = fig.plot(poss,spec_sums,'o')

    _figall.suptitle(npy_file)
    fig.set_title('normed spec sum of %i to %i nm'%(min_wl,max_wl))
    fig.grid()
#    if sc_on:
#        fig.set_ylabel('spec sum/sc power/laser power$^2$, au')
#    else:
#        fig.set_ylabel('spec sum/laser power$^2$, au') #not true if not normalized?
    fig.set_ylabel('spec sum, au')
    
    figscpower = _figall.add_subplot(224)
    figlaserpower = figscpower.twinx()
    linesc, = figscpower.plot(poss,sc_powers,'o',color='C1')
    linelaser, = figlaserpower.plot(poss,laser_powers,'o',color='C2')
    figscpower.set_title('sc and laser powers monitor')
    figscpower.set_xlabel('fdl pos, mm')
    figscpower.set_ylabel('sc power, nW',color='C1')
    figscpower.grid()
    figlaserpower.set_ylabel('laser power, au',color='C2')
    plt.get_current_fig_manager().window.showMaximized()
#    _fig.tight_layout()
    poss = np.array(poss)
    spec_sums = np.array(spec_sums)
    
    if show_all_raw:
        _figall_raw = plt.figure(npy_file+' raw')
        figall_both = _figall_raw.add_subplot(321)
        figall_sc = _figall_raw.add_subplot(323)
        figall_laser = _figall_raw.add_subplot(325)
        
        for j,spec in enumerate(raw_both_specs):
            figall_both.plot(wls,raw_both_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            figall_sc.plot(wls,raw_sc_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            figall_laser.plot(wls,raw_laser_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            
        figall_both.set_ylabel('both beams on')
        figall_sc.set_ylabel('only sc')
        figall_laser.set_ylabel('only 1550')
        figall_laser.set_xlabel('Wavelength, nm')
        figall_both.set_title(npy_file+' raw')
        fig_both = _figall_raw.add_subplot(322)
        fig_sc = _figall_raw.add_subplot(324)
        fig_laser = _figall_raw.add_subplot(326)
        
        if plot_each_run_with_different_color:
            raw_both_sums_reshaped = np.array(raw_both_sums).reshape((rep_len,pos_len))
            raw_sc_sums_reshaped = np.array(raw_sc_sums).reshape((rep_len,pos_len))
            raw_laser_sums_reshaped = np.array(raw_laser_sums).reshape((rep_len,pos_len))
            for j in range(len(raw_both_sums_reshaped)):
                fig_both.plot(poss[:pos_len],raw_both_sums_reshaped[j],'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
                fig_sc.plot(poss[:pos_len],raw_sc_sums_reshaped[j],'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
                fig_laser.plot(poss[:pos_len],raw_laser_sums_reshaped[j],'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
        else:
            fig_both.plot(poss,raw_both_sums,'o')
            fig_sc.plot(poss,raw_sc_sums,'o')
            fig_laser.plot(poss,raw_laser_sums,'o')
            
        fig_both.set_title('raw spec sum of %i to %i nm'%(min_wl,max_wl))
        fig_both.grid()
        fig_sc.grid()
        fig_laser.grid()
        fig_sc.set_ylabel('raw spec sum, au')
        
        fig_laser.set_xlabel('fdl pos, mm')
        plt.get_current_fig_manager().window.showMaximized()
    
    plt.pause(0.1)
    _figall.savefig(os.path.join(main_path,'summary.png'))
    if show_all_raw:
        _figall_raw.savefig(os.path.join(main_path,'all raw.png'))
        
    global img_dl_poss,laser_xs,laser_ys
    try:
        img_dl_poss,laser_xs,laser_ys = anal_images_scan_delayline_with_andor6(npy_file,um_per_px=2./17,plot_each_run_with_different_color=plot_each_run_with_different_color)
    except cv2.error:
        print('%s failed to extract laser spot'%npy_file)

#%%
class DataAndorScanDL7():
    def __init__(self, npy):
        self._L = 0
        self._R = 1
        self._H = 2
        self._V = 3
        self._pol_dic = {"L":self._L,"R":self._R,"H":self._H,"V":self._V}
        
        self.raw_data = npy[1]
        self.andor_wls = npy[0][0]
        self.sc_wl = npy[0][1]
        
        self.tags = np.array([datum[0] for datum in self.raw_data])
        self.raw_spectra = np.array([datum[1][0] for datum in self.raw_data])
        self.spectra = np.median(self.raw_spectra,axis=1)
        self.ave_laser_powers = np.array([datum[1][1] for datum in self.raw_data])
        self.ave_sc_powers = np.array([datum[1][2] for datum in self.raw_data])
        
        self.repeat_nums = sorted(list(set([datum[0][0] for datum in self.raw_data]))) #list of available repeat number
        self.fdl_poss = sorted(list(set([datum[0][1] for datum in self.raw_data]))) #list of available fdl pos
        self.laser_combis = sorted(list(set([datum[0][2] for datum in self.raw_data]))) #list of available laser combinations
        self.ana_angs = sorted(list(set([datum[0][3] for datum in self.raw_data]))) #list of available analyzer angles
        self.sc_pols = sorted(list(set([datum[0][4] for datum in self.raw_data]))) #list of available sc polarization
        self.sc_alphas = sorted(list(set([datum[0][5] for datum in self.raw_data]))) #list of available sc ref alphas
        self.sc_gammas = sorted(list(set([datum[0][6] for datum in self.raw_data]))) #list of available sc ref gammas
        
        #reshaped stuff = [rep num, fdl pos, sc pol, ana ang, laser combi]
        self.reshaped_spectra = self.spectra.reshape((len(self.repeat_nums),len(self.fdl_poss),len(self.sc_pols),len(self.ana_angs),len(self.laser_combis),len(self.andor_wls)))
        self.reshaped_ave_laser_powers = self.ave_laser_powers.reshape((len(self.repeat_nums),len(self.fdl_poss),len(self.sc_pols),len(self.ana_angs),len(self.laser_combis)))
        self.reshaped_ave_sc_powers = self.ave_sc_powers.reshape((len(self.repeat_nums),len(self.fdl_poss),len(self.sc_pols),len(self.ana_angs),len(self.laser_combis)))
        self.reshaped_tags = self.tags.reshape((len(self.repeat_nums),len(self.fdl_poss),len(self.sc_pols),len(self.ana_angs),len(self.laser_combis),len(self.tags[0])))
        
        self.repeat_num_dic = {}
        for i in range(len(self.repeat_nums)):
            curr_val = self.reshaped_tags[i,0,0,0,0][0]
            self.repeat_num_dic[curr_val] = i
            
        self.fdl_pos_dic = {}
        for i in range(len(self.fdl_poss)):
            curr_val = self.reshaped_tags[0,i,0,0,0][1]
            self.fdl_pos_dic[curr_val] = i
        
        self.sc_pol_dic = {}
        for i in range(len(self.sc_pols)):
            curr_val = self.reshaped_tags[0,0,i,0,0][2]
            self.sc_pol_dic[curr_val] = i
        
        self.ana_ang_dic = {}
        for i in range(len(self.ana_angs)):
            curr_val = self.reshaped_tags[0,0,0,i,0][3]
            self.ana_ang_dic[curr_val] = i
        
        self.laser_combi_dic = {}
        for i in range(len(self.laser_combis)):
            curr_val = self.reshaped_tags[0,0,0,0,i][4]
            self.laser_combi_dic[curr_val] = i
    
    def get_repeat_num_from_tag(tag):
        return tag[0]
    
    def get_fld_pos_from_tag(tag):
        return tag[1]
    
    def get_laser_combi_from_tag(tag):
        return tag[2]
    
    def get_ana_ang_from_tag(tag):
        return tag[3]
    
    def get_sc_pol_from_tag(tag):
        return tag[4]
    
    def get_sc_alpha_from_tag(tag):
        return tag[5]
    
    def get_sc_gamma_from_tag(tag):
        return tag[6]
    
    def get_spectra(self,repeat_num,fdl_pos,sc_pol,ana_ang,laser_combi,return_tag=False):
        """
        only use "all" once.
        """
        #sanity check
        if repeat_num != 'all':
            repeat_num = get_nearest_values_from_list(repeat_num,self.repeat_nums)
        if fdl_pos != 'all':
            fdl_pos = get_nearest_values_from_list(fdl_pos,self.fdl_poss)
        if laser_combi != 'all':
            laser_combi = get_nearest_values_from_list(laser_combi,self.laser_combis)
        if ana_ang != 'all':
            ana_ang = get_nearest_values_from_list(ana_ang,self.ana_angs)
        if sc_pol not in ["L","R","H","V","all"]:
            print("Invalid input sc_pol: %s, use ['L','R','H','V','all']."%(str(sc_pol)))
            return
        elif sc_pol == 'all':
            pass
        else: #convert to integer
            sc_pol = self._pol_dic[sc_pol]
        
        if repeat_num != 'all':
            repeat_num_idx = self.repeat_num_dic[repeat_num]
        if fdl_pos != 'all':
            fdl_pos_idx = self.fdl_pos_dic[fdl_pos]
        if sc_pol != 'all':
            sc_pol_idx = self.sc_pol_dic[sc_pol]
        if ana_ang != 'all':
            ana_ang_idx = self.ana_ang_dic[ana_ang]
        if laser_combi != 'all':
            laser_combi_idx = self.laser_combi_dic[laser_combi]
        
        if repeat_num == 'all':
            ans = self.reshaped_spectra[:, fdl_pos_idx, sc_pol_idx, ana_ang_idx, laser_combi_idx]
            tag = self.reshaped_tags[:, fdl_pos_idx, sc_pol_idx, ana_ang_idx, laser_combi_idx][:,0]
        elif fdl_pos == 'all':
            ans = self.reshaped_spectra[repeat_num_idx, :, sc_pol_idx, ana_ang_idx, laser_combi_idx]
            tag = self.reshaped_tags[repeat_num_idx, :, sc_pol_idx, ana_ang_idx, laser_combi_idx][:,1]
        elif sc_pol == 'all':
            ans = self.reshaped_spectra[repeat_num_idx, fdl_pos_idx, :, ana_ang_idx, laser_combi_idx]
            tag = self.reshaped_tags[repeat_num_idx, fdl_pos_idx, :, ana_ang_idx, laser_combi_idx][:,2]
        elif ana_ang == 'all':
            ans = self.reshaped_spectra[repeat_num_idx, fdl_pos_idx, sc_pol_idx, :, laser_combi_idx]
            tag = self.reshaped_tags[repeat_num_idx, fdl_pos_idx, sc_pol_idx, :, laser_combi_idx][:,3]
        elif laser_combi == 'all':
            ans = self.reshaped_spectra[repeat_num_idx, fdl_pos_idx, sc_pol_idx, ana_ang_idx, :]
            tag = self.reshaped_tags[repeat_num_idx, fdl_pos_idx, sc_pol_idx, ana_ang_idx, :][:,4]
        else:
            ans = self.reshaped_spectra[repeat_num_idx, fdl_pos_idx, sc_pol_idx, ana_ang_idx, laser_combi_idx]
            tag = [repeat_num,fdl_pos,sc_pol,ana_ang,laser_combi]
        
        if return_tag:
            return [tag,ans]
        else:
            return ans

def anal_scan_delayline_with_andor7(npy_file,min_wl=765,max_wl=795,fit=False,normalize=True,minus_bg=True,minus_iSHG=False,show_all_raw=False,plot_each_run_with_different_color=True):
    """Everything needs to be changed."""
    return

    global poss, spec_sums, specs, sc_powers, laser_powers, lonly_specs, sconly_specs, processed_specs, _figall, _fig, sconly_sc_powers
    main_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor7', npy_file)
    # data will be [[sc wavelength], wavelengths, [both on], [laser only], [sc only]]
    # each data point is appended in those last 3 lists, [fdl_pos, spectrum, sc_power, laser_power, sc_gamma (if applicable)]
    # sc_gamma is a measure of the reference polarisation of the SC (after 1 reflection by NPBS) parametrised by alpha, gamma
    data = DataAndorScanDL7(np.load(os.path.join(main_path,'data.npy')))
    
    wls = data.andor_wls
    both_beams = data[2]
    laser_only = data[3]
    sc_only = data[4]
    
    # data that will be analysed, regroup from data points into categories
    poss = list(map(lambda x: x[0], both_beams))
    specs = np.array(list(map(lambda x: x[1], both_beams)))
    sc_powers = np.array(list(map(lambda x: x[2], both_beams)))
    laser_powers = np.array(list(map(lambda x: x[3], both_beams)))
    sc_gammas = np.array(list(map(lambda x: x[4], both_beams)))
    
    #other data
    lonly_poss = list(map(lambda x: x[0], laser_only))
    lonly_specs = np.array(list(map(lambda x: x[1], laser_only)))
    lonly_laser_powers = np.array(list(map(lambda x: x[3], laser_only)))
    
    sconly_poss = list(map(lambda x: x[0], sc_only))
    sconly_specs = np.array(list(map(lambda x: x[1], sc_only)))
    sconly_sc_powers = np.array(list(map(lambda x: x[2], sc_only)))
    sconly_sc_gammas = np.array(list(map(lambda x: x[4], sc_only)))
    
    # Don't call with normalize True if there is not enough data (missing powermeter)
    
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)
    
    spec_sums = []
    processed_specs = []
    raw_both_specs, raw_sc_specs, raw_laser_specs = [], [], []
    raw_both_sums, raw_sc_sums, raw_laser_sums = [], [], []
    for i in range(len(specs)):
        curr_spec = np.median(specs[i],0)
#        curr_spec = remove_outlier3(range(len(curr_spec)),curr_spec,0.01)[1]
        if show_all_raw:
            curr_spec = np.median(specs[i],0)
            raw_both_specs.append(copy.copy(curr_spec))
            raw_both_sums.append(np.sum(curr_spec[min_idx:max_idx]))
            
            curr_sconly_spec = np.median(sconly_specs[i],0)
            raw_sc_specs.append(curr_sconly_spec)
            raw_sc_sums.append(np.sum(curr_sconly_spec[min_idx:max_idx]))
            
            curr_lonly_spec = np.median(lonly_specs[i],0)
            raw_laser_specs.append(curr_lonly_spec)
            raw_laser_sums.append(np.sum(curr_lonly_spec[min_idx:max_idx]))
        if fit:
            curr_popt = poly_fitter(wls[min_idx:max_idx],curr_spec[min_idx:max_idx],plot=False)
            curr_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
        else:
            curr_spec = curr_spec[min_idx:max_idx]
        if normalize:
            curr_spec /= sconly_sc_powers[i]
#                    curr_spec /= np.power(laser_powers[i],2)
        if minus_bg:
            curr_lonly_spec = np.median(lonly_specs[i],0)
#            curr_lonly_spec = remove_outlier3(range(len(curr_lonly_spec)),curr_lonly_spec,0.01)[1]
            if fit:
                curr_popt = poly_fitter(wls[min_idx:max_idx],curr_lonly_spec[min_idx:max_idx],plot=False)
                curr_lonly_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
            else:
                curr_lonly_spec = curr_lonly_spec[min_idx:max_idx]
            if normalize:
    #                    curr_lonly_spec /= np.power(laser_powers[i],2)
                pass
            curr_sconly_spec = np.median(sconly_specs[i],0)
#            curr_sconly_spec = remove_outlier3(range(len(curr_sconly_spec)),curr_sconly_spec,0.01)[1]
            if fit:
                curr_popt = poly_fitter(wls[min_idx:max_idx],curr_sconly_spec[min_idx:max_idx],plot=False)
                curr_sconly_spec = np.poly1d(curr_popt)(wls[min_idx:max_idx])
            else:
                curr_sconly_spec = curr_sconly_spec[min_idx:max_idx]
            if normalize:
                curr_sconly_spec /= sconly_sc_powers[i]
            curr_signal_spec = curr_spec - curr_sconly_spec
        else:
            curr_signal_spec = curr_spec
        if minus_iSHG:
            curr_lonly_spec = (curr_lonly_spec - 544.5) #542.3 for 33kHz, 335.1 for 50kHz, 146.1 for 100 kHz
            if normalize:
                curr_lonly_spec /= sconly_sc_powers[i]
            curr_signal_spec = curr_signal_spec - curr_lonly_spec
        curr_spec_sum = np.sum(curr_signal_spec)
        spec_sums.append(curr_spec_sum)
        processed_specs.append(curr_signal_spec)
       
#    try:
#        min_y = np.min(processed_specs[min_idx-5:max_idx+5])
#        max_y = np.max(processed_specs[min_idx-5:max_idx+5])
#    except:
#        min_y = np.min(processed_specs[min_idx:max_idx])
#        max_y = np.max(processed_specs[min_idx:max_idx])
    _figall = plt.figure(npy_file)
    figall = _figall.add_subplot(121)
    
    for j,spec in enumerate(processed_specs):
        if fit:
            figall.plot(wls[min_idx:max_idx],spec,c=plt.cm.RdYlGn(255*j/(len(processed_specs)-1)))
        else:
            figall.plot(wls[min_idx:max_idx],spec,c=plt.cm.RdYlGn(255*j/(len(processed_specs)-1)))
#    figall.set_xlim(min_wl-5,max_wl+5)
#    figall.set_ylim(min_y-10,max_y)
    figall.set_xlabel('Wavelength, nm')
    figall.set_title(npy_file)
    fig = _figall.add_subplot(222)
    if plot_each_run_with_different_color:
        rep_len = len(poss)/(np.argmin(np.diff(poss)))
        pos_len = len(poss)/rep_len
        spec_sums_reshaped = np.array(spec_sums).reshape((rep_len,pos_len))
        for j,spec_sum in enumerate(spec_sums_reshaped):
            line,= fig.plot(poss[:pos_len],spec_sum,'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
        m = np.mean(spec_sums_reshaped,0)
        e = np.std(spec_sums_reshaped,0)/np.sqrt(rep_len)
        fig.errorbar(poss[:pos_len],m,e,ms=3,lw=0,elinewidth=1,capsize=2,marker='o',color='k')
    else:
        line, = fig.plot(poss,spec_sums,'o')

    _figall.suptitle(npy_file)
    fig.set_title('normed spec sum of %i to %i nm'%(min_wl,max_wl))
    fig.grid()
#    if sc_on:
#        fig.set_ylabel('spec sum/sc power/laser power$^2$, au')
#    else:
#        fig.set_ylabel('spec sum/laser power$^2$, au') #not true if not normalized?
    fig.set_ylabel('spec sum, au')
    
    figscpower = _figall.add_subplot(224)
    figlaserpower = figscpower.twinx()
    linesc, = figscpower.plot(poss,sc_powers,'o',color='C1')
    linelaser, = figlaserpower.plot(poss,laser_powers,'o',color='C2')
    figscpower.set_title('sc and laser powers monitor')
    figscpower.set_xlabel('fdl pos, mm')
    figscpower.set_ylabel('sc power, nW',color='C1')
    figscpower.grid()
    figlaserpower.set_ylabel('laser power, au',color='C2')
    plt.get_current_fig_manager().window.showMaximized()
#    _fig.tight_layout()
    poss = np.array(poss)
    spec_sums = np.array(spec_sums)
    
    if show_all_raw:
        _figall_raw = plt.figure(npy_file+' raw')
        figall_both = _figall_raw.add_subplot(321)
        figall_sc = _figall_raw.add_subplot(323)
        figall_laser = _figall_raw.add_subplot(325)
        
        for j,spec in enumerate(raw_both_specs):
            figall_both.plot(wls,raw_both_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            figall_sc.plot(wls,raw_sc_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            figall_laser.plot(wls,raw_laser_specs[j],c=plt.cm.RdYlGn(255*j/(len(raw_both_specs)-1)))
            
        figall_both.set_ylabel('both beams on')
        figall_sc.set_ylabel('only sc')
        figall_laser.set_ylabel('only 1550')
        figall_laser.set_xlabel('Wavelength, nm')
        figall_both.set_title(npy_file+' raw')
        fig_both = _figall_raw.add_subplot(322)
        fig_sc = _figall_raw.add_subplot(324)
        fig_laser = _figall_raw.add_subplot(326)
        
        if plot_each_run_with_different_color:
            raw_both_sums_reshaped = np.array(raw_both_sums).reshape((rep_len,pos_len))
            raw_sc_sums_reshaped = np.array(raw_sc_sums).reshape((rep_len,pos_len))
            raw_laser_sums_reshaped = np.array(raw_laser_sums).reshape((rep_len,pos_len))
            for j in range(len(raw_both_sums_reshaped)):
                fig_both.plot(poss[:pos_len],raw_both_sums_reshaped[j],'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
                fig_sc.plot(poss[:pos_len],raw_sc_sums_reshaped[j],'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
                fig_laser.plot(poss[:pos_len],raw_laser_sums_reshaped[j],'o',c=plt.cm.RdYlGn(255*j/(rep_len-1)))
        else:
            fig_both.plot(poss,raw_both_sums,'o')
            fig_sc.plot(poss,raw_sc_sums,'o')
            fig_laser.plot(poss,raw_laser_sums,'o')
            
        fig_both.set_title('raw spec sum of %i to %i nm'%(min_wl,max_wl))
        fig_both.grid()
        fig_sc.grid()
        fig_laser.grid()
        fig_sc.set_ylabel('raw spec sum, au')
        
        fig_laser.set_xlabel('fdl pos, mm')
        plt.get_current_fig_manager().window.showMaximized()
    
    plt.pause(0.1)
    _figall.savefig(os.path.join(main_path,'summary.png'))
    if show_all_raw:
        _figall_raw.savefig(os.path.join(main_path,'all raw.png'))
        
    global img_dl_poss,laser_xs,laser_ys
    try:
        img_dl_poss,laser_xs,laser_ys = anal_images_scan_delayline_with_andor6(npy_file,um_per_px=2./17,plot_each_run_with_different_color=plot_each_run_with_different_color)
    except cv2.error:
        print('%s failed to extract laser spot'%npy_file)

#%%

def real_time_anal_scan_delayline_with_andor3(npy_file,min_wl=760,max_wl=790,refresh_delay=1,normalize=True):
    data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor3',npy_file+'.npy'))
#    wls = data[1][5:-5]
    wls = data[1]
    min_idx = get_nearest_idx_from_list(min_wl,wls)
    max_idx = get_nearest_idx_from_list(max_wl,wls)
    _fig,fig,line,figall,figscpower,figlaserpower,linesc,linelaser = anal_scan_delayline_with_andor3(npy_file,min_wl=min_wl,max_wl=max_wl,normalize=normalize)
    prev_data_len = 0
    num = 0
    global spec_sums, raw_spec_sums, sc_powers,laser_powers
    while True:
        try:
            data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor3',npy_file+'.npy'))
        except:
            plt.pause(0.1)
            data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_andor3',npy_file+'.npy'))
        data = data[2:]
        curr_data_len = len(data)
        if curr_data_len == prev_data_len:
            if num > 10:
                return
            plt.pause(refresh_delay)
            num += 1
            continue
        else:
            num = 0
        poss = list(map(lambda x: x[0],data))
        specs = np.array(list(map(lambda x: x[1],data)))
        sc_powers = np.array(list(map(lambda x: x[2],data)))
        laser_powers = np.array(list(map(lambda x: x[3],data)))
        spec_sums = np.array(list(map(lambda spec:           np.sum(list(map(lambda x: x[min_idx:max_idx] - np.median(np.append(x[:min_idx],x[max_idx:])),spec)))        ,specs)))
        raw_spec_sums = spec_sums
        if normalize:
            spec_sums = spec_sums/sc_powers/np.square(laser_powers)
        
        line.set_xdata(poss)
        line.set_ydata(spec_sums)
        fig.relim(True)
        fig.autoscale_view(True,True,True)
        linesc.set_xdata(poss)
        linesc.set_ydata(sc_powers)
        linelaser.set_xdata(poss)
        linelaser.set_ydata(laser_powers)
        figscpower.relim(True)
        figscpower.autoscale_view(True,True,True)
        figlaserpower.relim(True)
        figlaserpower.autoscale_view(True,True,True)
        
        while curr_data_len > len(figall.lines):
            if len(specs.shape) > 2:
                figall.plot(wls,np.mean(specs[len(figall.lines)],0))
            else:
                figall.plot(wls,specs[len(figall.lines)])
        
        plt.draw_all()
        prev_data_len = curr_data_len
        plt.pause(refresh_delay)

#%%

def real_time_anal_scan_delayline(npy_file,min_wl=500,max_wl=515,refresh_delay=1):
    min_idx = get_nearest_idx_from_list(min_wl,hero_pixel_wavelengths)
    max_idx = get_nearest_idx_from_list(max_wl,hero_pixel_wavelengths)
    _fig, fig, line, figall, fig2, line2 = anal_scan_delayline(npy_file,min_wl=min_wl,max_wl=max_wl)
    prev_data_len = 0
    num = 0
    global spec_sums
    while True:
        try:
            data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
        except:
            plt.pause(0.1)
            data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline',npy_file+'.npy'))
        
        curr_data_len = len(data)
        if curr_data_len == prev_data_len:
            if num > 10:
                return
            plt.pause(refresh_delay)
            num += 1
            continue
        else:
            num = 0
        poss = list(map(lambda x: x[0][0],data))
        lockins = np.array(map(lambda x: x[0][1],data))*1e3
        specs = list(map(lambda x: x[1],data))
        spec_sums = list(map(lambda x: np.sum(x[min_idx:max_idx]),specs))
        
        line.set_xdata(poss)
        line.set_ydata(spec_sums)
        fig.relim(True)
        fig.autoscale_view(True,True,True)
        
        line2.set_xdata(poss)
        line2.set_ydata(lockins)
        fig2.relim(True)
        fig2.autoscale_view(True,True,True)
        
        while curr_data_len > len(figall.lines):
            figall.plot(hero_pixel_wavelengths,specs[len(figall.lines)])
        
        plt.draw_all()
        prev_data_len = curr_data_len
        plt.pause(refresh_delay)

#%%
def anal_scan_delayline_with_bg(npy_file,min_wl=770,max_wl=777.5,fdl_pos_start='all',fdl_pos_end='all',lockin_as_pump_ref_order=2,plot_scan_fdl=True):
    global poss, spec_sums, lockins
    try:
        data = np.load(os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_bg',npy_file+'.npy'))
    except:
        supsample = npy_file.split('_a')[0]
        data = np.load(os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_multiple_alpha_with_bg',supsample,npy_file+'.npy'))
    info_line = data[0]
    for l in info_line:
        print l
    data = data[1:]
    poss = np.array(map(lambda x: x[0][0],data))
    if fdl_pos_start == 'all':
        min_pos_idx = 0
    else:
        min_pos_idx = get_nearest_idx_from_list(fdl_pos_start,poss)
    if fdl_pos_end == 'all':
        max_pos_idx = -1
    else:
        max_pos_idx = get_nearest_idx_from_list(fdl_pos_end,poss)
    data = data[min_pos_idx:max_pos_idx]
    poss = poss[min_pos_idx:max_pos_idx]
    lockins = np.nan_to_num(np.array(map(lambda x: x[0][1],data))*1e3)
    lockins_bg = np.nan_to_num(np.array(map(lambda x: x[0][2],data))*1e3)
    lockins = lockins - lockins_bg
    old_lockins = lockins
    poss_li,lockins,poss_li_out,lockins_out = remove_outlier(poss,lockins,5000)
    if len(lockins_out) > 0:
        print('Outlier(s) for lock-in removed:\n    poss = %s\n    lockins = %s'%(poss_li_out,lockins_out))
    specs = np.array(map(lambda x: x[1],data))
    specs_bg = np.array(map(lambda x: x[2],data))
    min_idx = get_nearest_idx_from_list(min_wl,hero_pixel_wavelengths)
    max_idx = get_nearest_idx_from_list(max_wl,hero_pixel_wavelengths)
    spec_sums = np.array(map(lambda x: np.sum(x[min_idx:max_idx]),specs))
    spec_sums_bg = np.array(map(lambda x: np.sum(x[min_idx:max_idx]),specs_bg))
    spec_sums = spec_sums - spec_sums_bg
    if lockin_as_pump_ref_order:
        spec_sums = spec_sums/(old_lockins**lockin_as_pump_ref_order)
    min_y = np.min((specs[0]-specs_bg[0])[min_idx-5:max_idx+5])
    max_y = np.max((specs[0]-specs_bg[0])[min_idx-5:max_idx+5])
    plt.ion()
    if plot_scan_fdl:
        _fig = plt.figure(npy_file,figsize=(15,9))
        figall = _fig.add_subplot(211)
        for i,spec in enumerate(specs):
            figall.plot(hero_pixel_wavelengths,spec-specs_bg[i])
        figall.set_xlim(min_wl-5,max_wl+5)
        figall.set_ylim(min_y-10,max_y+10)
        figall.set_ylabel('Intensity - bg, au')
        figall.set_xlabel('Wavelength, nm')
        figall.set_title('All specs')
        fig = _fig.add_subplot(212)
        fig2 = fig.twinx()
        line, = fig.plot(poss,spec_sums,'o',markersize=1,color='C0')
        line2, = fig2.plot(poss_li,lockins,'o',markersize=1,color='C1')
        _fig.suptitle(npy_file)
        fig.set_title('spec sum of %.1f to %.1f nm and lock-in amplitude'%(min_wl,max_wl))
        fig.set_xlabel('fdl pos, mm')
        fig.set_ylabel('spec sum - bg, au',color='C0')
        fig2.set_ylabel('lock-in amp - bg, mV',color='C1')
        fig2.format_coord = make_format(fig2, fig)
        plt.pause(.01)
        plt.get_current_fig_manager().window.showMaximized()
        plt.tight_layout(rect=[0, 0.00, 1, 0.95])
        plt.pause(.01)
        
        def on_xlims_change(a):
            fdl_pos_start,fdl_pos_end = a.get_xlim()
            min_pos_idx = get_nearest_idx_from_list(fdl_pos_start,poss)
            max_pos_idx = get_nearest_idx_from_list(fdl_pos_end,poss)
            figall.lines=[]
            plt.draw()
            for i in range(min_pos_idx,max_pos_idx):
                figall.plot(hero_pixel_wavelengths,specs[i]-specs_bg[i])
    
        fig.callbacks.connect('xlim_changed',on_xlims_change)
    
        return (_fig,fig,line,figall,fig2,line2)
#%%
def real_time_anal_scan_delayline_with_bg(npy_file,min_wl=770,max_wl=777.5,refresh_delay=10,lockin_as_pump_ref_order=2):
    min_idx = get_nearest_idx_from_list(min_wl,hero_pixel_wavelengths)
    max_idx = get_nearest_idx_from_list(max_wl,hero_pixel_wavelengths)
    _fig, fig, line, figall, fig2, line2 = anal_scan_delayline_with_bg(npy_file,min_wl=min_wl,max_wl=max_wl,lockin_as_pump_ref_order=lockin_as_pump_ref_order)
    prev_data_len = 0
    num = 0
    _data_path = os.path.join(r'D:/Nonlinear_setup/Experimental_data/scan_delayline_with_bg',npy_file+'.npy')
    if not os.path.isfile(_data_path):
        supsample = npy_file.split('_a')[0]
        _data_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_multiple_alpha_with_bg',supsample,npy_file+'.npy')
    global spec_sums
    while True:
        try:
            data = np.load(_data_path)
            info_line = data[0]
            data = data[1:]
        except:
            plt.pause(0.1)
            data = np.load(_data_path)
            info_line = data[0]
            data = data[1:]
        
        curr_data_len = len(data)
        if curr_data_len == prev_data_len:
            if num > 10:
                return
            plt.pause(refresh_delay)
            num += 1
            continue
        else:
            num = 0
        poss = np.array(map(lambda x: x[0][0],data))
        lockins = np.nan_to_num(np.array(map(lambda x: x[0][1],data))*1e3)
        lockins_bg = np.nan_to_num(np.array(map(lambda x: x[0][2],data))*1e3)
        lockins = lockins - lockins_bg
        old_lockins = lockins
        poss_li,lockins,poss_li_out,lockins_out = remove_outlier(poss,lockins,5000)
        specs = np.array(map(lambda x: x[1],data))
        specs_bg = np.array(map(lambda x: x[2],data))
        spec_sums = np.array(map(lambda x: np.sum(x[min_idx:max_idx]),specs))
        spec_sums_bg = np.array(map(lambda x: np.sum(x[min_idx:max_idx]),specs_bg))
        spec_sums = spec_sums - spec_sums_bg
        if lockin_as_pump_ref_order:
            spec_sums = spec_sums/(old_lockins**lockin_as_pump_ref_order)
        
        line.set_xdata(poss)
        line.set_ydata(spec_sums)
        fig.relim(True)
        fig.autoscale_view(True,True,True)
        
        line2.set_xdata(poss_li)
        line2.set_ydata(lockins)
        fig2.relim(True)
        fig2.autoscale_view(True,True,True)
        
        while curr_data_len > len(figall.lines):
            figall.plot(hero_pixel_wavelengths,specs[len(figall.lines)]-specs_bg[len(figall.lines)])
        
        plt.draw_all()
        prev_data_len = curr_data_len
        plt.pause(refresh_delay)
#%%

def anal_scan_delayline_multiple_Aang_with_bg(sample,get_temperature=True):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_multiple_Aang_with_bg',sample)
    npz_files = list(filter(lambda name: name.endswith('.npz'),os.listdir(main_path)))
    npy_files = list(filter(lambda name: name.endswith('.npy'),os.listdir(main_path)))
    def get_a_from_name(name):
        try:
            return float(name.split('_A')[1].split('.npz')[0])/10
        except:
            return float(name.split('_A')[1].split('.npy')[0])/10
    npz_files.sort(key=get_a_from_name)
    npy_files.sort(key=get_a_from_name)
    
    global ana_angs,all_Xs,all_Xerrs,all_thetas,all_theta_errs,all_Xs_sc,all_Xerrs_sc,all_thetas_sc,all_theta_errs_sc,data_to_origin
    ana_angs = np.array(list(map(get_a_from_name,npz_files)))
    all_Xs,all_Xerrs,all_thetas,all_theta_errs,all_Xs_sc,all_Xerrs_sc,all_thetas_sc,all_theta_errs_sc = [],[],[],[],[],[],[],[]
    for i,npz_file in enumerate(npz_files):
        global fdl_poss
        with np.load(os.path.join(main_path,npz_file)) as data:
            theta_errs = data['theta_errs']
            fdl_poss = data['fdl_poss']
            gains = data['gains']
            thetas = data['thetas']
            freqs = data['freqs']
            Xerrs = data['Xerrs']*1e6
            Xs = data['Xs']*1e6
            theta_errs_sc = data['theta_errs_sc']
            gains_sc = data['gains_sc']
            thetas_sc = data['thetas_sc']
            freqs_sc = data['freqs_sc']
            Xerrs_sc = data['Xerrs_sc']*1e6
            Xs_sc = data['Xs_sc']*1e6
        all_Xs.append(Xs)
        all_Xerrs.append(Xerrs)
        all_thetas.append(thetas)
        all_theta_errs.append(theta_errs)
        all_Xs_sc.append(Xs_sc)
        all_Xerrs_sc.append(Xerrs_sc)
        all_thetas_sc.append(thetas_sc)
        all_theta_errs_sc.append(theta_errs_sc)
        temp_name = 'temp_A%s.npy'%(npz_file.split('_A')[1].split('.npz')[0])
        try:
            if get_temperature:
                timestamps = np.load(os.path.join(main_path,npy_files[i]))
                start_time,end_time = timestamps
                curr_temp = get_temp_in_range(start_time,end_time)
                try:
                    old_temp = np.load(os.path.join(main_path,temp_name))
                    if len(old_temp) < len(curr_temp):
                        raise IOError
                except IOError:
                    np.save(os.path.join(main_path,temp_name),curr_temp)
        except:
            print('Unable to read TH+ logger temperature.')
            
        try:
            temps = get_temp_only(np.load(os.path.join(main_path,temp_name)) )
        except:
            temps = np.array([0])
        temp = np.mean(temps)
        temp_e = np.std(temps)
        
        curr_ana_ang = ana_angs[i]
        _fig = plt.figure('%s %.1fdeg'%(sample,curr_ana_ang))
        plt.get_current_fig_manager().window.showMaximized()
        fig = _fig.add_subplot(211)
        _fig.suptitle('%s\nA = %.1f$^o$, T = %.2f $\pm$ %.2f$^o$C'%(sample,curr_ana_ang,temp,temp_e))
        try:
            fig.errorbar(fdl_poss,Xs,yerr=Xerrs,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C0',label='without SC')
        except:
            fdl_poss = np.array(sorted(list(set(fdl_poss))))
            fig.errorbar(fdl_poss,Xs,yerr=Xerrs,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C0',label='without SC')
        fig.errorbar(fdl_poss,Xs_sc,yerr=Xerrs_sc,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C1',label='with SC')
        fig.set_ylabel('lock-in reading, $\mu$V',color='C0')
        fig.legend()
        plt.grid()
        fig2 = _fig.add_subplot(212)
        fig2.errorbar(fdl_poss,thetas,yerr=theta_errs,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C0',label='without SC')
        fig2.errorbar(fdl_poss,thetas_sc,yerr=theta_errs_sc,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C1',label='with SC')
        fig2.set_ylabel('lock-in angle, deg',color='C1')
        fig2.set_xlabel('delay line position, mm')
        fig2.axhline(0,color='black')
        fig2.format_coord = make_format(fig2, fig)
        plt.grid()
        plt.pause(.01)
        plt.tight_layout(rect=(0,0,1,0.95))
        
    all_Xs = np.array(all_Xs)
    all_Xerrs = np.array(all_Xerrs)
    all_thetas = np.array(all_thetas)
    all_theta_errs = np.array(all_theta_errs)
    data_to_origin = np.array(zip(all_Xs,all_Xerrs,all_thetas,all_theta_errs))


#%%
def anal_scan_delayline_with_lockin(sample,get_temperature=True):
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_lockin',sample)
    global theta_errs,fdl_poss,gains,thetas,freqs,Xerrs,Xs
    with np.load(os.path.join(main_path,'data.npz')) as data:
        theta_errs = data['theta_errs']
        fdl_poss = data['fdl_poss']
        gains = data['gains']
        thetas = data['thetas']
        freqs = data['freqs']
        Xerrs = data['Xerrs']*1e6
        Xs = data['Xs']*1e6
    try:
        if get_temperature:
            timestamps = np.load(os.path.join(main_path,'timestamps.npy'))
            start_time,end_time = timestamps
            curr_temp = get_temp_in_range(start_time,end_time)
            try:
                old_temp = np.load(os.path.join(main_path,'temp.npy'))
                if len(old_temp) < len(curr_temp):
                    raise IOError
            except IOError:
                np.save(os.path.join(main_path,'temp'),curr_temp)
    except:
        print('Unable to read TH+ logger temperature.')
        
    try:
        temps = get_temp_only(np.load(os.path.join(main_path,'temp.npy')) )
    except:
        temps = np.array([0])
    temp = np.mean(temps)
    temp_e = np.std(temps)
    
    _fig = plt.figure(sample)
    plt.get_current_fig_manager().window.showMaximized()
    fig = _fig.add_subplot(111)
    fig.errorbar(fdl_poss,Xs,yerr=Xerrs,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C0')
    fig.set_xlabel('delay line position, mm')
    fig.set_ylabel('lock-in reading, $\mu$V',color='C0')
    plt.grid()
    fig2 = fig.twinx()
    fig2.errorbar(fdl_poss,thetas,yerr=theta_errs,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C1')
    fig2.set_ylabel('lock-in angle, deg',color='C1')
    fig2.axhline(0,color='black')
    plt.suptitle('%s @ %.2f $\pm$ %.2f $^o$C'%(sample,temp,temp_e))
    fig2.format_coord = make_format(fig2, fig)
    plt.pause(.01)
    plt.tight_layout(rect=(0,0,1,0.95))

#%%
def anal_scan_delayline_with_lockin_XY(sample,get_temperature=True,errorbar=True):
    SCALE = 1e6 # multiply to change readings from V to uV
    
    # Load data from file data.npz
    main_path = os.path.join(r'D:\Nonlinear_setup\Experimental_data\scan_delayline_with_lockin',sample)
    with np.load(os.path.join(main_path,'data.npz')) as data:
        fdl_poss = data['fdl_poss']
        Xs = data['Xs']*SCALE
        Xerrs = data['Xerrs']*SCALE
        Ys = data['Ys']*SCALE
        Yerrs = data['Yerrs']*SCALE
        thetas = data['thetas']
        thetaerrs = data['thetaerrs']
    
    # get temperature readings
    try:
        if get_temperature:
            timestamps = np.load(os.path.join(main_path,'timestamps.npy'))
            start_time,end_time = timestamps
            curr_temp = get_temp_in_range(start_time,end_time)
            try:
                old_temp = np.load(os.path.join(main_path,'temp.npy'))
                if len(old_temp) < len(curr_temp):
                    raise IOError
            except IOError:
                np.save(os.path.join(main_path,'temp'),curr_temp)
    except:
        print('Unable to read TH+ logger temperature.')
    # if temperature is not readable, it will be just zero
    try:
        temps = get_temp_only(np.load(os.path.join(main_path,'temp.npy')) )
    except:
        temps = np.array([0])
    temp = np.mean(temps)
    temp_e = np.std(temps)
    
    # Plot stuff
    _fig = plt.figure(sample)
    plt.get_current_fig_manager().window.showMaximized()
    fig = _fig.add_subplot(111)
    # Xs
    if errorbar == True:
        fig.errorbar(fdl_poss,Xs,yerr=Xerrs,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C0')
    else:
        fig.scatter(fdl_poss,Xs,marker='o',linewidth=0,color='C0')
    fig.set_xlabel('delay line position, mm')
    fig.set_ylabel('lock-in reading, $\mu$V',color='C0')
    plt.grid()
    # Ys
    fig2 = fig.twinx()
    if errorbar == True:
        fig2.errorbar(fdl_poss,Ys,yerr=Yerrs,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C1')
    else:
        fig2.scatter(fdl_poss,Ys,marker='o',linewidth=0,alpha=0.5,color='C1')
    fig2.set_ylabel('lock-in quadrature, $\mu$V',color='C1')
    #fig2.spines['right'].set_color('C1')
    fig2.tick_params(axis='y',colors='C1')
    fig2.axhline(0,color='C1')
    # Thetas
    fig3 = fig.twinx()
    if errorbar == True:
        fig3.errorbar(fdl_poss,thetas,yerr=thetaerrs,capsize=2,marker='o',lw=0,ms=2,elinewidth=1,color='C2')
    else:
        fig3.scatter(fdl_poss,thetas,marker='o',linewidth=0,alpha=0.3,color='C2')
    fig3.set_ylabel('lock-in phase, deg',color='C2')
    fig3.tick_params(axis='y',colors='C2')
    fig3.axhline(0,color='C2')
    plt.suptitle('%s @ %.2f $\pm$ %.2f $^o$C'%(sample,temp,temp_e))
    fig2.format_coord = make_format(fig2, fig)
    plt.pause(.01)
    plt.tight_layout(rect=(0,0,1,0.95))

#%%
#_______________________________________________________________________
def multiple_petals_n7(ang,A0,A1,A2,A3,A4,A5,A6,A7,phi1,phi2,phi3,phi4,phi5,phi6,phi7):
    ans = A0 + A1*(np.cos(1*(ang-phi1)/180.*np.pi)) + A2*(np.cos(2*(ang-phi2)/180.*np.pi)) + A3*(np.cos(3*(ang-phi3)/180.*np.pi)) + A4*(np.cos(4*(ang-phi4)/180.*np.pi)) + A5*(np.cos(5*(ang-phi5)/180.*np.pi)) + A6*(np.cos(6*(ang-phi6)/180.*np.pi)) + A7*(np.cos(7*(ang-phi7)/180.*np.pi))
    return ans
def multiple_petals_n7_fitter(angs,intens):
    As = [np.max(intens)]*8
    phi = angs[intens.argmax()]
    if phi > 180:
        phi -= 180
    phis = [phi]*7
    p0 = (As + phis)
    popt, pcov = sp.optimize.curve_fit(multiple_petals_n7, angs, intens, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def six_petals(ang,A,phi,R):
    return A*(np.square(np.cos(3*(ang-phi)/180.*np.pi)) + R*np.square(np.sin(3*(ang-phi)/180.*np.pi)))
def _six_petals_fitter(angs,intens):
    A = np.max(intens)
    phi = angs[intens.argmax()]
    R = np.min(intens)/A
    p0 = (A, phi, R)
    popt, pcov = sp.optimize.curve_fit(six_petals, angs, intens, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def two_petals(ang,A,phi,R):
    return A*(np.square(np.cos((ang-phi)/180.*np.pi)) + R*np.square(np.sin((ang-phi)/180.*np.pi)))
def _two_petals_fitter(angs,intens):
    A = np.max(intens)
    phi = angs[intens.argmax()]
    if phi > 180:
        phi -= 180
    R = np.abs(np.min(intens)/A)
    p0 = (A, phi, R)
    popt, pcov = sp.optimize.curve_fit(two_petals, angs, intens, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def multiple_petals_n7_less6(ang,A0,A1,A2,A3,A4,A5,A7,phi1,phi2,phi3,phi4,phi5,phi7):
    ans = A0 + A1*(np.cos(1*(ang-phi1)/180.*np.pi)) + A2*(np.cos(2*(ang-phi2)/180.*np.pi)) + A3*(np.cos(3*(ang-phi3)/180.*np.pi)) + A4*(np.cos(4*(ang-phi4)/180.*np.pi)) + A5*(np.cos(5*(ang-phi5)/180.*np.pi)) + A7*(np.cos(7*(ang-phi7)/180.*np.pi))
    return ans
def correcting_petals(angs,intens):
    popt, perr = multiple_petals_n7_fitter(angs,intens)
    A0,A1,A2,A3,A4,A5,A6,A7,phi1,phi2,phi3,phi4,phi5,phi6,phi7 = popt
    correction = multiple_petals_n7_less6(angs,A0,A1,A2,A3,A4,A5,A7,phi1,phi2,phi3,phi4,phi5,phi7)
    corrected_intens = (np.array(intens)-correction + A0)
    return corrected_intens

def multiple_petals_n7_less2(ang,A0,A1,A3,A4,A5,A6,A7,phi1,phi3,phi4,phi5,phi6,phi7):
    ans = A0 + A1*(np.cos(1*(ang-phi1)/180.*np.pi)) + A6*(np.cos(2*(ang-phi6)/180.*np.pi)) + A3*(np.cos(3*(ang-phi3)/180.*np.pi)) + A4*(np.cos(4*(ang-phi4)/180.*np.pi)) + A5*(np.cos(5*(ang-phi5)/180.*np.pi)) + A7*(np.cos(7*(ang-phi7)/180.*np.pi))
    return ans
def correcting_2petals(angs,intens):
    popt, perr = multiple_petals_n7_fitter(angs,intens)
    A0,A1,A2,A3,A4,A5,A6,A7,phi1,phi2,phi3,phi4,phi5,phi6,phi7 = popt
    correction = multiple_petals_n7_less2(angs,A0,A1,A3,A4,A5,A6,A7,phi1,phi3,phi4,phi5,phi6,phi7)
    corrected_intens = np.array(np.array(intens)-correction + A0)
    return corrected_intens

#def put_angles_to_same_range(angs):
#    if len(angs) < 2:
#        return angs
#    else:
#        ans = []
#        for i,ang in enumerate(angs):
#            if i == 0:
#                while ang < 60:
#                    ang += 60
#                while ang > 60:
#                    ang -= 60
#                ans.append(ang)
#            else:
#                while ang < ans[-1] - 60*0.95:
#                    ang += 60
#                while ang > ans[-1] + 60*0.95:
#                    ang -= 60
#                ans.append(ang)
#        return ans

def get_hero_idx(wl):
    diff = list(np.abs(hero_pixel_wavelengths-wl))
    return diff.index(np.min(diff))

global _wl_chosen
_wl_chosen = [0]
def onclick(event):
    _wl_chosen[0] = event.xdata
    

def PMT_power_fW(V,V_e,gain,verbose=False):
    """
    Returns PMT reading and its uncertainty in fW.
    V = voltage reading from PMT, V
    V_e = uncertainty of V, V
    gain = gain voltage used in PMT, V
    
    calibration done on 30 Oct 2018:
    sens = 1956.20324 # pW/V @ 0.50V @ 780nm
    sens_e = 0.9281
    sens_coeff = np.sum(np.array([-1.13441E-5,0,0,0.0051,-0.02627,2.44576E-9,0.28712,-0.78918,0.86687,-0.3119])* #from fitting varying gain calibration curve PMT max reading vs gain
                        (gain**np.arange(10)) #fitted to polynomial of order 10
                        )/8.2093476429999823e-05 #normalized to gain at 0.50V
    """
    sens = 1956.20324 # pW/V @ 0.50V @ 780nm
    sens_e = 0.9281
    sens_coeff = np.sum(np.array([-1.13441E-5,0,0,0.0051,-0.02627,2.44576E-9,0.28712,-0.78918,0.86687,-0.3119])* #from fitting varying gain calibration curve PMT max reading vs gain
                        (gain**np.arange(10)) #fitted to polynomial of order 10
                        )/8.2093476429999823e-05 #normalized to gain at 0.50V
                           
    pW = sens/sens_coeff*V
    pW_e = np.sqrt(np.square(V*sens_e/sens_coeff) + np.square(sens/sens_coeff*V_e))
    
    fW = pW*1e3
    fW_e = pW_e*1e3
    
    if verbose:
        print(round_to_error_SI(fW,fW_e,'f','W'))
    return fW, fW_e

def round_to_n(x,n):
    return round(x, get_leading_order(x) + (n - 1))

def get_leading_order(x):
    return -int(np.floor(np.log10(x)))

def round_to_error(val,err):
    new_err = round_to_n(err,1)
    new_val = round(val,get_leading_order(new_err))
    return new_val,new_err

def round_to_error_SI(val,err,init_prefix=' ',unit=''):
    precesion = -1-get_leading_order(val)
    val *= si_prefix.si_prefix_scale(init_prefix)
    err *= si_prefix.si_prefix_scale(init_prefix)
    val,err = round_to_error(val,err)
    val_str = si_prefix.si_format(val,precesion)
    val_str,val_prefix = val_str.split(' ')
    err_str = u'%g'%(err/si_prefix.si_prefix_scale(val_prefix))
    dp = 0
    if u'.' in err_str:
        dp = int(len(err_str.split('.')[-1]))
    val_str = ('%%.%if'%(dp))%(float(val_str))
    return u'(%s \xb1 %s) %s%s'%(val_str,err_str,val_prefix,unit)

def process_and_save_img(path):
    image = np.load(path)
    subtract_background_rolling_ball(image, 150, light_background=False, use_paraboloid=True, do_presmooth=True)
    np.save(path[:-4]+'_noBG.npy', image)

def equalize_histogram_and_8bit(img):
    return ((img-np.min(img))/(np.max(img)-np.min(img)) * 255).astype(np.uint8)