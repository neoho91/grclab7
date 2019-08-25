# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:47:23 2019

@author: Neo
"""

import time
import os
import sys
import numpy as np
import scipy as sp
import scipy.interpolate
import threading
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))

try:
    from powermeter_analog import *
    #wrapping powermeter that it is using
    def pmp2_power():
        return pma_power()
    def pmp2_wl(wl=None):
        return pma_wl(wl)
    def pmp2_zero():
        return pma_zero()
except:
    print('Powermeter for polarimeter 2 not connected')
try:
    import polarimeter_rot_mount_2 as prm2
except:
    print('Polarimeter rotational mount 2 not connected')
    
from neo_common_code import *
QWP_pola_wl_Delta_2 = np.load(os.path.join(r'D:\Nonlinear_setup\Python_codes\polarimeter2_calib','QWP_pola_wl_Delta_08Aug19_1433.npy'))
QWP_pola_wl_Delta_2_fx = sp.interpolate.interp1d(QWP_pola_wl_Delta_2[0],QWP_pola_wl_Delta_2[1],fill_value='cubic')

polarimeter_temporary_data_2 = []

def polarimeter2_measure_slow(wl=742,angs=np.arange(0,360,15),
                             ave_num=10,timesleep=0.1,
                             take_bg=False,
                             live_plot=True,
                             plot=True,
                             is_calibrate=False,
                             verbose=True,
                             return_max_angle=False):
    """
    angs = QWP angles, iterable
    ave_num = number of measurement at each QWP angle to do average, default 10
    timesleep = time to wait after moving QWP in seconds, default 0.1
    """
    global data,data_dev
    data = []
    data_dev = []
    data_len = len(angs)
    prm2.home_rot_pol()
    pmp2_wl(wl)
    if not is_calibrate:
        Delta = QWP_pola_wl_Delta_2_fx(wl)
    if live_plot:
        _fig = plt.figure()
        fig = _fig.add_subplot(111,projection='polar')
    
    if take_bg:
        raw_input('Getting background. Cover laser and press ENTER.')
        print('Acquiring background...')
        pmp2_zero()
        plt.pause(1)
        raw_input('Done. Press ENTER to start taking data.')
    if verbose:
        print('Acquiring data...')
        prints('\n')
        prev_completed = ''
    for i,ang in enumerate(angs):
        prm2.set_pola_QWP_ang(ang)
        plt.pause(timesleep)
        curr_data = []
        for j in range(ave_num):
            curr_data.append(pmp2_power())
        data.append(np.mean(curr_data))
        data_dev.append(np.std(curr_data))
        
        if live_plot:
            fig.cla()
            if is_calibrate:
                signal_rotQfixA3_fitter(angs[:len(data)],data,plot=plot,fig=fig)
            else:
                signal_rotQfixA_fitter(angs[:len(data)],data,plot=plot,fig=fig,Delta=Delta)
            plt.pause(1e-6)
        
        if verbose:
            completed = 'QWP at %.1fdeg (%.2f percent)'%(ang,(i+1)*100./data_len)
            prints(completed,prev_completed)
            prev_completed = completed
    
    polarimeter_temporary_datum = [angs,data,data_dev]
    polarimeter_temporary_data_2.append(polarimeter_temporary_datum)
    
    if live_plot:
        plt.cla()
        if is_calibrate:
            return signal_rotQfixA3_fitter(angs,data,fig=fig,plot=plot,fit_curve_plot_full=True)
        else:
            if return_max_angle:
                popt,perr = signal_rotQfixA_fitter(angs,data,fig=fig,plot=plot,fit_curve_plot_full=True,Delta=Delta)
                A,alpha,gamma = popt
                angle_range = np.linspace(0,180,1000)
                polarimeter_fit = signal_rotQfixA(angle_range,alpha,gamma,QWP_pola_wl_Delta_2_fx(wl),1)
                max_ang = angle_range[np.argmax(polarimeter_fit)]
                return popt,perr,max_ang
            else:
                return signal_rotQfixA_fitter(angs,data,fig=fig,plot=plot,fit_curve_plot_full=True,Delta=Delta)
    else:
        if is_calibrate:
            return signal_rotQfixA3_fitter(angs,data,plot=plot,fit_curve_plot_full=True)
        else:
            if return_max_angle:
                popt,perr = signal_rotQfixA_fitter(angs,data,plot=plot,fit_curve_plot_full=True,Delta=Delta)
                A,alpha,gamma = popt
                angle_range = np.linspace(0,180,1000)
                polarimeter_fit = signal_rotQfixA(angle_range,alpha,gamma,QWP_pola_wl_Delta_2_fx(wl),1)
                max_ang = angle_range[np.argmax(polarimeter_fit)]
                return popt,perr,max_ang
            else:
                return signal_rotQfixA_fitter(angs,data,plot=plot,fit_curve_plot_full=True,Delta=Delta)

def polarimeter2_measure(init_angle=-360,final_angle=350,timesleep=0.01): #to be fixed
    home_rot_pol()
    prm2.set_pola_QWP_ang(init_angle)
    global data
    data=[]
    
    def rotating_loop():
        prm2.set_pola_QWP_ang(final_angle)
    rotating_th = threading.Thread(target=rotating_loop)
    
    def acquire_data_loop():
        while rotating_th.isAlive():
            data.append(pmp2_power())
            time.sleep(timesleep)
    acquire_data_th=threading.Thread(target=acquire_data_loop)
    
    rotating_th.start()
    acquire_data_th.start()
    
    rotating_th.join()
    acquire_data_th.join()
    
    angs = np.linspace(-360,350,len(data))
    polarimeter_temporary_datum = [angs,data]
    return signal_rotQfixA_fitter(angs,data)

def calib_polarimeter2_QWP_Delta():
    wls = np.arange(700,789,1)
    Deltas = []
    Deltas_err = []
    for i,wl in enumerate(wls):
        go_to_wl(wl,2,visualize=0,move_fdl=0)
        pmp2_wl(wl)
        plt.pause(1)
        curr_Delta = []
        for j in range(3):
            popt,perr = polarimeter2_measure_slow(angs = np.arange(0,360,15), timesleep = 0.1, ave_num = 3, take_bg=0, live_plot=0, is_calibrate=True)
            curr_Delta.append(popt[2])
            plt.pause(1e-6)
        plt.close('all')
        Deltas.append(np.mean(curr_Delta))
        Deltas_err.append(np.std(curr_Delta))
        plt.errorbar(wls[:len(Deltas)],Deltas,yerr=Deltas_err,marker='o',capsize=2)
    
    data_to_save = np.array([wls,Deltas,Deltas_err])
    data_path = os.path.join(r'D:\Nonlinear_setup\Python_codes\polarimeter2_calib','QWP_pola_wl_Delta_'+create_date_str()+'.npy')
    np.save(data_path,data_to_save)
    
#%%
def signal_rotQfixA(theta,alpha,gamma,Delta=0,A=1):
    theta = theta/180.*np.pi
    alpha = alpha/180.*np.pi
    gamma = gamma/180.*np.pi
    return A*(4 + 2*np.cos(2*alpha)*np.cos(2*gamma)*(1-np.sin(2*np.pi*Delta)) + 2*np.cos(2*gamma)*np.cos(2*(alpha-2*theta))*(np.sin(2*np.pi*Delta)+1) + 4*np.sin(2*gamma)*np.sin(2*theta)*np.cos(2*np.pi*Delta))/8

def signal_rotQfixA_fitter(thetas,data,Delta=0,p0=None,fig=None,plot=True,fit_curve_plot_full=False):
    """
    parametrized in alpha and gamma.
    """
    if p0 == None:
        p0 = (np.max(data),0.,0.)
    def _fx(theta,A,alpha,gamma):
        return signal_rotQfixA(theta,alpha,gamma,Delta=Delta,A=A)
    
    if fig == None and plot:
        _fig = plt.figure()
        fig = _fig.add_subplot(111,projection='polar')
    try:
        popt, pcov = sp.optimize.curve_fit(_fx, thetas, data, p0)
        perr = np.sqrt(np.diag(pcov))
        if plot:
            fig.plot(thetas/180.*np.pi,data,'o')
            if fit_curve_plot_full:
                thetas_fit = np.linspace(0,360,100)
            else:
                thetas_fit = np.linspace(min(thetas),max(thetas),100)
            fig.plot(thetas_fit/180.*np.pi,_fx(thetas_fit,*popt))
            fig.set_title(r'A = %.3e, $\alpha$ = %.2f deg, $\gamma$ = %.2f deg, $\Delta$ = %.2f wave'%(popt[0],popt[1],popt[2],Delta))
        return popt, perr
    except:
        if plot:
            fig.plot(thetas/180.*np.pi,data,'o')
        return


def signal_rotQfixA2(theta,phase,phase_axis,alpha,A=1):
    theta = theta/180.*np.pi
    phase = phase/180.*np.pi
    phase_axis = phase_axis/180.*np.pi
    alpha = alpha/180.*np.pi
    return A/16*(8+2*np.cos(2*alpha)+np.cos(2*alpha-phase)+np.cos(2*alpha+phase)+2*np.cos(2*alpha-4*theta)+np.cos(2*alpha-phase-4*theta)+np.cos(2*alpha+phase-4*theta)+2*np.cos(2*alpha-4*phase_axis)-np.cos(2*alpha-phase-4*phase_axis)-np.cos(2*alpha+phase-4*phase_axis)-2*(-1+np.cos(phase))*np.cos(2*alpha+4*theta-4*phase_axis)+8*np.sin(phase)*np.sin(2*theta)*np.sin(2*alpha-2*phase_axis))

def signal_rotQfixA2_fitter(thetas,data,alpha=0,Delta=0,p0=None,fig=None,plot=True,fit_curve_plot_full=False):
    """
    parametrized in phase shift and fast axis angle.
    """
    if p0 == None:
        p0 = (np.max(data),1.,1.)
    def _fx(theta,A,phase,phase_axis):
        return signal_rotQfixA2(theta,phase,phase_axis,alpha,A)
    
    if fig == None and plot:
        _fig = plt.figure()
        fig = _fig.add_subplot(111,projection='polar')
    try:
        popt, pcov = sp.optimize.curve_fit(_fx, thetas, data, p0)
        perr = np.sqrt(np.diag(pcov))
        if plot:
            fig.plot(thetas/180.*np.pi,data,'o')
            if fit_curve_plot_full:
                thetas_fit = np.linspace(0,360,100)
            else:
                thetas_fit = np.linspace(min(thetas),max(thetas),100)
            fig.plot(thetas_fit/180.*np.pi,_fx(thetas_fit,*popt))
            fig.set_title(r'A = %.3e, phase = %.2f deg, phase_axis = %.2f deg, $\alpha$ = %.1f deg'%(popt[0],popt[1],popt[2],alpha))
        return popt, perr
    except:
        if plot:
            fig.plot(thetas/180.*np.pi,data,'o')
        return


def signal_rotQfixA3(theta,phi,Delta,A=1):
    theta = theta/180.*np.pi
    phi = phi/180.*np.pi
    return A/4*(3 + np.cos(4*(phi - theta)) -  2*np.sin(2*Delta*np.pi)*np.square(np.sin(2*(phi - theta))))

def signal_rotQfixA3_fitter(thetas,data,p0=None,fig=None,plot=True,fit_curve_plot_full=False):
    """
    parametrized in max angle and Delta, used for calibration. Input pol is linear, parallel to analyzer.
    """
    if p0 == None:
        p0 = (np.max(data),thetas[get_nearest_idx_from_list(np.max(data),data)],0.)
    def _fx(theta,A,phi,Delta):
        return signal_rotQfixA3(theta,phi,Delta,A)
    
    if fig == None and plot:
        _fig = plt.figure()
        fig = _fig.add_subplot(111,projection='polar')
    try:
        popt, pcov = sp.optimize.curve_fit(_fx, thetas, data, p0)
        perr = np.sqrt(np.diag(pcov))
        if plot:
            fig.plot(thetas/180.*np.pi,data,'o')
            if fit_curve_plot_full:
                thetas_fit = np.linspace(0,360,100)
            else:
                thetas_fit = np.linspace(min(thetas),max(thetas),100)
            fig.plot(thetas_fit/180.*np.pi,_fx(thetas_fit,*popt))
            fig.set_title(r'A = %.3e, $\phi$ = %.2f deg, $\Delta$ = %.4f'%(popt[0],popt[1],popt[2]))
        return popt, perr
    except:
        if plot:
            fig.plot(thetas/180.*np.pi,data,'o')
        return
#%%
def get_gamma_alpha(thetas,data,Delta=0,verbose=False):
    fftdata = np.fft.fft(data)/len(data)
    dc = np.real(fftdata[0])
    twotheta = -(fftdata[2]-fftdata[-2])
    fourtheta_1 = fftdata[4]
    fourtheta_2 = fftdata[-4]
    
    gamma1 = np.arcsin(2*np.imag(twotheta)/np.cos(2*np.pi*Delta))/2
#    gamma2 = np.arcsin(np.imag(twotheta)/(dc-(np.real(fourtheta_1)+np.real(fourtheta_2))))/2
#    gamma = np.mean([gamma1,gamma2])
    gamma = gamma1
    
    alpha1 = np.angle(fourtheta_2)/2
    alpha2 = -np.angle(fourtheta_1)/2
    alpha = np.mean([alpha1,alpha2])
#    alpha = np.sign(alpha1)*np.arccos((4*dc-2)/np.sin(2*abs(gamma)))/2
#    print(gamma1/np.pi*180)
#    print(gamma2/np.pi*180)
    
#    print(alpha1/np.pi*180)
#    print(alpha2/np.pi*180)
    
#    print('%.5f%% deviation from model.'%(((dc-(np.real(fourtheta_1)+np.real(fourtheta_2)))-0.5)/0.5*100))
    if verbose:
        print('alpha = %.2f deg, gamma = %.2f deg'%(alpha/np.pi*180,gamma/np.pi*180))
    return alpha/np.pi*180,gamma/np.pi*180

def separate_thetas(thetas,data,period=360):
    """
    expected input are over 1 period longer.
    """
    theta_1 = thetas[0]
    theta_final = theta_1 + period
    theta_final_idx = get_nearest_idx_from_list(theta_final,thetas)
    chopped_thetas = []
    chopped_data = []
    while len(thetas) >= theta_final_idx:
        curr_thetas = thetas[:theta_final_idx]
        curr_data = data[:theta_final_idx]
        thetas = thetas[theta_final_idx:]
        data = data[theta_final_idx:]
        chopped_thetas.append(curr_thetas)
        chopped_data.append(curr_data)
    curr_theta,curr_data,flag = fill_in_thetas(thetas,data,period=period)
    if flag:
        chopped_thetas.append(curr_thetas)
        chopped_data.append(curr_data)
    return np.array(chopped_thetas),np.array(chopped_data)

def fill_in_thetas(thetas,data,period=360):
    """
    expected input are less than 1 period, but more than 0.5 period.
    """
    theta_1 = thetas[0]
    theta_incre = np.mean(np.diff(thetas))
    theta_final = theta_1 + period
    thetas_new = np.arange(theta_1,theta_final,theta_incre)
    if len(thetas_new) > 2*len(thetas):
        return thetas,data,False
    return thetas_new,np.interp(thetas_new,thetas,data,period=period/2),True