# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:45:38 2017

@author: Neo
"""
import visa
import time
import numpy as np
import scipy as sp
import os
import sys
import threading
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))
import interference_filter_calc as calc

try:
    from powermeter_analog import *
except:
    print "powermeter analog not connected"
try:
    from powermeter_digital import *
except:
    print "powermeter digital not connected"
try:
    from powermeter_usb_interface import *
except:
    print "powermeter usb interface not connected"
try:
    from WMP_fine_dl import *
except:
    print("Fine delay line not connected")

rm=visa.ResourceManager()
rot1 = rm.open_resource(u'ASRL6::INSTR')
rot2 = rm.open_resource(u'ASRL7::INSTR')
rot1.clear()
rot2.clear()
rot1.timeout=5000
rot2.timeout=5000
_deg_conv = 360/262144.
AOI_offset1 = 27.6
AOI_offset2 = 27.6

def _ask_rot(q,rot=1):
#    slider.clear()
    if rot==1:
        _ans = rot1.ask(q)
    else:
        _ans = rot2.ask(q)
    try:
        return _ans[3:-2]
    except:
        if '001F' in _ans:
            return 0
        return 1

def _conv_to_deg(h):
    ans = int(h,16)
    if ans > 262144:
        ans = -4294967296+ans
    return ans*_deg_conv

def _conv_to_hex(d):
    ans = d/_deg_conv
    if d < 0:
        ans=0XFFFFFFFF+ans+2
    ans = '%08X'%ans
    return ans

def home_rot(rot):
    ans = _ask_rot(u'0ho0',rot)
    return _conv_to_deg(ans)

def get_pos_rot(rot):
    pos = _ask_rot(u'0gp',rot)
    try:
        return _conv_to_deg(pos)
    except:
        time.sleep(1e-3)
        return get_pos_rot(rot)

def get_AOI_rot1():
    return get_pos_rot(1)+AOI_offset1

def get_AOI_rot2():
    return get_pos_rot(2)+AOI_offset2

def set_AOI_rot1(ang,aang=[None]):
    ans = move_arot(ang-AOI_offset1,1)+AOI_offset1
    aang[0]=ans
    return ans

def set_AOI_rot2(ang,aang=[None]):
    ans = move_arot(ang-AOI_offset2,2)+AOI_offset2
    aang[0]=ans
    return ans

def move_arot(pos,rot):
    pos = _ask_rot(u'0ma%s'%_conv_to_hex(pos),rot)
    return _conv_to_deg(pos)

def set_rot_step(step):
    h_step = _conv_to_hex(step)
    a1 = _ask_rot(u'0sj%s'%h_step,1)
    a2 = _ask_rot(u'0sj%s'%h_step,2)
    j1 = _ask_rot(u'0gj',1)
    j2 = _ask_rot(u'0gj',2)
    return 'jogstep = %f and %f'%(_conv_to_deg(j1),_conv_to_deg(j2))
    
def move_rotf(rot):
    pos = _ask_rot(u'0fw',rot)
    return _conv_to_deg(pos)

def move_rotb(rot):
    pos = _ask_rot(u'0bw',rot)
    return _conv_to_deg(pos)

def _go_to_wl(wl,df=4):
    """
    wl = wavelength of output light, in nm
    df = FWHM of output light, in THz
    """
    AOI1, AOI2 = calc.get_angles(wl,df=df)
    aAOI1,aAOI2 = [None],[None]
    th1 = threading.Thread(target=set_AOI_rot1,args=(AOI1,aAOI1))
    th2 = threading.Thread(target=set_AOI_rot2,args=(-AOI2,aAOI2))
    th1.start()
    th2.start()
    th1.join()
    th2.join()       
    return (aAOI1[0],aAOI2[0])

def versachrome1_to_wl(central_wl,visualize=True):
    if visualize:
        fwhm = calc.fwhm_from_wlc1(central_wl)
        d = calc.slant_from_wlc1(central_wl)
        h = calc.T_peak_from_wlc1(central_wl)
        wls = np.linspace(660,810,1000)
        Ts = calc.trapz_curve(wls,h,central_wl,d,fwhm)
        _fig = plt.figure('Versachrome 1 Transmission')
        _fig.clf()
        fig = _fig.add_subplot(111)
        fig.plot(wls,Ts,lw=2,color='black')
        fig.plot(central_wl+(fwhm-d)/2.,h,'o',color='C0')
        fig.plot(central_wl-(fwhm-d)/2.,h,'o',color='C0')
        fig.plot(central_wl+fwhm/2.,h/2.,'o',color='C1')
        fig.plot(central_wl-fwhm/2.,h/2.,'o',color='C1')
        fig.plot(central_wl+(fwhm+d)/2.,0.,'o',color='C2')
        fig.plot(central_wl-(fwhm+d)/2.,0,'o',color='C2')
        fig.set_title('Versachrome 1 Transmission')
        fig.set_xlabel('wavelength, nm')
        fig.set_ylabel('transmission, au')
        fig.set_xlim(660,810)
        fig.set_ylim(-0.01,1.1)
        fig.text(central_wl,0.7,'$\lambda_{central}$ = %.2f nm\nFW @ max = %.2f nm\nFWHM = %.2f nm\nFW @ min = %.2f nm'%(central_wl,fwhm-d,fwhm,fwhm+d),
                 horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+fwhm/2.,h/2.-0.03,'%.2f'%(central_wl+fwhm/2.),
                 horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-fwhm/2.,h/2.-0.03,'%.2f'%(central_wl-fwhm/2.),
                 horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+(fwhm+d)/2.,0.03,'%.2f'%(central_wl+(fwhm+d)/2.),
                 horizontalalignment='left',verticalalignment='bottom',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-(fwhm+d)/2.,0.03,'%.2f'%(central_wl-(fwhm+d)/2.),
                 horizontalalignment='right',verticalalignment='bottom',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+(fwhm-d)/2.,h-0.03,'%.2f'%(central_wl+(fwhm-d)/2.),
                 horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-(fwhm-d)/2.,h-0.03,'%.2f'%(central_wl-(fwhm-d)/2.),
                 horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.grid()
#        plt.pause(1e-6)
        
    ang = calc.angle_from_wlc1(central_wl)
    set_AOI_rot1(ang)

def versachrome2_to_wl(central_wl,visualize=True):
    if visualize:
        fwhm = calc.fwhm_from_wlc2(central_wl)
        d = calc.slant_from_wlc2(central_wl)
        h = calc.T_peak_from_wlc2(central_wl)
        wls = np.linspace(660,810,1000)
        Ts = calc.trapz_curve(wls,h,central_wl,d,fwhm)
        _fig = plt.figure('Versachrome 2 Transmission')
        _fig.clf()
        fig = _fig.add_subplot(111)
        fig.plot(wls,Ts,lw=2,color='black')
        fig.plot(central_wl+(fwhm-d)/2.,h,'o',color='C0')
        fig.plot(central_wl-(fwhm-d)/2.,h,'o',color='C0')
        fig.plot(central_wl+fwhm/2.,h/2.,'o',color='C1')
        fig.plot(central_wl-fwhm/2.,h/2.,'o',color='C1')
        fig.plot(central_wl+(fwhm+d)/2.,0.,'o',color='C2')
        fig.plot(central_wl-(fwhm+d)/2.,0,'o',color='C2')
        fig.set_title('Versachrome 2 Transmission')
        fig.set_xlabel('wavelength, nm')
        fig.set_ylabel('transmission, au')
        fig.set_xlim(660,810)
        fig.set_ylim(-0.01,1.1)
        fig.text(central_wl,0.7,'$\lambda_{central}$ = %.2f nm\nFW @ max = %.2f nm\nFWHM = %.2f nm\nFW @ min = %.2f nm'%(central_wl,fwhm-d,fwhm,fwhm+d),
                 horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+fwhm/2.,h/2.-0.03,'%.2f'%(central_wl+fwhm/2.),
                 horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-fwhm/2.,h/2.-0.03,'%.2f'%(central_wl-fwhm/2.),
                 horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+(fwhm+d)/2.,0.03,'%.2f'%(central_wl+(fwhm+d)/2.),
                 horizontalalignment='left',verticalalignment='bottom',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-(fwhm+d)/2.,0.03,'%.2f'%(central_wl-(fwhm+d)/2.),
                 horizontalalignment='right',verticalalignment='bottom',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+(fwhm-d)/2.,h-0.03,'%.2f'%(central_wl+(fwhm-d)/2.),
                 horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-(fwhm-d)/2.,h-0.03,'%.2f'%(central_wl-(fwhm-d)/2.),
                 horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.grid()
#        plt.pause(1e-6)
        
    ang = calc.angle_from_wlc2(central_wl)
    set_AOI_rot2(ang)
        
    
def go_to_wl(wl,dwl=2,visualize=True,move_fdl=False,fdl_offset=-0.02887):
    """
    wl = wavelength of output light, in nm
    dwl = FWHM of output light, in nm
    visualize = True to see the calibrated transmission
    """
    AOI1, AOI2 = calc.get_angles(wl,dwl=dwl)
    if visualize:
        wlc1 = calc.wlc_from_angle1(AOI1)
        fwhm1 = calc.fwhm_from_angle1(AOI1)
        d1 = calc.slant_from_angle1(AOI1)
        h1 = calc.T_peak_from_angle1(AOI1)
        
        wlc2 = calc.wlc_from_angle2(AOI2)
        fwhm2 = calc.fwhm_from_angle2(AOI2)
        d2 = calc.slant_from_angle2(AOI2)
        h2 = calc.T_peak_from_angle2(AOI2)
        
        wls = np.linspace(660,810,1000)
        T1 = calc.trapz_curve(wls,h1,wlc1,d1,fwhm1)
        T2 = calc.trapz_curve(wls,h2,wlc2,d2,fwhm2)
        Ts = T1*T2
        
        _fig = plt.figure('Combined 2 Versachromes Transmission')
        _fig.clf()
        fig = _fig.add_subplot(111)
        fig.plot(wls,Ts,lw=2,color='black')
        fig.plot(wls,T1,color='grey',ls='--')
        fig.plot(wls,T2,color='grey',ls=':')
        
        popt, perr = calc.fit_trapz_curve(wls,Ts)
        h, central_wl, d, fwhm = popt
        fig.plot(central_wl+(fwhm-d)/2.,h,'o',color='C0')
        fig.plot(central_wl-(fwhm-d)/2.,h,'o',color='C0')
        fig.plot(central_wl+fwhm/2.,h/2.,'o',color='C1')
        fig.plot(central_wl-fwhm/2.,h/2.,'o',color='C1')
        fig.plot(central_wl+(fwhm+d)/2.,0.,'o',color='C2')
        fig.plot(central_wl-(fwhm+d)/2.,0,'o',color='C2')
        fig.set_title('Combined 2 Versachromes Transmission')
        fig.set_xlabel('wavelength, nm')
        fig.set_ylabel('transmission, au')
        fig.set_xlim(660,810)
        fig.set_ylim(-0.01,1.1)
        fig.text(central_wl,0.6,'$\lambda_{central}$ = %.2f nm\nFW @ max = %.2f nm\nFWHM = %.2f nm\nFW @ min = %.2f nm'%(central_wl,fwhm-d,fwhm,fwhm+d),
                 horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+fwhm/2.+2,h/2.-0.03,'%.2f'%(central_wl+fwhm/2.),
                 horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-fwhm/2.-2,h/2.-0.03,'%.2f'%(central_wl-fwhm/2.),
                 horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+(fwhm+d)/2.+2,0.03,'%.2f'%(central_wl+(fwhm+d)/2.),
                 horizontalalignment='left',verticalalignment='bottom',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-(fwhm+d)/2.-2,0.03,'%.2f'%(central_wl-(fwhm+d)/2.),
                 horizontalalignment='right',verticalalignment='bottom',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl+(fwhm-d)/2.+2,h-0.03,'%.2f'%(central_wl+(fwhm-d)/2.),
                 horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.text(central_wl-(fwhm-d)/2.-2,h-0.03,'%.2f'%(central_wl-(fwhm-d)/2.),
                 horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='white',alpha=0.8))
        fig.grid()

    aAOI1,aAOI2 = [None],[None]
    th1 = threading.Thread(target=set_AOI_rot1,args=(AOI1,aAOI1))
    th2 = threading.Thread(target=set_AOI_rot2,args=(AOI2,aAOI2))
    th1.start()
    th2.start()
    th1.join()
    th2.join()
    if move_fdl:
        fdl_pos = wl_fdl_pos(wl,fdl_offset)
        print('Moving to fine delay line position ' + str(fdl_pos) + ' mm')
        move_fdl_abs(fdl_pos)
    return (aAOI1[0],aAOI2[0])

def wl_fdl_pos(wl,fdl_offset=-0.02887):
    with np.load(os.path.join(r'D:\Nonlinear_setup\Python_codes\FilterCalib',
                                  r'SFG_SC_vs_fdl_pos.npz')) as data:
        wls = data['wls']
        fdl_poss = data['fdl_poss']
        f_wl_to_fdl_pos = sp.interpolate.interp1d(wls,fdl_poss,kind='cubic')
        fdl_pos = f_wl_to_fdl_pos(wl) + fdl_offset
    return fdl_pos


def open_filters():
    def move_filter1():
        set_AOI_rot1(90)
    def move_filter2():
        set_AOI_rot2(90)
    th1 = threading.Thread(target=move_filter1)
    th2 = threading.Thread(target=move_filter2)
    th1.start()
    th2.start()
    th1.join()
    th2.join()

global wavelength_angles
wavelength_angles = []

def save_wl_ang_ang(wl,fwhm):
    pos1 = get_pos_rot(1)
    pos2 = get_pos_rot(2)
    curr_data = [wl,fwhm,pos1,pos2]
    wavelength_angles.append(curr_data)
    np.save(os.path.join(r'D:\WMP_setup\Python_codes','wavelength_angles.npy'),np.array(wavelength_angles))
    print 'Saved %s'%str(curr_data)

'''
###
# Code to calibrate the delay line position for SFG with different SC wavelengths.
# First run delayline scans named as below ("SFG_LNB_run001_wl%i").
# Then run the below code modified appropriately.
# File saved as npz file containing wavelengths as wls and corresponding delayline positions for SFG peaks as fdl_poss.
###
wls = np.arange(789,699.9,-1)
all_fdl_poss = []
all_SFG_data = []

# analyse delayline scan data
for i,wl in enumerate(wls):
    central_wl=1./(1./wl+1./1550)
    anal_scan_delayline_with_andor("SFG_LNB_run001_wl%i"%wl,min_wl=400,max_wl=600,fit=False)
    all_fdl_poss.append(copy.copy(np.array(poss)))
    all_SFG_data.append(copy.copy(np.array(spec_sums)))
    plt.close('all')
    print "Done %i nm (%.2f percent completed)\n"%(wl,100.*(i+1)/len(wls))
all_fdl_poss=np.array(all_fdl_poss)
all_SFG_data=np.array(all_SFG_data)

# format data to copy to Origin
data_to_origin = []
for i in range(len(wls)):
    data_to_origin.append(all_fdl_poss[i])
    data_to_origin.append(all_SFG_data[i])
data_to_origin = np.array(data_to_origin)
# cropped data
data_to_origin2=[]
for i in range(len(wls)):
    curr_max_idx = all_SFG_data[i].argmax()
    data_to_origin2.append(all_fdl_poss[i][curr_max_idx-19:curr_max_idx+20])
    data_to_origin2.append(all_SFG_data[i][curr_max_idx-29:curr_max_idx+20]/np.max(all_SFG_data[i]))
data_to_origin2 = np.array(data_to_origin2)
all_fdl_poss2 = []
all_SFG_data2 = []
for i in range(len(wls)):
    curr_max_idx = all_SFG_data[i].argmax()
    all_fdl_poss2.append(all_fdl_poss[i][curr_max_idx-19:curr_max_idx+20])
    all_SFG_data2.append(all_SFG_data[i][curr_max_idx-19:curr_max_idx+20]/np.max(all_SFG_data[i]))
all_fdl_poss2=np.array(all_fdl_poss2)
all_SFG_data2=np.array(all_SFG_data2)
fitted_data = []

# fit data to obtain position of peak, plot and save
for i in range(len(wls)):
    popt,perr = gaussian_fitter(all_fdl_poss2[i],all_SFG_data2[i],plot=0,can_be_null=0)
    fitted_data.append(popt[0])
fitted_data=np.array(fitted_data)
plt.figure()
plt.plot(wls,fitted_data,'o')
np.savez(r'D:\Nonlinear_setup\Python_codes\FilterCalib\SFG_SC_vs_fdl_pos',wls=wls,fdl_poss=fitted_data)
'''

rot1_info = _ask_rot(u'0in',rot1)
rot2_info = _ask_rot(u'0in',rot2)
print 'Filter rotational stages (%s and %s) online.'%(rot1_info,rot2_info)