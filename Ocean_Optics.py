    
import oceanoptics    
import time
import sys
import os
import select 
import numpy as np
import visa
from subprocess import Popen, list2cmdline
import itertools
import winsound
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy.optimize import leastsq
import matplotlib
import matplotlib.pyplot as plt
import Tkinter
import brewer2mpl
from ctypes import byref, pointer, c_long, c_float, c_bool
from textwrap import wrap
import seabreeze
seabreeze.use('pyseabreeze')
import seabreeze.spectrometers as oospec 
import csv


# ocean optics
time.sleep(1)
devices = oospec.list_devices()
time.sleep(1)

spec = oospec.Spectrometer(devices[0])

def ploting(xdata,ydata,int_time,title):  
    
           

    plt.clf()
    fig11=plt.figure(1)
          
    font = {'family':'Serif'}
    matplotlib.rc('font', **font)
    text={'color':'#252525'}
    matplotlib.rc('text', **text)
          
    fig11.patch.set_facecolor('white')
    #fig11.patch.set_alpha(0.5)
    fig11.patch.set_edgecolor('#efedf5')
    fig1=fig11.add_subplot(111)
    fig1.patch.set_facecolor('#fee8c8')
    fig1.patch.set_alpha(0.2)
    fig1.plot(xdata,ydata,linewidth=2,color='#238b45')#%d'%i    
    fig1.grid(True,color='#67000d',linewidth=1.5)
    titlefont={'family':'Serif','size':'20','color':'#800026','weight':'normal','style':'normal'}
    axisfont={'family':'Serif','size':'14','color':'#252525','weight':'normal','style':'normal'}
    fig1.legend(loc="center right",shadow=True,fancybox=True,title="Legend")
          
    plt.rc('legend',**{'fontsize':12})
    plt.title(title,**titlefont)
    plt.xlabel('Wavelengths nm',**axisfont)
    plt.ylabel('Intensities ( a. u.)',**axisfont)
    plt.subplots_adjust(wspace=0.4,hspace=0.4)    
    plt.draw()
    plt.pause(1e-6)
    
    
    
    
def ocean(int_time_spec):
    
   
     
    time.sleep(3)
    wavelength=spec.wavelengths()[100:]
    spec.integration_time_micros(int_time_spec) 
   
    i =0
    
    try:
        while True:
            
            intensities=spec.intensities()[100:]
            t = 'Max at %.4f (%.1f %%)'%calc_centroid(wavelength,intensities)
            ploting(wavelength,intensities,int_time_spec, t)
           
            
            
    except KeyboardInterrupt: # Type ctrl+c to stop !!!
        #Laser off        
        plt.close() 
        

def calc_centroid(wl,inten):
    max_index = inten.argmax()
    crop_range = 100
    bg = np.average(inten[-100:])
    inten_crop = inten[max_index-crop_range:max_index+crop_range] - bg
    wl_crop = wl[max_index-crop_range:max_index+crop_range]
    
    centroid = sum(inten_crop*wl_crop)/inten_crop.sum()
    
    return (centroid,inten.max()/655.35)
     
def get_n_spec(int_time_ms,n):
    spec.integration_time_micros(int_time_ms)
    
    all_spec = spec.intensities()    
    for i in range(n-1):
        all_spec += spec.intensities()
        
    result = all_spec / float(n)
    
    return result
    
def save_spec(filename,data):
    
    filepath='C:\Users\TOPTICA\Desktop\Nonlinear_setup\Experimental_data\spec_malus_law'
    _filename= os.path.join(filepath, filename+".txt")  
    
    np.savetxt(_filename,np.transpose(np.array((spec.wavelengths()[3:],data[3:]))))
    print 'Done.'

global _wavelength, _prev_int_time_us
_wavelength=spec.wavelengths()[100:]
_prev_int_time_us=[30000]
def get_centroid(int_time_us='previous'):
    if int_time_us != 'previous':
        spec.integration_time_micros(int_time_us) 
    else:
        int_time_us = _prev_int_time_us[0]
        
    intensities=spec.intensities()[100:]
    ans = calc_centroid(_wavelength,intensities)
    ans = (ans[0],ans[1],int_time_us,ans[1]*1000.0/int_time_us)
    
    while ans[1] > 90:
        int_time_us /= 1.4
        if int_time_us < 1000:
            int_time_us = 1000
            ans = get_centroid(int_time_us)
            print 'Signal for Ocean Optics is too strong. Please use ND filter.'
            break
        ans = get_centroid(int_time_us)
    while ans[1] < 20:
        int_time_us *= 1.4
        if int_time_us > 10e6:
            ans = get_centroid(int_time_us)
            print 'Signal for Ocean Optics is too weak.'
            break
        ans = get_centroid(int_time_us)
   
    _prev_int_time_us[0] = ans[2]
    return ans