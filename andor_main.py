# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:47:56 2019

@author: Donna
"""

import visa
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PyAndor import ShamRockController
from PyAndor import AndorCamera

#initialize camera
andor_cam = AndorCamera()
andor_cam.Initialize(acquisitionMode="single")

#initialize spectrometer
spec = ShamRockController()
spec.Initialize()
andor_spec = spec.Connect()

#Acquire and set camera pixel width and number of pixels
PixelSize = andor_cam.GetPixelSize()
PixelNumber = andor_cam.GetDetector()
andor_spec.SetPixelWidth(PixelSize[0])
andor_spec.SetNumberPixels(PixelNumber[0])

#set cooler temperatures, cooler ON/OFF
def cooler_on(temp=-65):
    andor_cam.CoolerON()
    andor_cam.SetFanMode("full")
    andor_cam.SetTemperature(temp)
    
cooler_on()

def cooler_temperature():
    return 'Cooler temperature is %.2f degC'%(andor_cam.GetTemperature())    
    
def cooler_off():
    andor_cam.CoolerOFF()

#permanent camera settings, not to be adjusted, for spectra acquisition
andor_cam.SetReadMode("FVB")
andor_cam.SetAcquisitionMode("single")
andor_cam.SetOutputAmplifier("EMCCD")
andor_cam.SetVSSpeed(0)

#definitions for camera control
def get_exposure_time():
    try:
        exp_time = andor_cam.GetAcquisitionTimings()
        return 'Exposure time is %.2f miliseconds'%(exp_time[0]*1000.0)
    except:
        return 'Error, pls try again'

def set_exposure_time(time):
    if time < 32.24:
        try:
            andor_cam.SetExposureTime(time/1000.0)
            return 'Out of range value, exposure time set to minimum. '+get_exposure_time()
        except:
            return 'invalid exposure time input'
    elif time > 32767998.05:
        try:
            andor_cam.SetExposureTime(time/1000.0)
            return 'Out of range value, exposure time set to maximum. '+get_exposure_time()
        except:
            return 'invalid exposure time input'
    else:
        try:
            andor_cam.SetExposureTime(time/1000.0)
            return get_exposure_time()
        except:
            return 'invalid exposure time input'
        
andor_cam.SetVSSpeed(0)
            
#definitions for spectrometer control
def get_wl():
    try:
        return 'Set wavelength is %.2f nm'%andor_spec.GetWavelength()
    except: 
        return 'Error, pls try again'

def set_wl(wl):
    try:
        andor_spec.SetWavelength(wl)
        return get_wl()
    except:
        return 'invalid wavelength input'
    
def get_grating():
    try:
        g = andor_spec.GetGrating()
        _grating = andor_spec.GetGratingInfo(g)
        return 'Set grating #%s: %s lines/mm, %s blaze wavelength'%(g,_grating[0],_grating[1])
    except:
        print 'Error, pls try again'

def set_grating(g):
    try:
        andor_spec.SetGrating(g)
        return get_grating()
    except:
        print 'invalid grating input'

def get_wl_vector():
    try:
        wl_vector = andor_spec.GetCalibration()
        return wl_vector
    except:
        print 'Error, pls try again'

#spectrum acquisition
def get_spectrum():
    andor_cam.StartAcquisition()
    andor_cam.WaitForAcquisition()
    spectrum = andor_cam.GetMostRecentImage()
    return spectrum
    
def measure_spectrum(wl=473,grating=3,exp_time=10):
    set_wl(wl)
    set_grating(grating)
    set_exposure_time(exp_time)
    spectrum = get_spectrum()
    wl_vector = get_wl_vector()
    plt.plot(wl_vector,spectrum)
    
def camera_close():
    andor_cam.ShutDown()
    spec.Close()

    
        

