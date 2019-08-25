# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:48:17 2017

@author: Neo
"""

import visa
import time
import numpy as np

rm=visa.ResourceManager()
yoko=rm.open_resource(u'USB0::0x0B21::0x0023::91G352192-1::1::INSTR',send_end=True)
#time.sleep(2)
#yoko=rm.open_resource(u'USB0::0x0B21::0x0023::91G352192-1::1::INSTR',send_end=True)
yoko.write_raw(u':COMMunicate:REMote 1')
time.sleep(2)
#cLEARS VISA OBJECT
yoko.write_raw(u'*CLS')

#Asks visa object identity
yoko.ask(u'*IDN?')
def send_yoko(u):
    try:
        if u[-1]=='?':
            return yoko.ask(unicode(u))
        else:
            yoko.write_raw(unicode(u))
    except:
        time.sleep(10)
        yoko.write_raw(u'*CLS')
        yoko.ask(u'*IDN?')
        send_yoko(u)
        
#send_yoko(u':measure:calculation:def2 "hilow(c2)"')
#send_yoko(u':measure:calculation:def1 "hilow(c1)"')
#send_yoko(u':measure:calculation:state2 1')
#send_yoko(u':measure:calculation:state1 1')
#
#send_yoko(u':measure:calculation:def2 "hilow(c2)"')
#send_yoko(u':measure:calculation:def1 "hilow(c1)"')
#send_yoko(u':measure:calculation:def3 "hilow(c3)"')
send_yoko(u':measure:calculation:def2 "mean(c2)"')
send_yoko(u':measure:calculation:def1 "mean(c1)"')
send_yoko(u':measure:calculation:def3 "mean(c3)"')
send_yoko(u':measure:calculation:state2 1')
send_yoko(u':measure:calculation:state1 1')
send_yoko(u':measure:calculation:state3 1')

send_yoko(':ANALysis:AHIStogram1:MEASure:PARameter:CALCulation:define1 "Peak"')
send_yoko(':ANALysis:AHIStogram1:MEASure:PARameter:CALCulation:state1 1')
send_yoko(':ANALysis:AHIStogram2:MEASure:PARameter:CALCulation:define1 "Mean"')
send_yoko(':ANALysis:AHIStogram2:MEASure:PARameter:CALCulation:state1 1')
send_yoko(':ANALysis:AHIStogram2:MEASure:PARameter:CALCulation:define2 "Sigma"')
send_yoko(':ANALysis:AHIStogram2:MEASure:PARameter:CALCulation:state2 1')

def yoko_get_ch_vdiv(ch):
    return float(send_yoko(':channel%i:vdiv?'%ch).split('VDIV')[1].split(':')[0])

def yoko_set_ch_vdiv(ch,vdiv):
    send_yoko(':channel%i:vdiv %f'%(ch,vdiv))
    time.sleep(1)

def yoko_ch1_autoscale_hist():
    curr_vdiv = np.array(yoko_get_ch_vdiv(1))
    div_from_centre = np.abs(np.sum(yoko_histogram_peaks_diff()/curr_vdiv))
    new_vdiv = curr_vdiv*div_from_centre/2.5
    yoko_set_ch_vdiv(1,new_vdiv)

def yoko_ch2_autoscale_amp():
    curr_vdiv = np.array(yoko_get_ch_vdiv(2))
    yoko_restart_stat()
    div_from_bottom = np.abs(yoko_ch2_amp()/curr_vdiv)
    while div_from_bottom > 11:
        yoko_set_ch_vdiv(2,curr_vdiv*2.)
        curr_vdiv = yoko_get_ch_vdiv(2)
        yoko_restart_stat()
        div_from_bottom = np.abs(yoko_ch2_amp()/curr_vdiv)
    new_vdiv = curr_vdiv*div_from_bottom/6.
    yoko_set_ch_vdiv(2,new_vdiv)
    
def yoko_ch3_autoscale_amp():
    curr_vdiv = np.array(yoko_get_ch_vdiv(3))
    yoko_restart_stat()
    div_from_bottom = np.abs(yoko_ch3_amp()/curr_vdiv)
    while div_from_bottom > 11:
        yoko_set_ch_vdiv(3,curr_vdiv*2.)
        curr_vdiv = yoko_get_ch_vdiv(3)
        yoko_restart_stat()
        div_from_bottom = np.abs(yoko_ch3_amp()/curr_vdiv)
    new_vdiv = curr_vdiv*div_from_bottom/6.
    yoko_set_ch_vdiv(3,new_vdiv)

def yoko_ch1_autoscale():
    send_yoko(':chan1:asc')
    time.sleep(1)

def yoko_ch2_autoscale():
    send_yoko(':chan2:asc')
    time.sleep(1)

def yoko_ch3_autoscale():
    send_yoko(':chan3:asc')
    time.sleep(1)

def yoko_histogram_peaks_diff():
    try:
        hist_peak_1 = float(send_yoko(':ANALysis:AHIStogram1:MEASure:PARameter:CALCulation:value1?').split('VAL1')[1].split(':')[0])
        hist_peak_2 = float(send_yoko(':ANALysis:AHIStogram2:MEASure:PARameter:CALCulation:value1?').split('VAL1')[1].split(':')[0])
        hist_std = float(send_yoko(':ANALysis:AHIStogram2:MEASure:PARameter:CALCulation:value2?').split('VAL2')[1].split(':')[0])
        return (hist_peak_2 - hist_peak_1,hist_std)
    except visa.VisaIOError, ValueError:
        time.sleep(0.02)
        send_yoko(u'*cls')
        return yoko_histogram_peaks_diff()
        
    
def yoko_remote():
    send_yoko(':COMMunicate:REMote 1')

def yoko_local():
    send_yoko(':COMMunicate:REMote 0')

def yoko_restart_stat():
    send_yoko(u':measure:continuous:restart')
    time.sleep(0.5)

def yoko_restart_meas():
    send_yoko(u':stop')
    send_yoko(u':start')
    time.sleep(1)

def yoko_ch1_amp():
    return float(send_yoko(u':measure:calculation:mean1?').split('MEAN1')[1])

def yoko_ch1_std():
    return float(send_yoko(u':measure:calculation:sdev1?').split('SDEV1')[1])

def yoko_ch2_amp():
    return float(send_yoko(u':measure:calculation:mean2?').split('MEAN2')[1])

def yoko_ch2_std():
    return float(send_yoko(u':measure:calculation:sdev2?').split('SDEV2')[1])

def yoko_ch3_amp():
    return float(send_yoko(u':measure:calculation:mean3?').split('MEAN3')[1])

def yoko_ch3_std():
    return float(send_yoko(u':measure:calculation:sdev3?').split('SDEV3')[1])

print('Yokogawa online.')
yoko_local()