# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:10:38 2018

@author: Neo
"""

import numpy as np
import time
import visa
rm=visa.ResourceManager()
yoko=rm.open_resource(u'USB0::0x0B21::0x0023::91G352192-1::1::INSTR',send_end=True)
time.sleep(0.5)
yoko=rm.open_resource(u'USB0::0x0B21::0x0023::91G352192-1::1::INSTR',send_end=True)
yoko.write_raw(u'*CLS')
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

send_yoko(u':waveform:format ASCII')

def yoko_ave():
    send_yoko(u':acquire:mode average')

def yoko_norm():
    send_yoko(u':acquire:mode normal')

def yoko_get_waveform(ch):
    send_yoko(u':waveform:trace %s'%ch)
    return yoko.query_ascii_values(u':WAVeform:SEND?',container=np.array)

def yoko_get_pd_wf():
    return yoko_get_waveform(u'1')

def yoko_get_pos_wf():
    return yoko_get_waveform(u'2')

def yoko_start():
    send_yoko(u':start')

def yoko_stop():
    send_yoko(u':stop')
    
def yoko_remote():
    send_yoko(':COMMunicate:REMote 1')

def yoko_local():
    send_yoko(':COMMunicate:REMote 0')

def yoko_get_interval_of_sample():
    return 1./float(yoko.ask(u':timebase:srate?').split('SRAT ')[-1])

def yoko_get_length():
    return int(yoko.ask(u':WAVeform:LENGth?').split('LENG ')[-1])

def yoko_get_time_stamp():
    dt = yoko_get_interval_of_sample()
    n = yoko_get_length()
    return np.linspace(0,(n-1)*dt,n)
    

yoko_local()
print('Yokogawa online.')