# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:55:00 2018

@author: c2dhgr
"""

import visa
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import nidaqmx

rm = visa.ResourceManager()

#ao0 = control port for X-mirror
_set_ao0 = nidaqmx.Task()
_set_ao0.ao_channels.add_ao_voltage_chan(u'Dev1/ao0')

#ao1 = control port for Y-mirror
_set_ao1 = nidaqmx.Task()
_set_ao1.ao_channels.add_ao_voltage_chan(u'Dev1/ao1')

#ai0 = monitor port for X-mirror control (jumpered to ao0)
_get_ai0 = nidaqmx.Task()
_get_ai0.ai_channels.add_ai_voltage_chan(u'Dev1/ai0',terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
                                         min_val=-10.,max_val=10.,units=nidaqmx.constants.VoltageUnits.VOLTS)

#ai1 = monitor port for Y-mirror control (jumpered to ao1)
_get_ai1 = nidaqmx.Task()
_get_ai1.ai_channels.add_ai_voltage_chan(u'Dev1/ai1',terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
                                         min_val=-10.,max_val=10.,units=nidaqmx.constants.VoltageUnits.VOLTS)

#ai2 = monitor port for X-mirror (from control board monitor J6)
_get_ai2 = nidaqmx.Task()
_get_ai2.ai_channels.add_ai_voltage_chan(u'Dev1/ai2',terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
                                         min_val=-10.,max_val=10.,units=nidaqmx.constants.VoltageUnits.VOLTS)

#ai3 = monitor port for Y-mirror (from control board monitor J6)
_get_ai3 = nidaqmx.Task()
_get_ai3.ai_channels.add_ai_voltage_chan(u'Dev1/ai3',terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
                                         min_val=-10.,max_val=10.,units=nidaqmx.constants.VoltageUnits.VOLTS)

#PFI4 = boolean signal to control LASER ON/OFF state
_set_pfi4 = nidaqmx.Task()
_set_pfi4.do_channels.add_do_chan(u'Dev1/pfi4')

##offset for 20x lens
#X_offset = -0.0256105734
#Y_offset =  0.0704633

#offset for 100x lens
X_offset = 0.024572530085365585
Y_offset = 0.12451188327594893

def get_X_mirror_return():
    return _get_ai2.read()

def get_Y_mirror_return():
    return _get_ai3.read()

def set_X_mirror(v):
    return (_set_ao0.write(v+X_offset) == 1,get_X_mirror_input())

def set_Y_mirror(v):
    return (_set_ao1.write(v+Y_offset) == 1,get_Y_mirror_input())
    
def get_X_mirror_input():
    return _get_ai0.read()

def get_Y_mirror_input():
    return _get_ai1.read()

def get_mirrors():
    return (get_X_mirror_input(),get_Y_mirror_input(),get_X_mirror_return(),get_Y_mirror_return())

def set_mirrors(x,y):
    set_X_mirror(x)
    set_Y_mirror(y)
    return get_mirrors()

def laser_state():
    if _set_pfi4.read() == 1:
        return 'laser ON'
    if _set_pfi4.read() == 0:
        return 'laser OFF'

def laser_on():
    _set_pfi4.write(bool(1))
    return laser_state()

def laser_off():
    _set_pfi4.write(bool(0))
    return laser_state()

laser_off()
set_mirrors(0,0)

