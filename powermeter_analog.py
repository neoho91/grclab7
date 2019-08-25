# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:02:18 2017

@author: Neo
"""
import visa
import time
from ThorlabsPM100 import ThorlabsPM100

rm = visa.ResourceManager()
rm.list_resources()      

power_meter1=rm.open_resource('USB0::0x1313::0x8079::P1001337::INSTR')
powermeter1 = ThorlabsPM100(inst=power_meter1)
powermeter1.configure.scalar.power()
powermeter1.sense.average.count = 300
#Following command sets the bandwith to high state=0
powermeter1.input.pdiode.filter.lpass.state=1
powermeter1.sense.power.dc.range.auto=1

def pma_wl(wl=None):
    if wl==None:
        return powermeter1.sense.correction.wavelength
    else:
        powermeter1.sense.correction.wavelength = wl
        return pma_wl(wl=None)

def pma_zero():
    powermeter1.sense.correction.collect.zero.initiate()
    
def pma_power():
    powermeter1.initiate.immediate()
    try:
        return powermeter1.read
    except:
        time.sleep(0.1)
        return powermeter1.read

def pma_refresh():
    powermeter1.input.pdiode.filter.lpass.state=0
    time.sleep(0.3)
    powermeter1.input.pdiode.filter.lpass.state=1

print 'Powermeter (Analog) online.'