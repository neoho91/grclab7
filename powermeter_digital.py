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

power_meter2=rm.open_resource('USB0::0x1313::0x8078::P0007871::INSTR')
#power_meter2=rm.open_resource(u'USB0::0x1313::0x8078::P0009181::INSTR')#goki's
powermeter2 = ThorlabsPM100(inst=power_meter2)
powermeter2.configure.scalar.power()
powermeter2.sense.average.count = 300
#Following command sets the bandwith to high state=0
powermeter2.input.pdiode.filter.lpass.state=1#1 for low bandwidth, ie filtering high freq
powermeter2.sense.power.dc.range.auto=1

def pmd_wl(wl=None):
    if wl==None:
        return powermeter2.sense.correction.wavelength
    else:
        powermeter2.sense.correction.wavelength = wl
        return pmd_wl(wl=None)

def pmd_zero():
    powermeter2.sense.correction.collect.zero.initiate()
    
def pmd_power():
    powermeter2.initiate.immediate()
    try:
        return powermeter2.read
    except:
        time.sleep(0.1)
        return powermeter2.read

def pmd_refresh():
    powermeter2.input.pdiode.filter.lpass.state=0
    time.sleep(0.3)
    powermeter2.input.pdiode.filter.lpass.state=1

print 'Powermeter (Digital) online.'