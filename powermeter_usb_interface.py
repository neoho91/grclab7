# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:52:06 2018

@author: Millie
"""

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

power_meter3=rm.open_resource('USB0::0x1313::0x8072::P2006195::INSTR')
powermeter3 = ThorlabsPM100(inst=power_meter3)
powermeter3.configure.scalar.power()
powermeter3.sense.average.count = 100
#Following command sets the bandwith to high state=0
powermeter3.input.pdiode.filter.lpass.state=0
powermeter3.sense.power.dc.range.auto=1

def pmu_wl(wl=None):
    if wl==None:
        return powermeter3.sense.correction.wavelength
    else:
        powermeter3.sense.correction.wavelength = wl
        return pmu_wl(wl=None)

def pmu_zero():
    powermeter3.sense.correction.collect.zero.initiate()
    
def pmu_power():
    try:
        return powermeter3.read
    except:
        time.sleep(0.1)
        return powermeter3.read

print 'Powermeter (USB interface) online.'