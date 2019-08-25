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

power_meter4=rm.open_resource(u'USB0::0x1313::0x8072::1903661::INSTR')
powermeter4 = ThorlabsPM100(inst=power_meter4)
powermeter4.configure.scalar.power()
powermeter4.sense.average.count = 5
#Following command sets the bandwith to high state=0
powermeter4.input.pdiode.filter.lpass.state=0 #0 = high bandwidth, 1 = low bandwidth
powermeter4.sense.power.dc.range.auto=1

def pmp_wl(wl=None):
    if wl==None:
        return powermeter4.sense.correction.wavelength
    else:
        powermeter4.sense.correction.wavelength = wl
        return pmp_wl(wl=None)

def pmp_zero():
    powermeter4.sense.correction.collect.zero.initiate()
    
def pmp_power():
    try:
        return powermeter4.read
    except:
        time.sleep(0.1)
        return powermeter4.read

print 'Powermeter (Polarimeter) online.'