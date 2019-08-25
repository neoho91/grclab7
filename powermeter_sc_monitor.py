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

power_meter6=rm.open_resource(u'USB0::0x1313::0x8072::1903728::INSTR')
powermeter6 = ThorlabsPM100(inst=power_meter6)
powermeter6.configure.scalar.power()
powermeter6.sense.average.count = 100
#Following command sets the bandwith to high state=0
powermeter6.input.pdiode.filter.lpass.state=1
powermeter6.sense.power.dc.range.auto=1

def pmscm_wl(wl=None):
    if wl==None:
        return powermeter6.sense.correction.wavelength
    else:
        powermeter6.sense.correction.wavelength = wl
        return pmscm_wl(wl=None)

def pmscm_zero():
    powermeter6.sense.correction.collect.zero.initiate()
    
def pmscm_power():
    try:
        return powermeter6.read
    except:
        time.sleep(0.1)
        return powermeter6.read

print 'Powermeter (SC monitor) online.'