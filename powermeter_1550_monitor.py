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

power_meter5=rm.open_resource(u'USB0::0x1313::0x8072::1903729::INSTR')
powermeter5 = ThorlabsPM100(inst=power_meter5)
powermeter5.configure.scalar.power()
powermeter5.sense.average.count = 100
#Following command sets the bandwith to high state=0
powermeter5.input.pdiode.filter.lpass.state=1
powermeter5.sense.power.dc.range.auto=1

def pm1550m_wl(wl=None):
    if wl==None:
        return powermeter5.sense.correction.wavelength
    else:
        powermeter5.sense.correction.wavelength = wl
        return pm1550m_wl(wl=None)

def pm1550m_zero():
    powermeter5.sense.correction.collect.zero.initiate()
    
def pm1550m_power():
    try:
        return powermeter5.read
    except:
        time.sleep(0.1)
        return powermeter5.read

print 'Powermeter (1550 monitor) online.'