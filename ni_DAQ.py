# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:59:02 2019

@author: Neo
"""

import nidaqmx

_AO1 = nidaqmx.Task()
_AO1.ao_channels.add_ao_voltage_chan(u'Dev1/ao1',min_val=0,max_val=5)


def set_DAQ_AO1(val):
    _AO1.write(val)