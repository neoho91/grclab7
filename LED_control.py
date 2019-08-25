# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:04:14 2019

@author: Neo
"""
import time
import numpy as np
import scipy as sp
import os
import sys
sys.path.append(os.path.abspath(r"D:\Nonlinear_setup\Python_codes"))

from ni_DAQ import *

def LED_power(p):
    set_DAQ_AO1(p*5)

    