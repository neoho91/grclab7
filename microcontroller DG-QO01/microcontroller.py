# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:33:28 2015

@author: GRCLAB7


"""

import visa
import time
import numpy as np
rm=visa.ResourceManager()
DG=rm.get_instrument("ASRL58::INSTR")

DG.write('*RST') #reset
idn = DG.ask('*IDN?')
if len(idn)>2:
    print idn[:-2] + ' online.'

relay_out_port = 0
relay_in_port = 2

def relay_on():
    DG.write('ONB '+str(relay_out_port))
    
def relay_off():
    DG.write('OFFB '+str(relay_out_port))

def on_chb():
    """
    Return the output channel(s) B that are turned on.
    """
    ans = DG.ask('portb?')
    ans = int(np.binary_repr(int(ans)))
    i=0
    result = []
    while ans > 0:
        ele = ans%10
        if ele != 0:
            result.append(i)
        i += 1
        ans /= 10
    return result
    
def on_chd():
    """
    Return the output channel(s) D that are turned on.
    """
    ans = DG.ask('portd?')
    ans = int(np.binary_repr(int(ans)))
    i=0
    result = []
    while ans > 0:
        ele = ans%10
        if ele != 0:
            result.append(i)
        i += 1
        ans /= 10
    return result

def on_cha():
    """
    Return the input channel(s) A that are turned on.
    """
    ans = DG.ask('porta?')
    ans = int(np.binary_repr(int(ans)))
    i=0
    result = []
    while ans > 0:
        ele = ans%10
        if ele == 0:
            result.append(i)
        i += 1
        ans /= 10
    return result
    
def switch(slp=0.08,verbose=True):
    relay_on()
    time.sleep(1e-2)
    relay_off()
    time.sleep(slp)
    if relay_in_port in on_cha():
        curr_stat = 0
        if verbose:
            print 'Relay is on. Laser is blocked (0).'
    else:
        curr_stat = 1
        if verbose:
            print 'Relay is off. Laser is unblocked (1).'
    return curr_stat

def block_laser(verbose=True):
    if relay_in_port in on_cha():
        if verbose:
            print 'Relay is already on. Laser is already being blocked (0).'
        return 0
    else:
        return switch(verbose=verbose)
    
def unblock_laser(verbose=True):
    if relay_in_port not in on_cha():
        if verbose:
            print 'Relay is already off. Laser is already being unblocked (1).'
        return 1
    else:
        return switch(verbose=verbose)