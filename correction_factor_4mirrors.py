# -*- coding: utf-8 -*-
"""
Created on Sun Sep 09 22:50:08 2018

@author: Neo
"""

import numpy as np

def CC_4mirrors(theta,theta_o,R,phi,ret):
    """
    returns correction coefficient, where inten_before_4mirrors(theta) = CC_4mirrors(theta,...)*inten_after_4mirrors(theta)
    theta = polarization angle wrt lab horizontal frame, deg
    theta_o = highest intensity angle of after 4mirror intensity, deg
    R = min:max intensity ratio of after 4mirror, obtained from fitting with Mallus' law
    phi = tilting angle of 4mirrors, deg
    ret = retardance of 4 mirrors, wave
    """
    theta = np.pi*theta/180
    theta_o = np.pi*theta_o/180
    phi = np.pi*phi/180
    return (1 - 2.*np.sqrt(R)/(1+R)*np.sin(2*np.pi*ret)*np.sin(2*(theta-phi)) + np.cos(2*np.arctan(np.sqrt(R)))*(np.square(np.cos(np.pi*ret))*np.cos(2*(theta-theta_o)) + np.square(np.sin(np.pi*ret))*np.cos(2*(theta+theta_o-2*phi)))
     ) / (
             2./(1+R)*(np.square(np.cos(theta-theta_o))+R*np.square(np.sin(theta-theta_o))))