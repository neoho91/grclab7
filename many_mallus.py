# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:30:10 2018

@author: Millie
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from fit_mallus import *

def mallus_many_rounds(angles,data):
    adiff = np.array(angles) - np.roll(np.array(angles),1)
    split_indices = []
    for i in range(0,len(adiff)):
        sgn = np.sign(adiff[0]) #tells if angle is increasing or decreasing
        if (sgn*adiff[i] < 0 and sgn*adiff[i-1] > 0 and i!=0): #if angle crosses 0
            split_indices.append(i)
    angle_sets = np.split(np.array(angles),split_indices)
    data_sets = np.split(np.array(data),split_indices)
    print('number of rotations = ' + str(len(angle_sets)))
#    for i in range(5):
#        print(angle_sets[i])
#    print(len(angle_sets),len(data_sets))
    
    fitted_sets = []
    circularity = []
    theta0 = []
    for i in range(0,len(angle_sets)):
        fitted_sets.append(mallus_fitter(angle_sets[i],data_sets[i],plot=False,title=False))
        circularity.append(fitted_sets[i][0][2])
        ang = fitted_sets[i][0][1]
        theta0.append(ang % 360)
#    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(np.arange(0,len(circularity)),circularity)
    ax1.set_title('circularity')
    ax2.plot(np.arange(0,len(theta0)),theta0,color='C1')
    ax2.set_title('fitted theta0')
    plt.show()