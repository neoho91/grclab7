# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 18:16:40 2018

@author: Millie
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
sys.path.append(r'D:\Nonlinear_setup\Python_codes')
from instrumental.drivers.cameras import uc480
cam = uc480.UC480_Camera()