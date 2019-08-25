# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:33:01 2018

@author: Millie
"""

import numpy as np
import scipy as sp
import copy
import matplotlib.pyplot as plt
import os
import sys
from neo_common_code import *
import PIL
import cv2
from scipy.ndimage import filters

source_file_path = os.path.join(r'C:/Users/Millie/Desktop/VP2 sample/WSe2_VP2_010x.png')
flake_file_path = os.path.join(r'C:/Users/Millie/Desktop/VP2 sample/test.jpg')

source_mag = 10.
flake_mag = 120.

source = np.array(PIL.Image.open(source_file_path).convert('L'))
flake = np.array(PIL.Image.open(flake_file_path).convert('L'))

def search_single_image(flake,source,source_mag,flake_mag,update=False):
    reduced_flake = cv2.resize(flake,(0,0),fx=source_mag/flake_mag,fy=source_mag/flake_mag)

    img = copy.copy(source)
    img2 = copy.copy(img)
    template = reduced_flake
    w,h = template.shape[::-1]
    
    img = img2.copy()
    other_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = eval(other_methods[1]) #default to 1
    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img,top_left, bottom_right, int(255*max_val), 2)
    
    if not update:
        global _fig,flake_im,source_im,fig2
        _fig = plt.figure('Live')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        fig = _fig.add_subplot(121)
        flake_im = fig.imshow(flake,cmap = 'gray')
        fig.set_title('Press B to capture new background.')
        plt.xticks([]), plt.yticks([])
        #plt.subplot(122),plt.imshow(res,cmap = 'gray')
        #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        fig2 = _fig.add_subplot(122)
        source_im = fig2.imshow(img,cmap = 'gray')
        fig2.set_title('Detected Point, matching coeff = %.4f'%max_val), plt.xticks([]), plt.yticks([])
        
        plt.pause(1e-6)
        plt.tight_layout(rect=(0,0,1,0.95))
        plt.show()
    else:
        flake_im.set_data(flake)
        source_im.set_data(img)
        fig2.set_title('Detected Point, matching coeff = %.4f'%max_val), plt.xticks([]), plt.yticks([])
        plt.show()
        plt.pause(1e-6)

def get_live_image_bg():
    cam.start_live_video()
    img = cam.latest_frame()
    np.save('D:/Nonlinear_setup/Python_codes/search_live_image_bg.npy',img)

def search_live_image(source=source,source_mag=10.,flake_mag=126.5):
    try:
        sys.path.append(r'D:\Nonlinear_setup\Python_codes')
        global cam, bg
        bg = np.load('D:/Nonlinear_setup/Python_codes/search_live_image_bg.npy')
        bg = 1.*bg/np.max(bg)
        try:
            from instrumental.drivers.cameras import uc480
            cam = uc480.UC480_Camera()
        except:
            pass
        clahe = cv2.createCLAHE(clipLimit=.1, tileGridSize=(80,80))
        def capture_image():
            img = cam.latest_frame()
            img = (img/bg).astype(np.uint8)
#            img = clahe.apply(img)
            img = cv2.equalizeHist(img)
            img = filters.minimum_filter(img,3)
#            img = cv2.fastNlMeansDenoising(img)
            return np.fliplr(img)
        def press(event):
            if event.key == 'b':
                get_live_image_bg()
                global bg
                bg = np.load('D:/Nonlinear_setup/Python_codes/search_live_image_bg.npy')
                bg = 1.*bg/np.max(bg)
        cam.auto_blacklevel = 1
        cam.auto_exposure = 1
        cam.auto_framerate = 1
        cam.auto_gain = 1
        cam.start_live_video()
#        cam.start_capture()
        flake = capture_image()
        search_single_image(flake,source,source_mag,flake_mag)
        _fig.canvas.mpl_connect('key_press_event',press)
        plt.pause(1e-6)
        while True:
            flake = capture_image()
            search_single_image(flake,source,source_mag,flake_mag,update=True)
            plt.pause(1e-6)
    except KeyboardInterrupt:
        cam.stop_live_video()
#        cam.close()

if __name__ == '__main__':
    search_live_image()
#    search_single_image(flake,source,source_mag,flake_mag)