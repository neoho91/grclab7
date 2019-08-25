# -*- coding: utf-8 -*-
"""
Created on Sun Sep 02 17:04:26 2018

@author: Neo
"""

import numpy as np
import time
import os
import sys

def load_TH_file(path):
    try:
        f = open(path)
        data = f.read()
        f.close()
    except IOError:
        time.sleep(1)
        f = open(path)
        data = f.read()
        f.close()
    finally:
        data=data.split('\n')
        return data[1:-1]

def get_temp_in_range(start_time,end_time,path='auto',keyword='sensor%i'%(max(list(map(lambda x: int(x.split('sensor')[-1].split('_')[0]),list(filter(lambda x: x.endswith('.txt') and 'config' not in x,os.listdir(r'C:\Users\Millie\Documents\Wifi Sensor Software'))))))),auto_extend_search=True,verbose=False):
    if path == 'auto':
        main_path = r'C:\Users\Millie\Documents\Wifi Sensor Software'
        fs = os.listdir(main_path)
        log = list(filter(lambda x: keyword in x,fs))[0]
        path = os.path.join(main_path,log)
    data = load_TH_file(path)
    if type(start_time) == float or type(start_time) == np.float_:
        start_time = time.localtime(start_time)
    elif type(start_time) == time.struct_time:
        pass
    else:
        start_time = _convert_TH_time_format(start_time)
    if type(end_time) == float or type(end_time) == np.float_:
        end_time = time.localtime(end_time)
    elif type(end_time) == time.struct_time:
        pass
    else:
        end_time = _convert_TH_time_format(end_time)
    curr_data = []
    for datum in data:
        curr_time = _convert_TH_time_format(datum.split(',')[1])
        if start_time > curr_time:
            continue
        else:
            if end_time < curr_time:
                break
            else:
                curr_data.append(datum)
    if len(curr_data) == 0 and auto_extend_search:
        if verbose:
            print('Temperature data not found, searching other log files.')
        main_path = r'C:\Users\Millie\Documents\Wifi Sensor Software'
        fs=list(filter(lambda x: x.endswith('.txt'),os.listdir(main_path)))
        log_fs = list(filter(lambda x: 'sensor' in x or 'readings' in x,fs))
        for log_f in log_fs:
            path = os.path.join(main_path,log_f)
            curr_data = get_temp_in_range(start_time,end_time,path=path,auto_extend_search=False,verbose=False)
            if len(curr_data) > 0:
                break
        if len(curr_data) == 0 and verbose:
            print('Unable to find temperature data in the time range.')
    TH_delete_wdf_files()
    return np.array(curr_data)

def _convert_TH_time_format(time_str='now'):
    if time_str != 'now':
        try:
            return time.strptime(time_str,'%d/%m/%Y %H:%M:%S')
        except ValueError:
            return time.strptime(time_str,'%Y-%m-%d %H:%M:%S')
    else:
        return time.localtime()

def get_temp_only(data):
    return np.array(list(map(lambda x: float(x.split(',')[2]),data)))

def get_latest_temp():
    datum = get_temp_in_range(time.time()-120.,time.time())[-1]
    print datum
    return get_temp_only([datum])[0]

def TH_delete_wdf_files():
    main_path = r'C:\Users\Millie\Documents\Wifi Sensor Software'
    wdf_files = list(filter(lambda x: x.endswith('.wdf'),os.listdir(main_path)))
    for wdf_file in wdf_files[:-1]:
        os.remove(os.path.join(main_path,wdf_file))