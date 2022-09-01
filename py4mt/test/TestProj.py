#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
# ---

"""
Created on Wed Nov  4 15:18:34 2020

@author: vrath
"""
import sys
mypath = ['/home/vrath/AEMpyX/aempy/modules/','/home/vrath/AEMpyX/aempy/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.append(pth)


import util
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# utm_zone = 32629

# itm_x = 526686.10
# itm_y = 612275.21
# print('itm (in): %1.6f  %1.6f'% (itm_x, itm_y))


# latitude, longitude = util.proj_itm_to_latlon(itm_x, itm_y)
# print('latlon : %1.7f %1.7f' %(latitude,longitude))

# itm_x, itm_y = util.proj_latlon_to_itm(latitude,longitude)
# print('itm (out): %1.6f  %1.6f'% (itm_x, itm_y))

# utm_x, utm_y = util.proj_latlon_to_utm(latitude, longitude, utm_zone)
# print('utm (out): %1.6f  %1.6f'% (utm_x, utm_y))
# utm_x, utm_y = util.proj_itm_to_utm(itm_x,itm_y, utm_zone)
# print('utm (out): %1.6f  %1.6f'% (utm_x, utm_y))

# print('\n\n\n')
# print('Example Ireland ')
# longitude =  -8.636038
# latitude =  52.461333
# utm_zone = 32629
# print('latlon (in): %1.7f %1.7f' %(latitude,longitude))

# itm_x, itm_y = util.proj_latlon_to_itm(latitude,longitude)
# print('itm (out): %1.6f  %1.6f'% (itm_x, itm_y))

# utm_x, utm_y = util.proj_latlon_to_utm(latitude,longitude,utm_zone)
# print('utm (out): %1.6f  %1.6f'% (utm_x, utm_y))

# latitude, longitude = util.proj_utm_to_latlon(utm_x, utm_y)
# print('latlon : %1.7f %1.7f' %(latitude,longitude))


# latitude, longitude = util.proj_itm_to_latlon(itm_x, itm_y)
# print('latlon : %1.7f %1.7f' %(latitude,longitude))
# print('\n\n\n')

print('Example Iceland ')
CenterLatLon = [65.771, -16.778]

longitude =  -16.778
latitude =  65.771
epsg = util.get_utm_zone(latitude, longitude)
utm_zone =epsg[0]
print('latlon (in): %1.7f %1.7f' %(latitude,longitude))

utm_x, utm_y = util.proj_latlon_to_utm(latitude,longitude,utm_zone)
print('utm (out): %1.6f  %1.6f  utm = %8i'% (utm_x, utm_y, utm_zone))

latitude, longitude = util.proj_utm_to_latlon(utm_x, utm_y, utm_zone)
print('latlon : %1.7f %1.7f' %(latitude,longitude))


print('\n\n\n')
