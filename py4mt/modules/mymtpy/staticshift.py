#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:20:03 2020

@author: sb & vr
(based on mtpy)

"""
# ==============================================================================

def estimate_static_spatial_median(edi_fn, 
                                   radius=20000.,
                                   prefix_remove='Bad_',
                                   freq_interval=[1.e-4,1.e4], 
                                   shift_tol=.15):
    """
    Remove static shift from a station using a spatial median filter.  This
    will look at all the edi files in the same directory as edi_fn and find
    those station within the given radius (meters).  Then it will find
    the medain static shift for the x and y modes and remove it, given that
    it is larger than the shift tolerance away from 1.  

    Arguments
    -----------------
        **edi_fn** : string
                        full path to edi file to have static shift removed

        **radius** : float
                        radius to look for nearby stations, in meters.
                        *default* is 1000 m
        **prefix_remove**: string
                        edi files begining with this string will not be used for 
                        interpolation
                        *default* is 'Bad_'.
        **freq_interval** : float
                        number of frequencies to skip from the highest
                        frequency.  Sometimes the highest frequencies are
                        not reliable due to noise or low signal in the AMT
                        deadband.  This allows you to skip those frequencies.
                        *default* is 4

        **shift_tol** : float
                        Tolerance on the median static shift correction.  If
                        the data is noisy the correction factor can be biased
                        away from 1.  Therefore the shift_tol is used to stop
                        that bias.  If 1-tol < correction < 1+tol then the
                        correction factor is set to 1.  *default* is 0.15


    Returns
    ----------------

        **shift_corrections** : (float, float)
                                static shift corrections for x and y modes

    """
    
    import os
    import numpy as np
    import mtpy.core.mt as mt
    
    # convert meters to decimal degrees so we don't have to deal with zone
    # changes
    meter_to_deg_factor = 8.994423457456377e-06
    dm_deg = radius * meter_to_deg_factor

    # make a list of edi files in the directory
    edi_path = os.path.dirname(edi_fn)
    edi_test = [os.path.abspath(os.path.join(edi_path, edi))
                for edi in os.listdir(edi_path)
                if edi.endswith('.edi')]
    
    
    # only keep good stations
    edi_list = []
    
    for ename in edi_test:
        if not os.path.basename(ename).startswith(prefix_remove):
            edi_list.append(ename)
            
    for ename in edi_list:
         print(ename)   
         
    # remove current site if in list
    try:
        edi_list.remove(os.path.abspath(edi_fn))
        print('Current station removed: '+os.path.basename(edi_fn))
    except:
        print('Current station not in list: '+os.path.basename(edi_fn))
    
    for ename in edi_list:
         print(os.path.basename(ename))
         
         
    # read the edi file
    
    mt_obj = mt.MT(edi_fn)
    freqs = mt_obj.Z.freq
    # nf = np.size(freqs)
    
  
    mt_obj.Z.compute_resistivity_phase()    
    
    
    
    fi =  np.where((freqs>freq_interval[0]) & (freqs<freq_interval[1]))
    imin=np.min(fi)
    imax=np.max(fi)
    interp_freq =  freqs[imin:imax]
    nf = np.size(interp_freq)

    # Find stations near by and store them in a list
    # dist_list = []
    mt_obj_list = []
    for kk, kk_edi in enumerate(edi_list):
        mt_obj_2 = mt.MT(kk_edi)
        delta_d = np.sqrt((mt_obj.lat - mt_obj_2.lat) ** 2 +
                          (mt_obj.lon - mt_obj_2.lon) ** 2)
        # dist_list.append(delta_d/meter_to_deg_factor)
        if delta_d <= dm_deg:
            mt_obj_2.delta_d = float(delta_d) / meter_to_deg_factor
            mt_obj_list.append(mt_obj_2)
    # print(dist_list)  
    # print(radius) 
    if len(mt_obj_list) == 0:
        print('No stations found within given radius {0:.2f} m'.format(radius))
        return 1.0, 1.0

    # extract the resistivity values from the near by stations
    res_array = np.zeros((len(mt_obj_list), nf, 2, 2))
    print('These stations are within the given {0} m radius:'.format(radius))
    
    for kk, mt_obj_kk in enumerate(mt_obj_list):
        print('\t{0} --> {1:.1f} m'.format(mt_obj_kk.station, mt_obj_kk.delta_d))
        interp_idx = np.where((interp_freq >= mt_obj_kk.Z.freq.min()) &
                              (interp_freq <= mt_obj_kk.Z.freq.max()))
        
        interp_freq_kk = interp_freq[interp_idx]
        #print(np.shape(interp_freq_kk ))
        Z_interp, Tip_interp = mt_obj_kk.interpolate(interp_freq_kk)
        Z_interp.compute_resistivity_phase()
        res_array[
            kk,
            interp_idx,
            :,
            :] = Z_interp.resistivity[
            0:len(interp_freq_kk),
            :,
            :]

    # compute the static shift of x-components
    static_shift_x = mt_obj.Z.resistivity[imin:imax, 0, 1] / \
        np.median(res_array[:, :, 0, 1], axis=0)
    static_shift_x = np.median(static_shift_x)

    # check to see if the estimated static shift is within given tolerance
    if 1 - shift_tol < static_shift_x and static_shift_x < 1 + shift_tol:
        static_shift_x = 1.0

    # compute the static shift of y-components
    static_shift_y = mt_obj.Z.resistivity[imin:imax, 1, 0] / \
        np.median(res_array[:, :, 1, 0], axis=0)
    static_shift_y = np.median(static_shift_y)

    # check to see if the estimated static shift is within given tolerance
    if 1 - shift_tol < static_shift_y and static_shift_y < 1 + shift_tol:
        static_shift_y = 1.0
        

    return static_shift_x, static_shift_y
