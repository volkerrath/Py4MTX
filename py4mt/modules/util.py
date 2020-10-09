import os
import sys
import numpy as np

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def unique(list,out=False): 
    '''
    find unique elements in list/array
        
    VR 9/20
    
    '''
    
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    # print list 
    if out:
        for x in unique_list: print(x)
        
    return unique_list

def strcount(keyword=None, fname=None):
    '''
    count occurences of keyword in file 
     Parameters
    ----------
    keywords : TYPE, optional
        DESCRIPTION. The default is None.
    fname : TYPE, optional
        DESCRIPTION. The default is None.
        
    VR 9/20
    '''   
    with open(fname, 'r') as fin:
        return sum([1 for line in fin if keyword in line])
    # sum([1 for line in fin if keyword not in line])


def strdelete(keyword=None, fname_in=None, fname_out=None, out = True):
    '''
    delete lines containing on of the keywords in list
    
    Parameters
    ----------
    keywords : TYPE, optional
        DESCRIPTION. The default is None.
    fname_in : TYPE, optional
        DESCRIPTION. The default is None.
    fname_out : TYPE, optional
        DESCRIPTION. The default is None.
    out : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    VR 9/20
    '''
    nn = strcount(keyword,fname_in)
    
    if out:
        print(str(nn)+' occurances of <'+keyword+'> in '+fname_in)
        
    # if fname_out == None: fname_out= fname_in
    with open(fname_in, 'r') as fin, open(fname_out, 'w') as fou:
        for line in fin:
                if not keyword in line:
                    fou.write(line)
 
def strreplace(key_in=None, key_out=None,fname_in=None, fname_out=None):
    '''
    replaces key_in in keywords by key_out

    Parameters
    ----------
    key_in : TYPE, optional
        DESCRIPTION. The default is None.
    key_out : TYPE, optional
        DESCRIPTION. The default is None.
    fname_in : TYPE, optional
        DESCRIPTION. The default is None.
    fname_out : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.
    
    VR 9/20

    '''

    with open(fname_in, 'r') as fin, open(fname_out, 'w') as fou:
        for line in fin:
            fou.write(line.replace(key_in, key_out))

# Construct list of EDI-files:
def gen_grid(LatLimits=None, nLat=None,LonLimits=None, nLon=None, out=True):
    small = 0.000001
# LonLimits = ( 6.275, 6.39)
# nLon = 31
    LonStep  = (LonLimits[1] - LonLimits[0])/nLon
    Lon = np.arange(LonLimits[0],LonLimits[1]+small,LonStep)

# LatLimits = (45.37,45.46)
# nLat = 31
    LatStep  = (LatLimits[1] - LatLimits[0])/nLat
    Lat = np.arange(LatLimits[0],LatLimits[1]+small,LatStep)
    
    return Lat, Lon

