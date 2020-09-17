import os
import sys

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


def strdelete(keywords=None, fname_in=None, fname_out=None, what = 'pos', out = True):
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
    what : TYPE, optional
        DESCRIPTION. The default is 'pos'.
    out : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    VR 9/20
    '''
    for kkey in keywords: 
        nn = strcount(kkey,fname_in)
        if out:
            print(str(nn)+' occurances of <'+kkey+'> in '+fname_in)
        
    if fname_out == None: fname_out= fname_in
    with open(fname_in, 'r') as fin, open(fname_out, 'w') as fou:
        for line in fin:
            if what == 'pos':
                if not any(key in line for key in keywords):
                    fou.write(line)
                else:
                    continue


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
    if fname_out == None: fname_out= fname_in
    with open(fname_in, 'r') as fin, open(fname_out, 'w') as fou:
        for line in fin:
            fou.write(line.replace(key_in, key_out))
