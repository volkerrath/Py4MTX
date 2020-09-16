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

def keycount(keyword, fname, what = 'pos'):
    with open(fname, 'r') as fin:
        if what == 'pos':
            return sum([1 for line in fin if keyword in line])
        if what == 'neg':
            return sum([1 for line in fin if keyword not in line])

def unique(list,out=False): 
  
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