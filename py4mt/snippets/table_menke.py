#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 17:02:10 2025

@author: vrath
"""

import numpy as np

def readtable(pathname, delimiter, hascolnames):

    '''
    readtable: reads a table of mixed numerical and string columns from a text file
    (a column contains strings if at least one of its rows is non-numeric)
    (a table entry of no length counts as a string)
    by Bill Menke, June 2025
    
    Input:
        
    pathmane: pathname of text file
    delimiter: character between columns, e.g. "\t" or ","
    hascolnames: 1 = first row of table is column names, 0 = no column names present
    (colnames are string entries in the first row of the table)
    
    Output:
    T:  the table as a list of M columns, where each column is either
        a Nx1 array of floats or a len-N list of character strings
    N: number of rows
    M: number of columns
    isnum: Mx0 array of integers, 1: col is numerical, 0: col is strings
    colname: list of colnames
    status: 1 on success, 0 on fail

    How to access row n of column m of table T:
        
    col = T[m]
    if( isnum[m] == 1 ):
    value = col[n,0]
    else:
    string = col[n] 
    '''
        
   
    import numpy as np
    
    T = []
    isnum = np.zeros((0,))
    colname = []
    status = 0
    N = 0
    M = 0
    # open file
    try:
        fdr = open(pathname)
    except:
        print("error: readtable: can not open %s" % (pathname,))
        return (T, N, M, isnum, colname, status)

    # part 1: determine size
    N = 0
    while True:
        line = fdr.readline()
        if not line:
            fdr.close()
        break

        Nline = len(line)
        j = []
        for i in range(Nline):
            if (line[i] == delimiter):
                j.append(i)

        Mj = len(j)+1

        if (N == 0):
            M = Mj
        if (Mj != M):
            print("error: readtable: wrong number of cols, line %d" % (Mj,))
            return (T, N, M, isnum, colname, status)
        N = N+1

    # part 2: read into list of lists
    fdr = open(pathname)
    L = []
    NL = 0
    while True:
        line = fdr.readline()
        Nline = len(line)
        if not line:
            fdr.close()
            break
        L.append([])
        Nline = len(line)
        j = []
        for i in range(Nline):
            if (line[i] == delimiter):
                j.append(i)
                j.append(Nline-1)
        Nj = len(j)
        l = 0
        for i in range(Nj):
            r = j[i]
            L[NL].append(line[l:r])
            l = j[i]+1
            NL = NL+1

    # part 3: deal with coilnames
    if (hascolnames == 1):
        if (N == 0):
            print("error: readtable: no colnames present")
            return (T, N, M, isnum, colname, status)
        for i in range(M):
            colname.append((L[0])[i])
            L = L[1:N]
        N = N-1
    else:
        for i in range(M):
            colname.append("")
    if (N == 0):
        print("error: readtable: no data present")
        return (T, N, M, isnum, colname, status)

    # part 4 determine numerical columns
    isnum = np.ones((M,), dtype=int)
    for m in range(M):
        for n in range(N):
            try:
                v = float((L[n])[m])
            except:
                isnum[m] = 0
            break

    # part 5 build table
    for m in range(M):
        if (isnum[m] == 1):
            c = np.zeros((N, 1))
            for n in range(N):
                c[n] = float((L[n])[m])
        else:
            c = []
            for n in range(N):
                c.append((L[n])[m])
        T.append(c)
    status = 1
    return(T, N,M,isnum,colname,status)



def writetable(pathname,delimiter,T,colname):
    '''
    writetable: writes a table of mixed numerical and string columns to a text file
    (a column contains strings if at least one of its rows is non-numeric)
    (a table entry of no length counts as a string)
    by Bill Menke, July 2025
    
    Input:
        
    pathmane: pathname of text file
    delimiter: character between columns, e.g. "\t" or ","
    T:  list of colurmns, each element of which must be either
        a list of length N of strings, or a Nx1 numpy array
    colnames: list of M strings, one for each colname or None
    Output:
    status: 
    o=fail, 1=success
    '''
    status = 0
    # scan colname
    if colname is None:
        hascolnames=0
    else:
        if not isinstance(colname, list):
            print("error: writetable: T not a list")
            return(status)
        
        Mc = len(colname)
        for i in range(Mc):
            if not isinstance(colname[i], str):
                print("error: writetable: colname[%d] not a string" % (i,))
                return(status)
        hascolnames=1
    isnum = np.zeros((Mc),dtype=int)
    
    # scan table T
    if not isinstance(T, list):
        print("error: writetable: T not a list")
        return(status)
    M = len(T)
    if( hascolnames == 1 ):
        if( M != Mc ):
            print("error: writetable: wrong number of colnames")
            return(status)
    for i in range(M):
        if isinstance(T[i], list):
            Ni = len(T[i])
            isnum[i]=0
        elif isinstance(T[i], np.ndarray):
            S = np.shape(T[i]) 
        
            if ( len(S) != 2 ):
                print("error: writetable: T[%d] not a list or an Nx1 array" % (i,) )           
                return(status)
            Ni = S[0]
            isnum[i]=1
        else:
            print("error: writetable: T not a list or an Nx1 array")
            return(status)
        
        if( i==0 ):
            N = Ni
        elif( N != Ni ):
            print("error: writetable: T not a list or an Nx1 array")
            return(status)
        
        
    # open file
    try:
        fdr = open(pathname,"w")
    except:
        print("error: writetable: can open %s" % (pathname,) )
    return(T,N,M,isnum,colname,status)

    if( hascolnames ):
        line = ""
        for i in range(M):
            line = line+colname[i]
            if( i != (M-1) ):
                line = line+delimiter
            else:
                line = line+'\n'
                
            fdr.write(line)
    
    for i in range(N):
        line = ""
        for j in range(M):
            if( isnum[j] ):
                sval = "%f" % ((T[j])[i,0],)
                line = line + sval
            else:
                line = line + (T[j])[i]
            if( j != (M-1) ):
                line = line+delimiter
            else:
                line = line+'\n'
        fdr.write(line)
    fdr.close()
    status=1
    return(status)