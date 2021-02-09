# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:08:06 2020

@author: vrath
"""

import os
import sys
import numpy as np
from pyproj import Proj, transform
from scipy.ndimage import (
    gaussian_filter,
    laplace,
    convolve,
    gaussian_gradient_magnitude,
)
from scipy.linalg import norm
from sys import exit as error
import fnmatch

# from numba import jit
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon


def get_filelist(searchstr=["*"], searchpath="."):
    """
    generates filelist from path and unix wildcard list

    author: VR 11/20
    """

    filelist = fnmatch.filter(os.listdir(searchpath), "*")
    for sstr in searchstr:
        filelist = fnmatch.filter(filelist, sstr)
        print(filelist)

    return filelist


def growHeader(Header=None, Addstr=None, Out=False):
    """
    Grows header string
    author: vrath
    last changed: 2020/12/4
    """
    if Out:
        print("Old Header:")
        print(" \n".join(Header))

    # Header = np.array2string(Header)

    if Addstr is None:
        pass
    else:
        for line in Addstr:
            print(line)
            Header.append(line)

    if Out:
        print("New Header:")
        print(" \n".join(Header))

    return Header


def printHeader(Header):
    """
    Grows header string
    author: vrath
    last changed: 2020/12/4
    """
    print("Current Header:")
    print("\n".join(Header))


def proj_utm_to_latlon(utm_x, utm_y, utm_zone=32629):
    """
    transform utm to latlon, using pyproj
    Look for other EPSG at https://epsg.io/
    VR 11/20
    """
    prj_wgs = Proj(init="epsg:4326")
    prj_utm = Proj(init="epsg:" + str(utm_zone))
    longitude, latitude = transform(prj_utm, prj_wgs, utm_x, utm_y)
    return latitude, longitude


def proj_latlon_to_utm(longitude, latitude, utm_zone=32629):
    """
    transform latlon to utm , using pyproj
    Look for other EPSG at https://epsg.io/

    VR 11/20
    """
    prj_wgs = Proj(init="epsg:4326")
    prj_utm = Proj(init="epsg:" + str(utm_zone))
    utm_x, utm_y = transform(prj_wgs, prj_utm, latitude, longitude)
    return utm_x, utm_y


def proj_latlon_to_itm(longitude, latitude):
    """
    transform latlon to itm , using pyproj
    Look for other EPSG at https://epsg.io/

    VR 11/20
    """
    prj_wgs = Proj(init="epsg:4326")
    prj_itm = Proj(init="epsg:2157")
    itm_x, itm_y = transform(prj_wgs, prj_itm, latitude, longitude)
    return itm_x, itm_y


def proj_itm_to_latlon(itm_x, itm_y):
    """
    transform itm to latlon, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 11/20
    """
    prj_wgs = Proj(init="epsg:4326")
    prj_itm = Proj(init="epsg:2157")
    longitude, latitude = transform(prj_itm, prj_wgs, itm_x, itm_y)
    return latitude, longitude


def proj_itm_to_utm(itm_x, itm_y, utm_zone=32629):
    """
    transform itm to utm, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 11/20
    """
    prj_utm = Proj(init="epsg:" + str(utm_zone))
    prj_itm = Proj(init="epsg:2157")
    utm_x, utm_y = transform(prj_itm, prj_utm, itm_x, itm_y)
    return utm_x, utm_y


def proj_utm_to_itm(utm_x, utm_y, utm_zone=32629):
    """
    transform utm to itm, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 11/20
    """
    prj_utm = Proj(init="epsg:" + str(utm_zone))
    prj_itm = Proj(init="epsg:2157")
    itm_x, itm_y = transform(prj_utm, prj_itm, utm_x, utm_y)
    return itm_x, itm_y


def splitall(path):
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def shift(l, n):
    return l[n:] + l[:n]


def get_files(SearchString=None, SearchDirectory="."):
    """
    FileList = get_files(Filterstring) produces a list
    of files from a searchstring (allows wildcards)

    VR 11/20
    """
    FileList = fnmatch.filter(os.listdir(SearchDirectory), SearchString)

    return FileList


def get_header(file, headstr="# ", Out=True):
    """
    Get header lines from file
    VR 11/20

    """
    if Out:
        print("\nHeader:")
    header = []
    with open(file) as fd:
        for line in fd:
            if line.startswith(headstr):
                # line=line[2:]
                line = line.replace(headstr, "")
                header.append(line)
                if Out:
                    print(line[1:-2])
    header = header[0:-1]  # new_list = list(chain(a[0:2], [a[4]], a[6:]))
    if Out:
        print("\n")
    Header = "".join(header)
    print(Header)
    return Header


def unique(list, out=False):
    """
    find unique elements in list/array

    VR 9/20
    """

    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    if out:
        for x in unique_list:
            print(x)

    return unique_list


def strcount(keyword=None, fname=None):
    """
    count occurences of keyword in file
     Parameters
    ----------
    keywords : TYPE, optional
        DESCRIPTION. The default is None.
    fname : TYPE, optional
        DESCRIPTION. The default is None.

    VR 9/20
    """
    with open(fname, "r") as fin:
        return sum([1 for line in fin if keyword in line])
    # sum([1 for line in fin if keyword not in line])


def strdelete(keyword=None, fname_in=None, fname_out=None, out=True):
    """
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
    """
    nn = strcount(keyword, fname_in)

    if out:
        print(str(nn) + " occurances of <" + keyword + "> in " + fname_in)

    # if fname_out == None: fname_out= fname_in
    with open(fname_in, "r") as fin, open(fname_out, "w") as fou:
        for line in fin:
            if keyword not in line:
                fou.write(line)


def strreplace(key_in=None, key_out=None, fname_in=None, fname_out=None):
    """
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

    """

    with open(fname_in, "r") as fin, open(fname_out, "w") as fou:
        for line in fin:
            fou.write(line.replace(key_in, key_out))


def gen_grid_latlon(LatLimits=None, nLat=None, LonLimits=None, nLon=None, out=True):
    """
    Generates equidistant 1-d grids in latLong.

    VR 11/20
    """
    small = 0.000001
    # LonLimits = ( 6.275, 6.39)
    # nLon = 31
    LonStep = (LonLimits[1] - LonLimits[0]) / nLon
    Lon = np.arange(LonLimits[0], LonLimits[1] + small, LonStep)

    # LatLimits = (45.37,45.46)
    # nLat = 31
    LatStep = (LatLimits[1] - LatLimits[0]) / nLat
    Lat = np.arange(LatLimits[0], LatLimits[1] + small, LatStep)

    return Lat, Lon


def gen_grid_utm(XLimits=None, nX=None, YLimits=None, nY=None, out=True):
    """
    Generates equidistant 1-d grids in m.

    VR 11/20
    """

    small = 0.000001
    # LonLimits = ( 6.275, 6.39)
    # nLon = 31
    XStep = (XLimits[1] - XLimits[0]) / nX
    X = np.arange(XLimits[0], XLimits[1] + small, XStep)

    # LatLimits = (45.37,45.46)
    # nLat = 31
    YStep = (YLimits[1] - YLimits[0]) / nY
    Y = np.arange(YLimits[0], YLimits[1] + small, YStep)

    return X, Y


def choose_data_poly(Data=None, PolyPoints=None, Out=True):
    """
    Chooses polygon area from aempy data set, given
    PolyPoints = [[X1 Y1,...[XN YN]]. First and last points will
    be connected for closure.

    VR 11/20
    """
    if Data.size == 0:
        error("No Data given!")
    if not PolyPoints:
        error("No Rectangle given!")

    Ddims = np.shape(Data)
    if Out:
        print("data matrix input: " + str(Ddims))

    Poly = []
    for row in np.arange(Ddims[0] - 1):
        if point_inside_polygon(Data[row, 1], Data[row, 1], PolyPoints):
            Poly.append(Data[row, :])

    Poly = np.asarray(Poly, dtype=float)
    if Out:
        Ddims = np.shape(Poly)
        print("data matrix output: " + str(Ddims))

    return Poly


# potentially faster:
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

# lons_lats_vect = np.column_stack((lons_vect, lats_vect)) # Reshape coordinates
# polygon = Polygon(lons_lats_vect) # create polygon
# point = Point(y,x) # create point
# print(polygon.contains(point)) # check if polygon contains point
# print(point.within(polygon)) # check if a point is in the polygon

# @jit(nopython=True)


def point_inside_polygon(x, y, poly):
    """
    Determine if a point is inside a given polygon or not, where
    the Polygon is given as a list of (x,y) pairs.
    Returns True  when point (x,y) ins inside polygon poly, False otherwise

    """
    # @jit(nopython=True)
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def choose_data_rect(Data=None, Corners=None, Out=True):
    """
    Chooses rectangular area from aempy data set, giveb
    the left lower and right uper corners in m as [minX maxX minY maxY]

    """
    if Data.size == 0:
        error("No Data given!")
    if not Corners:
        error("No Rectangle given!")

    Ddims = np.shape(Data)
    if Out:
        print("data matrix input: " + str(Ddims))
    Rect = []
    for row in np.arange(Ddims[0] - 1):
        if (
            Data[row, 1] > Corners[0]
            and Data[row, 1] < Corners[1]
            and Data[row, 2] > Corners[2]
            and Data[row, 2] < Corners[3]
        ):
            Rect.append(Data[row, :])
    Rect = np.asarray(Rect, dtype=float)
    if Out:
        Ddims = np.shape(Rect)
        print("data matrix output: " + str(Ddims))

    return Rect


def proj_to_line(x, y, line):
    """
    Projects a point onto a line, where line is represented by two arbitrary
    points. as an array
    """
    #    http://www.vcskicks.com/code-snippet/point-projection.php
    #    private Point Project(Point line1, Point line2, Point toProject)
    # {
    #    double m = (double)(line2.Y - line1.Y) / (line2.X - line1.X);
    #    double b = (double)line1.Y - (m * line1.X);
    #
    #    double x = (m * toProject.Y + toProject.X - m * b) / (m * m + 1);
    #    double y = (m * m * toProject.Y + m * toProject.X + b) / (m * m + 1);
    #
    #    return new Point((int)x, (int)y);
    # }
    x1 = line[0, 0]
    x2 = line[1, 0]
    y1 = line[0, 1]
    y2 = line[1, 1]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    #
    xn = (m * y + x - m * b) / (m * m + 1.0)
    yn = (m * m * y + m * x + b) / (m * m + 1.0)
    #
    return xn, yn


# }
