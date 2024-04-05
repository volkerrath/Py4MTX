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
#       format_version: "1.5"
#       jupytext_version: 1.11.3
# ---

"""
script to visualise rjmcmcmt based on Ross Brodie"s original matlab
plotting routines for rjmcmc inversion results.

CreationDate:   2017/10/17  -  Developer:      rakib.hassan@ga.gov.au

Revision History:

10/19  VR (Volker Rath)
    * adapted colorbar, sizes, and more
03/20  VR (Volker Rath)
    * minor improvements, less memory
03/22  VR more options, minor graphical improvements

04/22  VR log z scale does not work

"""
import os
import sys
import numpy as np
from mtpy.core.mt import MT
from mt_metadata import TF_XML

PY4MT_ROOT = os.environ["PY4MT_ROOT"]
myfilename = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in myfilename:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import plotrjmcmc as pmc
import util

PlotFmt = ".pdf" #".png"
RhoPlotLim = [1, 10000]
DepthPlotLim = 25000.
LogDepth = False
ColorMap ="rainbow"
#ColorMap ="viridis"

PdfC = True
if not ".pdf" in PlotFmt:
    PdfC = False
    print("No PDF catalog because no pdf output!")
OutStrng = "_edited_imp_rjmcmc"

DataOut = True
DataName= "test_results.dat"
WRef = False

# PdfCName  = "Limerick2022_results.pdf"
# edi_in_dir = r"/home/vrath/Limerick2022/reports/EDI_edited_Z/"
# results_in_dir =r"/home/vrath/Limerick2022/reports/EDI_edited_Z_results/" #Mar02/out_edited/"
# plots_dir = r"/home/vrath/Limerick2022/reports/Plots/"  #r"/home/vrath/Limerick2022/work/Mar02/output/"

PdfCName  = "Test_results.pdf"
edi_in_dir = r"/home/vrath/rjmcmc_mt/work/edi/"
results_in_dir =r"/home/vrath/rjmcmc_mt/work/results/" #Mar02/out_edited/"
# plots_dir = r"/home/vrath/Limerick2022/3D/plots_all20km/"  #r"/home/vrath/Limerick2022/work/Mar02/output/"
plots_dir = r"/home/vrath/rjmcmc_mt/work/plots/"  #r"/home/vrath/Limerick2022/work/Mar02/output/"

edi_files = []
files = os.listdir(edi_in_dir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)

result_files = []
files = os.listdir(results_in_dir)
for entry in files:
    if not entry.startswith("."):
        result_files.append(entry)

result_files = sorted(result_files)
nfiles = len(result_files)

if PdfC:
    pdf_list= []
    for filename in result_files:
        name, ext = os.path.splitext(filename)
        pdf_list.append(plots_dir+name+OutStrng+".pdf")



if not os.path.isdir(plots_dir):
    print(" File: %s does not exist, but will be created" % plots_dir)
    os.mkdir(plots_dir)



count = 0
for filename in result_files:
    count = count + 1
    print("\n")
    print(str(count) + " of " + str(nfiles))


    infile = results_in_dir+filename
    name, ext = os.path.splitext(filename)
    outfile = plots_dir+name +OutStrng + PlotFmt
    print(infile)
    print(outfile)

    r = pmc.Results(infile,
                    outfile,
                    plotSizeInches="11x8",
                    maxDepth=DepthPlotLim,
                    zLog=LogDepth,
                    colormap=ColorMap
                    )

    r.plot()

    if DataOut:
        name_edi, ext = os.path.splitext(filename)
        file_i = edi_in_dir + name_edi+".edi"
        mt = MT()
        mt.read(file_i)
        lat = mt.station_metadata.location.latitude
        lon = mt.station_metadata.location.longitude
        elev = mt.station_metadata.location.elevation

        name_result,_ = os.path.splitext(outfile)

        data_in = np.loadtxt(name_result+".dat")
        sd = np.shape(data_in)
        lon = np.ones_like(data_in[:,0]).reshape(sd[0],1)*lon
        lat = np.ones_like(data_in[:,0]).reshape(sd[0],1)*lat

        if WRef:
            elev = np.ones_like(data_in[:,0])*elev
            data_in[:,0]= elev.flatten() - data_in[:,0]

        data_out=np.append(lat,lon,axis = 1)
        data_out=np.append(data_out,data_in,axis = 1)
        header = name.split("_")[0]+"  lat, lon, depth, median, q10, q90, mean, mode"
        np.savetxt(name_result+".dat", data_out, delimiter="  ", header=header)

        sit = np.full_like(lat, name.split("_")[0], dtype=object)
        tmp = np.append(sit,data_out, axis=1)
        if count ==1:
            data_all = tmp
        else:
            data_all = np.append(data_all,tmp, axis=0)


if DataOut:
    header = "All data:  site, lat, lon, depth, median, q10, q90, mean, mode"
    fmt = "%s  %14.7f  %14.7f  %15.5f  %18.5e  %18.5e %18.5e  %18.5e  %18.5e"
    np.savetxt(plots_dir+DataName, data_all, delimiter="  ", header=header, fmt=fmt)

if PdfC:
    util.make_pdf_catalog(plots_dir, PdfList=pdf_list, FileName=plots_dir+PdfCName)
