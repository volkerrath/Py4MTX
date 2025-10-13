import pyMT.data_structures as DS
from pyMT.IO import verify_input
import os
from pyMT.WSExceptions import WSFileError
from pyMT.GUI_common.classes import FileInputParser
from pyMT.utils import project, edge2center
import pyproj
import numpy as np

def transform_locations(raw_data, original_zone, UTM):
    # raw_data.locations = raw_data.get_locs(mode='latlong')
    if UTM.lower() == 'lam':
        print("Reminder: Current Lambert Transformation is set to EPSG3978")
        # transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3978')
        transformer = pyproj.Transformer.from_crs(original_zone, 'epsg:3978')
        if original_zone == 'epsg:4326':
            for ii, (lat, lon) in enumerate(raw_data.locations):
                x, y = transformer.transform(lat, lon)
                raw_data.locations[ii, :] = y, x
        else:
            for ii, (lat, lon) in enumerate(raw_data.locations):
                x, y = transformer.transform(lon, lat)
                raw_data.locations[ii, :] = y, x

def transform_model(model, original_zone):
    transformer = pyproj.Transformer.from_crs(original_zone, 'epsg:3978')
    cx = edge2center(model.dx)
    cy = edge2center(model.dy)
    X = np.tile(cx, model.ny).flatten()
    Y = np.tile(cy, [model.nx, 1]).T.flatten()
    new_x, new_y = [], []
    for ii, (xx, yy) in enumerate(zip(X, Y)):
        x, y = transformer.transform(yy, xx)
        new_x.append(y)
        new_y.append(x)
    return new_x, new_y


files_dict = FileInputParser.read_pystart('E:/phd/NextCloud/data/Regions/MetalEarth/wst/pywst_finalists.pymt')
# files_dict = FileInputParser.read_pystart('E:/phd/NextCloud/data/Regions/MetalEarth/sturgeon/pywst2stu.pymt')
base_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/'
write_data = 1
write_model = 0 
for data_set in files_dict.keys():
    # if data_set in ['ZK_cull', 'ZK_cull-s2', 'with_usarray', 'cull_crust-s2', 'cull_crust']:
    # if data_set not in ['ZK_cull', 'ZK_cull-s2']:
    if data_set not in ['with_usarray']:
        continue
    print(data_set)
    data_file = files_dict[data_set]['list']
    model_file = files_dict[data_set]['model']
    if not os.path.isabs(data_file):
        data_file = base_path + data_file
    if not os.path.isabs(model_file):
        model_file = base_path + model_file
# data_file = 'E:/phd/NextCloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/sorted_cull1b.lst'
# data_file = 'E:/phd/NextCloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/line2a_plus.lst'
# model_file = 'E:/phd/NextCloud/data/Regions/snorcle/line2a_plus/line2a-0p2_all_lastIter.rho'
    # UTM = 'lam'
    # outfile = 'E:/phd/NextCloud/data/Regions/snorcle/vtk_files/models/line2a/line2a_all_SG.vtk'
    outfile = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/vtk_files/models/{}_lambert_model.vtk'.format(data_set)
    data_out = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/vtk_files/{}_lambert_sites.vtk'.format(data_set)
    # original_zone = 'epsg:26909'# Zone 9
    # original_zone = 'epsg:32614'# Zone 14
    original_zone = 'epsg:32615'# Zone 15
    # original_zone = 'epsg:32616'# Zone 16
    # original_zone = 'epsg:32617'# Zone 17
    # original_zone = 'epsg:4326'
    projection = 'EPSG3978'
    data = DS.RawData(data_file)
    model = DS.Model(model_file)
    
    # for ii in range(15):
    #     model.dx_delete(0)
    #     model.dy_delete(0)
    #     model.dz_delete(0)
    #     model.dx_delete(-1)
    #     model.dy_delete(-1)
    #     model.dz_delete(-1)



    version = '# vtk DataFile Version 4.2\n'
    #################################################
    # # Hack segment to convert the models that were already in Lambert
    # data.locations = data.get_locs(mode='latlong')
    # transform_locations(data, original_zone=original_zone, UTM='lam')
    # model.origin = data.origin
    # model.to_UTM()
    # new_x, new_y = edge2center(model.dx), edge2center(model.dy)
    # new_x = np.tile(new_x, model.ny).flatten()
    # new_y = np.tile(new_y, [model.nx, 1]).T.flatten()
    ##################################################
    model.origin = data.origin
    model.to_UTM()
    new_x, new_y = transform_model(model, original_zone)
    transform_locations(data, original_zone=original_zone, UTM='lam')

    NX, NY, NZ = model.nx, model.ny, model.nz
    NP = NX*NY*NZ
    # Structured Grid
    if write_model:
        with open(outfile, 'w') as f:
            f.write(version)
            f.write('Projection: {} \n'.format(projection))
            f.write('ASCII\n')
            f.write('DATASET STRUCTURED_GRID\n')
            f.write('DIMENSIONS {} {} {} \n'.format(NX, NY, NZ))
            f.write('POINTS {} float\n'.format(NP))
            for iz in range(NZ):
                cc = 0
                for ix in range(NX):
                    for iy in range(NY):
                        f.write('{} {} {}\n'.format(new_y[cc], new_x[cc], -model.dz[iz]))
                        cc += 1
            f.write('POINT_DATA {}\n'.format(NP))
            f.write('SCALARS Resistivity float\n')
            f.write('LOOKUP_TABLE default\n')
            for iz in range(NZ):
                for iy in range(NY):
                    for ix in range(NX):
                        f.write('{}\n'.format(model.vals[ix, iy, iz]))
    if write_data:
        data.to_vtk(outfile=data_out, UTM='lambert', origin=(0,0))
    # # Point Cloud
    # with open(outfile, 'w') as f:
    #     f.write(version)
    #     f.write('Projection: {} \n'.format(projection))
    #     f.write('ASCII\n')
    #     f.write('DATASET POLYDATA\n')
    #     # f.write('DIMENSIONS {} {} {} \n'.format(ns, ns, 1))
    #     f.write('POINTS {} float\n'.format(NP))
    #     for iz in range(NZ):
    #         cc = 0
    #         for ix in range(NX):
    #             for iy in range(NY):
    #                 f.write('{} {} {}\n'.format(new_y[cc], new_x[cc], -model.dz[iz]))
    #                 cc += 1
    #     f.write('POINT_DATA {}\n'.format(NP))
    #     f.write('SCALARS dummy float\n')
    #     f.write('LOOKUP_TABLE default\n')
    #     for iz in range(NZ):
    #         for iy in range(NY):
    #             for ix in range(NX):
    #                 f.write('{}\n'.format(model.vals[ix, iy, iz]))
