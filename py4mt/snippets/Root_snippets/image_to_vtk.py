# from matplotlib.pyplot import imread
from PIL import Image
import numpy as np


# file_name = 'E:/phd/NextCloud/data/ArcMap/Golden Triangle/Lambert Images/terranes.tif'
# outfile = 'E:/phd/NextCloud/data/ArcMap/Golden Triangle/Lambert Images/vtks/terranes.vtk'
file_name = 'E:/phd/NextCloud/data/ArcMap/WST/WS_outlines_wMCR.tif'
outfile = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/vtk_files/WS_outlines_wMCR.vtk'
projection = 'EPSG3978'
is_banded = 1 # If the raster is banded colours or just values
im = Image.open(file_name)
im = np.array(im).T
if is_banded:
    vals = np.zeros((im.shape[1], im.shape[2]))
    for ix in range(im.shape[1]):
        for iy in range(im.shape[2]):
            vals[ix, iy] = np.dot(im[:, ix, iy], [0.2989, 0.587, 0.114])
    im = vals
im[im<250] = 0 # Hack for binary images (line drawings)
NX = im.shape[0]
NY = im.shape[1]
with open(file_name[:-3] + 'tfw', 'r') as f:
    xsize = float(f.readline())
    dummy = f.readline()
    dummy = f.readline()
    ysize = 1 * float(f.readline())
    x1 = float(f.readline())
    y2 = float(f.readline())
x2 = x1 + xsize * NX
y1 = y2 + ysize * NY
x_locs = np.linspace(x1, x2, NX)
y_locs = np.linspace(y1, y2, NY)

version = '# vtk DataFile Version 3.0\n'
if '.vtk' not in outfile:
    outfile = ''.join([outfile, '.vtk'])

# with open(outfile, 'w') as f:
#     f.write(version)
#     f.write('{}   Projection: {} \n'.format(modname, projection))
#     f.write('ASCII\n')
#     f.write('DATASET RECTILINEAR_GRID\n')
#     f.write('DIMENSIONS {} {} {}\n'.format(NX + 1, NY + 1, 1))
#     for dim in ('x', 'y', 'z'):
#         f.write('{}_COORDINATES {} float\n'.format(dim.upper(),
#                                                    len(getattr(values, ''.join(['_d', dim])))))
#         gridlines = getattr(values, ''.join(['_d', dim]))
#         for edge in gridlines:
#             f.write('{} '.format(str(edge)))
#         f.write('\n')
#         # for ii in range(getattr(values, ''.join(['n', dim]))):
#         #     midpoint = (gridlines[ii] + gridlines[ii + 1]) / 2
#         #     f.write('{} '.format(str(midpoint)))
#         # f.write('\n')
#     f.write('POINT_DATA {}\n'.format((NX + 1) * (NY + 1) * (NZ + 1)))

#     for ii, value in enumerate(scalars):

#         f.write('SCALARS {} float\n'.format(value))
#         f.write('LOOKUP_TABLE default\n')
#         # print(len())
#         for iz in range(NZ + 1):
#             for iy in range(NY + 1):
#                 for ix in range(NX + 1):
#                         xx = min([ix, NX - 1])
#                         yy = min([iy, NY - 1])
#                         zz = min([iz, NZ - 1])
#                         f.write('{}\n'.format(to_write[ii].vals[xx, yy, zz]))
NP = np.sum(im < 1e10)
# NP = np.sum(im > 0)
with open(outfile, 'w') as f:
    f.write(version)
    f.write('Projection: {} \n'.format(projection))
    f.write('ASCII\n')
    f.write('DATASET POLYDATA\n')
    # f.write('DATASET STRUCTURED_GRID\n')
    # f.write('DIMENSIONS {} {} {} \n'.format(NX, NY, 1))
    f.write('POINTS {} float\n'.format(NP))
    for ix in range(NX):
        for iy in range(NY):
            val = im[ix, NY-1-iy]
            # if val.size 
            if val < 1e10:
            # if val > 0:
                # f.write('{} {} {}\n'.format(x_locs[ix], y_locs[iy], -val * 1000)) # For LAB / Moho
                # f.write('{} {} {}\n'.format(x_locs[ix], y_locs[iy], val)) # For DEM
                f.write('{} {} {}\n'.format(x_locs[ix], y_locs[iy], 0)) # For basemap
    f.write('POINT_DATA {}\n'.format(NP))
    f.write('SCALARS dummy float\n')
    f.write('LOOKUP_TABLE default\n')
    for ix in range(NX):
        for iy in range(NY):
            val = im[ix, NY-1-iy]
            if val < 1e10:
            # if val > 0:
                # f.write('{}\n'.format(-im[ix, NY-1-iy] * 1000)) # For LAB / Moho
                # f.write('{}\n'.format(im[ix, NY-1-iy]))  # For DEM
                f.write('{}\n'.format(val))  # For basemap
