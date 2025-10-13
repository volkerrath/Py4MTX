import pyMT.data_structures as DS
import pyMT.IO as IO
import pyMT.utils as utils
import numpy as np
import rasterio # Not included in the pyMT isntall; you will have to manually install it
import os
from rasterio.transform import Affine
from scipy.interpolate import RegularGridInterpolator

# This function now defined as a method within Model objects
def average_conductance(model, idx_0, idx_1):
	conductance = np.zeros((model.nx, model.ny))
	for ii in range(idx_0, idx_1):
		conductance += model.zCS[ii] * (1 / model.vals[:, :, ii])
	return conductance

model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/swzFinish_lastIter.rho')
data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
out_tag = 'test_geotiff_lambert'
output_dir = 'E:/my_modules/pyMT/pyMT/scripts/test_outputs/'
projection = 'utm' # {lambert, UTM}. If UTM, specify the zone below.
utm_zone = '17' # Make sure this is the correct UTM zone for your area or it might not project correctly
rho_axes = ['X'] # Keep as X unless you have an anisotropic model
#Trim the model by x1, x2, y1, y2, z1, z2 cells in each direction
# Highly recommended if you have padding, as otherwise the interpolation will result in far too many cells
trim_pads = [15, 15, 15, 15, 15, 15]
# Can use one of these to set the first slice to conductance or depth-weighted resistivity
# Trim everything except this slice if you want to write it out.
# model.vals[:, :, 0] = model.conductance()
# model.vals[:, :, 0] = model.depth_weighted_rho()


# Check if the folder exists, if not, create it
if not os.path.isdir(output_dir):
	os.makedirs(output_dir)
# Set up the EPSG code based on the project
# Note that these codes have only been checked on Canadian data
if projection.lower() == 'lambert':
	epsg = 'EPSG:3979'
elif projection.lower() == 'utm':
	epsg = 'EPSG:326{}'.format(int(utm_zone))

# # Remove the pads so you don't have a million cells
if len(trim_pads) == 6:
	if trim_pads[0]:
		for ii in range(trim_pads[0]):
			model.dx_delete(0)
	if trim_pads[1]:
		for ii in range(trim_pads[1]):
			model.dx_delete(-1)
	if trim_pads[2]:
		for ii in range(trim_pads[2]):
			model.dy_delete(0)
	if trim_pads[3]:
		for ii in range(trim_pads[3]):
			model.dy_delete(-1)
	if trim_pads[4]:
		for ii in range(trim_pads[4]):
			model.dz_delete(0)
	if trim_pads[5]:
		for ii in range(trim_pads[5]):
			model.dz_delete(-1)

data.locations = data.get_locs(mode=projection)
model.origin = data.origin
model.project_model(system=projection, origin=data.origin, utm_zone=utm_zone)
# model.to_UTM()

x, y, z = (utils.edge2center(model.dy),
		   utils.edge2center(model.dx),
		   utils.edge2center(model.dz))
X, Y, Z =  np.meshgrid(x, y, z, indexing='ij', sparse=True)
# points = np.array((X.flatten(), Y.flatten())).T
# Z = np.array([[iz] * X.size for iz in model.dz])

xCS = np.min(model.xCS)
yCS = np.min(model.yCS)

x_reg, y_reg = (np.linspace(x[0], x[-1], int((x[-1] - x[0]) / yCS)),
				np.linspace(y[0], y[-1], int((y[-1] - y[0]) / xCS)))
qx, qy = np.meshgrid(x_reg, y_reg)


transform = Affine.translation(x[0], y[0]) * Affine.scale(np.min(model.yCS), np.min(model.xCS))

for kind in rho_axes:
	if kind == 'X':
		RGI = RegularGridInterpolator((x, y, z),
								  model.rho_x.transpose([1, 0, 2]), method='nearest')
	elif kind == 'Y':
		RGI = RegularGridInterpolator((x, y, z),
								  model.rho_y.transpose([1, 0, 2]), method='nearest')
	elif kind == 'X-Y':
		RGI = RegularGridInterpolator((x, y, z),
								  model.rho_x.transpose([1, 0, 2]) / model.rho_y.transpose([1, 0, 2]), method='nearest')
	for iz in range(model.nz):
		outfile = output_dir + out_tag + '_{:5.2f}km.tiff'.format(z[iz] / 1000)
		with rasterio.open(outfile,
							'w', driver='GTiff',
							height=y_reg.size, width=x_reg.size, count=1,
							dtype=np.float64,
							crs={'init': epsg}, # Lambert
							# crs={'init': 'EPSG:32617'}, # UTM zone (last 2 digits)
							transform=transform) as dst:
			dst.set_band_description(1, 'depth = {:5.2f} km'.format(z[iz]/1000))
			query_points = np.zeros((qx.size, 3))
			query_points[:, 0] = qx.flatten()
			query_points[:, 1] = qy.flatten()
			query_points[:, 2] = z[iz]
			vals = RGI(query_points)
			dst.write(np.log10(vals.reshape([qx.shape[0], qx.shape[1]])), 1)

# # Uncomment and shift indents for a single large geotiff
# with rasterio.open(outfile,
# 					'w', driver='GTiff',
# 					height=y_reg.size, width=x_reg.size, count=17, #model.nz,
# 					dtype=np.float64,
# 					# crs={'init': 'EPSG:3979'}, # Lambert
# 					crs={'init': 'EPSG:32617'}, # UTM zone (last 2 digits)
# 					transform=transform) as dst:
# 	for iz in range(17):
# 		# dst.update_tags(iz + 1, depth=int(z[iz]))
# 		dst.set_band_description(iz + 1, 'depth = {:5.2f} km'.format(z[iz]/1000))
# 		query_points = np.zeros((qx.size, 3))
# 		query_points[:, 0] = qx.flatten()
# 		query_points[:, 1] = qy.flatten()
# 		query_points[:, 2] = z[iz]
# 		vals = RGI(query_points)
# 		dst.write(np.log10(vals.reshape([qx.shape[0], qx.shape[1]])), iz + 1)

	
