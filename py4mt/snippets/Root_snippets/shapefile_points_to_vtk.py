import itertools
import numpy as np
import pyvista as pv
import geopandas as gpd
#for windows users
from shapely import speedups
speedups.disable()

## Only points works right now...
input_file = 'E:/phd/NextCloud/data/ArcMap/Golden Triangle/Lambert Images/shp/major_faults_EPSG3978.shp'
output_file = 'E:/phd/NextCloud/data/ArcMap/Golden Triangle/Lambert Images/vtks/major_faults_EPSG3978.vtk'
#create geodataframes from all shapefiles
Df = gpd.read_file(input_file)
data_type = 'poly'
# For point type geometry
#create emtpy lists to collect point information
cellSec = []
cellTypeSec = []
pointSec = []

if data_type == 'points':
    # iterate over the points
    i = 0
    for index, valuesx in Df.iterrows():
        x, y, z = Df.loc[index].geometry.x, Df.loc[index].geometry.y, 0
        pointSec.append([x,y,z])
        cellTypeSec.append([1])
        cellSec.append([1,i])
        i+=1

    #convert list to numpy arrays
    cellArray = np.array(cellSec)
    cellTypeArray = np.array(cellTypeSec)
    pointArray = np.array(pointSec)

    #create the unstructured grid object
    pointUgrid = pv.PolyData(pointArray)

    #we can add some values to the point
    # pointUgrid.cell_arrays["Elev"] = 0

    #plot and save as vtk
    # pointUgrid.plot()
    pointUgrid.save(output_file,binary=False)

#############################################
elif data_type == 'lines':
    #create emtpy dict to store the partial unstructure grids
    lineTubes = {}
    cc = 0
    #iterate over the points
    for index, values in Df.iterrows():
        try:
            cellSec = []
            linePointSec = []

            #iterate over the geometry coords
            zipObject = zip(values.geometry.xy[0],values.geometry.xy[1],itertools.repeat(0))
            for linePoint in zipObject:
                linePointSec.append([linePoint[0],linePoint[1],linePoint[2]])

            #get the number of vertex from the line and create the cell sequence
            nPoints = len(list(Df.loc[index].geometry.coords))
            cellSec = [nPoints] + [i for i in range(nPoints)]

            #convert list to numpy arrays
            cellSecArray = np.array(cellSec)
            cellTypeArray = np.array([4])
            linePointArray = np.array(linePointSec)

            partialLineUgrid = pv.UnstructuredGrid(cellSecArray,cellTypeArray,linePointArray)   
            #we can add some values to the point
            # partialLineUgrid.cell_arrays["Elev"] = values.Elev
            lineTubes[str(cc)] = partialLineUgrid
            cc += 1
        except NotImplementedError:
            print('Skipping one...')
    #merge all tubes and export resulting vtk
    lineBlocks = pv.MultiBlock(lineTubes)
    lineGrid = lineBlocks.combine()
    lineGrid.save(output_file,binary=False)
    lineGrid.plot()

elif data_type == 'poly':
        #create emtpy dict to store the partial unstructure grids
    polyTubes = {}

    #iterate over the points
    for index, values in Df.iterrows():
        cellSec = []
        linePointSec = []

        #iterate over the geometry coords
        try:
            zipObject = zip(values.geometry.xy[0],values.geometry.xy[1],itertools.repeat(0))
        except (NotImplementedError, AttributeError):
            zipObject = zip(values.geometry.exterior.xy[0],
                            values.geometry.exterior.xy[1],
                            itertools.repeat(0))
        for linePoint in zipObject:
            linePointSec.append([linePoint[0],linePoint[1],linePoint[2]])

        #get the number of vertex from the line and create the cell sequence
        try:
            nPoints = len(list(Df.loc[index].geometry.xy[0]))
        except NotImplementedError:
            nPoints = len(list(Df.loc[index].geometry.exterior.coords))
        cellSec = [nPoints] + [i for i in range(nPoints)]

        #convert list to numpy arrays
        cellSecArray = np.array(cellSec)
        cellTypeArray = np.array([4])
        linePointArray = np.array(linePointSec)

        partialPolyUgrid = pv.UnstructuredGrid(cellSecArray,cellTypeArray,linePointArray)   
        #we can add some values to the point
        # partialPolyUgrid.cell_arrays["Elev"] = values.Elev
        #    partialPolyUgrid.save('../vtk/partiallakePoly.vtk',binary=False)
        polyTubes[str(index)] = partialPolyUgrid

    #merge all tubes and export resulting vtk
    polyBlocks = pv.MultiBlock(polyTubes)
    polyGrid = polyBlocks.combine()
    polyGrid.save(output_file,binary=False)
    polyGrid.plot()