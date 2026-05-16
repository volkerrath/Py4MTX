//-------------------------------------------------------------------------------------------------------
// InversionHDF5Writer.cpp
//
// Compiled only when -D_WRITE_INVERSION_DATA_HDF5 is defined.
// See InversionHDF5Writer.h for the full layout description.
//
// Link with:  -lhdf5
//-------------------------------------------------------------------------------------------------------
#ifdef _WRITE_INVERSION_DATA_HDF5

#include "InversionHDF5Writer.h"
#include "OutputFiles.h"

#include <hdf5.h>
#include <sstream>

// ---------------------------------------------------------------------------
// Internal helper – create one compressed dataset (1-D or 2-D double).
//   nRows   number of rows  (or elements for 1-D)
//   nCols   number of columns; pass 0 to create a 1-D dataset
// ---------------------------------------------------------------------------
static bool writeDoubleDataset( hid_t        fileID,
                                const char*  name,
                                const double* data,
                                hsize_t      nRows,
                                hsize_t      nCols )
{
    const int   rank    = (nCols > 0) ? 2 : 1;
    hsize_t     dims[2] = { nRows, nCols };

    hid_t space = H5Screate_simple( rank, dims, NULL );
    if( space < 0 ){
        OutputFiles::m_logFile << "HDF5 Error : H5Screate_simple failed for '"
                               << name << "'." << std::endl;
        return false;
    }

    // Chunked + gzip-6 property list
    hid_t plist = H5Pcreate( H5P_DATASET_CREATE );
    if( plist >= 0 ){
        hsize_t chunk[2];
        if( rank == 2 ){
            chunk[0] = (nRows > 128) ? static_cast<hsize_t>(128) : nRows;
            chunk[1] = nCols;
        } else {
            chunk[0] = (nRows > 4096) ? static_cast<hsize_t>(4096) : nRows;
        }
        H5Pset_chunk( plist, rank, chunk );
        H5Pset_deflate( plist, 6 );
    }

    hid_t dset = H5Dcreate2( fileID, name, H5T_NATIVE_DOUBLE, space,
                              H5P_DEFAULT, plist, H5P_DEFAULT );
    if( dset < 0 ){
        OutputFiles::m_logFile << "HDF5 Error : H5Dcreate2 failed for '"
                               << name << "'." << std::endl;
        H5Sclose( space );
        if( plist >= 0 ) H5Pclose( plist );
        return false;
    }

    herr_t err = H5Dwrite( dset, H5T_NATIVE_DOUBLE,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, data );
    H5Dclose( dset );
    H5Sclose( space );
    if( plist >= 0 ) H5Pclose( plist );

    if( err < 0 ){
        OutputFiles::m_logFile << "HDF5 Error : H5Dwrite failed for '"
                               << name << "'." << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Internal helper – write a scalar int attribute into an open group/file.
// ---------------------------------------------------------------------------
static void writeIntAttribute( hid_t loc, const char* name, int value )
{
    hid_t space = H5Screate( H5S_SCALAR );
    hid_t attr  = H5Acreate2( loc, name, H5T_NATIVE_INT,
                               space, H5P_DEFAULT, H5P_DEFAULT );
    if( attr >= 0 ){
        H5Awrite( attr, H5T_NATIVE_INT, &value );
        H5Aclose( attr );
    }
    H5Sclose( space );
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------
bool InversionHDF5Writer::write( const std::string& fileName,
                                 const double*      sensitivityMatrix,
                                 const double*      residualVector,
                                 const double*      errorVector,
                                 const double*      sensitivityVector,
                                 const double*      resistivityVector,
                                 int                nDataTotal,
                                 int                nModel,
                                 int                iterationNumber )
{
    OutputFiles::m_logFile << "# Writing inversion data to HDF5 : "
                           << fileName << std::endl;

    hid_t fileID = H5Fcreate( fileName.c_str(), H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT );
    if( fileID < 0 ){
        OutputFiles::m_logFile << "HDF5 Error : Cannot create '"
                               << fileName << "'." << std::endl;
        return false;
    }

    const hsize_t nd = static_cast<hsize_t>(nDataTotal);
    const hsize_t nm = static_cast<hsize_t>(nModel);

    bool ok = true;
    ok &= writeDoubleDataset( fileID, "residual_vector",    residualVector,   nd, 0  );
    ok &= writeDoubleDataset( fileID, "error_vector",       errorVector,      nd, 0  );
    ok &= writeDoubleDataset( fileID, "sensitivity_vector", sensitivityVector, nm, 0  );
    ok &= writeDoubleDataset( fileID, "resistivity_vector", resistivityVector, nm, 0  );
    ok &= writeDoubleDataset( fileID, "sensitivity_matrix", sensitivityMatrix, nd, nm );

    hid_t grp = H5Gcreate2( fileID, "attributes",
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    if( grp >= 0 ){
        writeIntAttribute( grp, "num_data",  nDataTotal );
        writeIntAttribute( grp, "num_model", nModel );
        writeIntAttribute( grp, "iteration", iterationNumber );
        H5Gclose( grp );
    } else {
        OutputFiles::m_logFile << "HDF5 Warning : Could not create /attributes group."
                               << std::endl;
    }

    H5Fclose( fileID );

    if( ok ){
        OutputFiles::m_logFile << "# HDF5 file written : " << fileName << std::endl;
    }
    return ok;
}

std::string InversionHDF5Writer::makeFileName( int iterationNumber )
{
    std::ostringstream oss;
    oss << "inversion_iter" << iterationNumber << ".h5";
    return oss.str();
}

#endif // _WRITE_INVERSION_DATA_HDF5
