//-------------------------------------------------------------------------------------------------------
// InversionHDF5Writer.h
//
// Writes sensitivity matrix, sensitivity vector, resistivity vector, residual
// vector, and error vector to a single HDF5 file on the final inversion iteration.
//
// Activated by:  -D_WRITE_INVERSION_DATA_HDF5
//
// HDF5 file layout  (inversion_iter<N>.h5)  — written once, final iteration only:
//
//   /sensitivity_matrix   [nDataTotal × nModel]   double, gzip-6   raw Jacobian J
//   /sensitivity_vector   [nModel]                double, gzip-6   diag(J^T J)
//   /resistivity_vector   [nModel]                double, gzip-6   log10(rho) free blocks
//   /residual_vector      [nDataTotal]            double, gzip-6   (d_obs - d_cal)/sigma
//   /error_vector         [nDataTotal]            double, gzip-6   same as residual_vector
//   /attributes  (group)
//       num_data   int
//       num_model  int
//       iteration  int
//
// Only PE 0 calls write().
// Link with:  -lhdf5
//-------------------------------------------------------------------------------------------------------
#ifndef INVERSION_HDF5_WRITER_H
#define INVERSION_HDF5_WRITER_H

#ifdef _WRITE_INVERSION_DATA_HDF5

#include <string>

class InversionHDF5Writer {
public:
    // Write all datasets to fileName. Called once per run on the final iteration.
    static bool write( const std::string& fileName,
                       const double*      sensitivityMatrix,   // [nDataTotal × nModel] raw J
                       const double*      residualVector,      // [nDataTotal]
                       const double*      errorVector,         // [nDataTotal]
                       const double*      sensitivityVector,   // [nModel]  diag(J^T J)
                       const double*      resistivityVector,   // [nModel]  log10(rho)
                       int                nDataTotal,
                       int                nModel,
                       int                iterationNumber );

    // Build the canonical output file name for a given iteration.
    static std::string makeFileName( int iterationNumber );
};

#endif // _WRITE_INVERSION_DATA_HDF5
#endif // INVERSION_HDF5_WRITER_H
