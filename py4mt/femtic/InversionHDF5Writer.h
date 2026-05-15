//-------------------------------------------------------------------------------------------------------
// InversionHDF5Writer.h
//
// Writes the assembled sensitivity matrix, residual vector, and error vector
// to a single HDF5 file at the end of each inversion iteration.
//
// Activated by:  -D_WRITE_INVERSION_DATA_HDF5
//
// Behaviour:
//   Every iteration  →  /residual_vector and /error_vector are written.
//   Final iteration  →  /sensitivity_matrix is written in addition.
//   "Final" means convergence reached OR maximum iteration count hit,
//   determined by the writeSensitivity flag passed by the caller.
//
// HDF5 file layout  (inversion_iter<N>.h5):
//
//   /residual_vector      [nDataTotal]            double, gzip-6   (every iter)
//   /error_vector         [nDataTotal]            double, gzip-6   (every iter)
//   /sensitivity_matrix   [nDataTotal × nModel]   double, gzip-6   (final iter only)
//       Raw Jacobian J (unweighted).
//   /sensitivity_vector   [nModel]                double, gzip-6   (final iter only)
//       diag( J^T Cd^{-1} J ) — sum of squared error-weighted Jacobian columns;
//       one scalar per free model parameter.
//   /resistivity_vector   [nModel]                double, gzip-6   (final iter only)
//       log10(resistivity) of free (non-fixed) blocks in inversion order.
//   /attributes  (group)
//       num_data             int
//       num_model            int
//       iteration            int
//       sensitivity_written  int  (1 = yes, 0 = no)
//
// Only PE 0 calls write().
//
// Link with:  -lhdf5
//-------------------------------------------------------------------------------------------------------
#ifndef INVERSION_HDF5_WRITER_H
#define INVERSION_HDF5_WRITER_H

#ifdef _WRITE_INVERSION_DATA_HDF5

#include <string>

class InversionHDF5Writer {
public:
    // Write residual and error vectors every iteration; sensitivity matrix only
    // when writeSensitivity == true.
    //
    //  fileName          – output path, e.g. "inversion_iter3.h5"
    //  sensitivityMatrix – row-major [nDataTotal × nModel]; ignored when
    //                      writeSensitivity == false (may be NULL)
    //  residualVector    – [nDataTotal]  normalised residuals  (d_obs - d_cal) / sigma
    //  errorVector       – [nDataTotal]  reciprocal std-dev weights  1/sigma
    //  nDataTotal        – number of data rows
    //  nModel            – number of model parameters (columns)
    //  iterationNumber   – stored as a scalar attribute
    //  writeSensitivity  – true on final iteration (converged or max iter reached)
    //
    // Write all datasets.
    //
    //  sensitivityMatrix – row-major [nDataTotal × nModel] raw J; ignored when
    //                      writeSensitivity == false (may be NULL)
    //  residualVector    – [nDataTotal]  normalised residuals
    //  errorVector       – [nDataTotal]  1/sigma weights (Cd^{-1/2} diagonal)
    //  sensitivityVector – [nModel]  diag( J^T Cd^{-1} J )
    //                      i.e. sum of squared error-weighted Jacobian columns;
    //                      ignored when writeSensitivity == false (may be NULL)
    //  resistivityVector – [nModel]  log10(resistivity) of free blocks;
    //                      ignored when writeSensitivity == false (may be NULL)
    //  nDataTotal        – number of data rows
    //  nModel            – number of free model parameters
    //  iterationNumber   – stored as a scalar attribute
    //  writeSensitivity  – true on final iteration (converged or max iter reached)
    //
    // Returns true on success.
    static bool write( const std::string& fileName,
                       const double*      sensitivityMatrix,
                       const double*      residualVector,
                       const double*      errorVector,
                       const double*      sensitivityVector,
                       const double*      resistivityVector,
                       int                nDataTotal,
                       int                nModel,
                       int                iterationNumber,
                       bool               writeSensitivity );

    // Build the canonical output file name for a given iteration.
    static std::string makeFileName( int iterationNumber );
};

#endif // _WRITE_INVERSION_DATA_HDF5
#endif // INVERSION_HDF5_WRITER_H
