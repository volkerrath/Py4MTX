#! /bin/bash
source /mnt/projects/geo/mt/oneapi/setvars.sh --force

export HDF5_ROOT=/mnt/projects/geo/mt/intel/hdf5_1.14/
export PATH=$HDF5_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH


num_mpi_proc=12

cat control.dat

unset OMP_NUM_THREADS


#
echo "  "
echo "Run on:  "
uname -n
#
START=$(date "+%s")
#
#
# # dynamical libraries
#n/
mpirun -np $num_mpi_proc /mnt/projects/geo/mt/bin/femtic-dabic_h5_results.x > LOG
#
END=$(date "+%s")
#
echo "  "
# echo "wall time used " $((END-START)) "s"
echo "wall time used in h:"
echo "scale=2; (${END}-${START})/3600" | bc

