#! /bin/bash
source /home/intel/oneapi/setvars.sh
# export HDF5_ROOT=$HOME/local/intel
# export PATH=$HDF5_ROOT/bin:$PATH
# export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH


num_mpi_proc=18
# num_hw_threads=4
#
# alpha=8.
# thresh=0.01
# niter=50
#
# # dist=0
# # distwght=''
# dist=1
# distwght=0.1
#
# cp control_template.dat TMP
# sed -i 's/num_threads/'${num_hw_threads}'/g' TMP
# sed -i 's/alpha/'${alpha}'/g' TMP
# sed -i 's/thresh/'${thresh}'/g' TMP
# sed -i 's/niter/'${niter}'/g' TMP
# sed -i 's/dist/'${dist}'/g' TMP
# sed -i 's/dstwght/'${distwght}'/g' TMP
# cp TMP control.dat
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
#
#export LD_LIBRARY_PATH=/home/vrath/Miniconda2024/envs/MKL2024/lib:$LD_LIBRARY_PATH
mpirun -np $num_mpi_proc /home/vrath/bin/$(uname -n)_femtic-dabic_152_nohdf5_release.x > abic.log
#
END=$(date "+%s")
#
echo "  "
# echo "wall time used " $((END-START)) "s"
echo "wall time used in h:"
echo "scale=2; (${END}-${START})/3600" | bc

