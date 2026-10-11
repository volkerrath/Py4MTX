#! /bin/bash
#OAR --name FEMTIC_RTO_Annecy_orig-err
#OAR --project mt-geothest
#OAR -l /nodes=1, walltime=3:0:00
###OAR -t fat
###OAR --notify mail:svetlana.byrdina@univ-smb.fr
#OAR --notify mail:vrath@cp.dias.ie

source /softs/intel/2026/oneapi/setvars.sh

ulimit -s unlimited
unset OMP_NUM_THREADS


num_mpi_proc=24
echo "number of mpi processes:" ${num_mpi_proc}

echo "  "
echo "Run on:  "
uname -n

cat control.dat

START=$(date "+%s")

mpirun -np $num_mpi_proc  /home/sbyrd/bin/femtic-dabic_h5_results.x > LOG



END=$(date "+%s")
echo "  "
echo "wall time used in h:"
echo "scale=2; (${END}-${START})/3600" | bc

 
