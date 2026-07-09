#! /bin/bash
#OAR --name FEMTIC_Misti-Ensemble
#OAR --project mt-geothest
#OAR -l /nodes=1, walltime=16:0:00
###OAR -l /core=72, walltime=16:0:00
#OAR -t fat
#OAR --notify mail:svetlana.byrdina@univ-smb.fr
#OAR --notify mail:vrath@cp.dias.ie

# source /softs/intel/2025/oneapi/setvars.sh
source /softs/intel/2023.1.0/oneapi/setvars.sh

ulimit -s unlimited
unset OMP_NUM_THREADS


num_mpi_proc=18
echo "number of mpi processes:" ${num_mpi_proc}

echo "  "
echo "Run on:  "
uname -n

cat control.dat

START=$(date "+%s")

# mpirun -np $num_mpi_proc /home/sbyrd/bin/femtic_43_2025_release.x> femtic.log
mpirun -np $num_mpi_proc /home/sbyrd/bin/femtic_43_2023_release.x> femtic.log

END=$(date "+%s")
echo "  "
echo "wall time used in h:"
echo "scale=2; (${END}-${START})/3600" | bc

