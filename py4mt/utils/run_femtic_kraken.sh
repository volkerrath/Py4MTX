#! /bin/bash
#OAR --name FEMTIC_DABIC_NP4
#OAR --project mt-geothest
###OAR -l /nodes=1, walltime=36:0:00
#OAR -l /core=96, walltime=12:0:00
###OAR -t fat
###OAR --notify mail:svetlana.byrdina@univ-smb.fr
#OAR --notify mail:vrath@cp.dias.ie

# module load gcc/12 openmpi/4.1.8/gcc-12.4.0 compiler-rt/2023.1.0 mkl/2023.1.0 ucx/1.18.1  oclfpga/2023.1.0

module load compiler/latest mpi/latest mkl/latest

ulimit -s unlimited
unset OMP_NUM_THREADS


num_mpi_proc=18
echo "number of mpi processes:" ${num_mpi_proc}

echo "  "
echo "Run on:  "
uname -n

cat control.dat

START=$(date "+%s")
# mpirun -np  $num_mpi_proc --prefix $OPENMPI_PATH -x LD_LIBRARY_PATH --mca plm_rsh_agent "oarsh" --mca pml ucx --mca btl ^tcp,openib,uct -x UCX_TLS=shm,self,dc,rc,ud, tcp /home/sbyrd/bin/gfemtic-dabic-mkl23-kraken.x > femtic.log

mpirun -np $num_mpi_proc /home/sbyrd/bin/kraken_femtic-dabic_15_debug.x > femtic.log

END=$(date "+%s")
echo "  "
echo "wall time used in h:"
echo "scale=2; (${END}-${START})/3600" | bc

