#! /bin/bash
#OAR --name FEMTIC_DABIC_test_with_distortion
#OAR --project mt-geothest
#OAR -l /nodes=1, walltime=36:0:00
#OAR -t fat
#OAR --notify mail:svetlana.byrdina@univ-smb.fr
#OAR --notify mail:vrath@cp.dias.ie

# module load compiler-intel-llvm mkl mpi

#module load compiler/2023.1.0 mkl/2023.1.0 mpi/2021.9.0
# module load compiler/latest mkl/latest mpi/latest

module load gcc/12 openmpi/4.1.8/gcc-12.4.0 compiler-rt/2023.1.0 mkl/2023.1.0 ucx/1.18.1  oclfpga/2023.1.0 tbb/2021.9.0
ulimit -s unlimited
unset OMP_NUM_THREADS


num_mpi_proc=16
echo "number of mpi processes:" ${num_mpi_proc}

echo "  "
echo "Run on:  "
uname -n

cat control.dat

START=$(date "+%s")
# mpirun --hostfile $OAR_NODEFILE --prefix $OPENMPI_PATH -x LD_LIBRARY_PATH --mca plm_rsh_agent "oarsh" --mca pml ucx --mca btl ^tcp,openib,uct -x UCX_TLS=shm,self,cuda_copy,dc,rc,ud,gdr_copy,tcp /home/sbyrd/bin/ifemtic43_2023_kraken.x

mpirun -np $num_mpi_proc /home/sbyrd/bin/gfemtic-dabic-mkl23-kraken.x > femtic.log

END=$(date "+%s")
echo "  "
echo "wall time used in h:"
echo "scale=2; (${END}-${START})/3600" | bc

# mpiexec.hydra -genvall -f $OAR_NODE_FILE -bootstrap-exec oarsh -env I_MPI_DEBUG 5 -n $num_mpi_proc ./ring
