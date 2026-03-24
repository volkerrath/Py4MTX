#! /bin/bash
#OAR --name rot_camila
#OAR --project mt-geothest
####OAR -l /nodes=1,walltime=16:0:00
#OAR -l /core=192,walltime=28:0:00
#OAR -t fat
#OAR --notify mail:svetlana.byrdina@univ-smb.fr
####OAR --notify mail:vrath@cp.dias.ie

module load compiler/latest mkl/latest mpi/latest
##module load compiler/2023.1.0 compiler/2023.1.0 mpi/2021.9.0


ulimit -s unlimited
unset OMP_NUM_THREADS



START=$(date "+%s")


mpirun -np 192 /home/sbyrd/bin/ifxMod3DMT_ANI3.x -I NLCG rot_40_jack.zani rot_40_jack_10xy.dat ann_06.inv ann.fwd rot_40_jack_04.cov rot_40_50_jack_prior_best_iso_NLCG_001.prm >annecy04.log

#mpiexec.hydra --hostfile $OAR_NODEFILE --prefix $OPENMPI_PATH -x $LD_LIBRARY_PATH --mca plm_rsh_agent "oarsh" --mca pml ucx --mca btl ^tcp,openib,uct -x UCX_TLS=shm,self,cuda_copy,dc,rc,ud,gdr_copy,tcp /home/sbyrd/bin/ifxMod3DMT_ANI3.x -I NLCG ann50_ZT2jack_Aplha04_NLCG_015_P3.zani jack_Z5.dat ann_04.inv ann.fwd ann_04.cov  >annecy04.log

#-J  ModEM_NLCG_026.rho ann25_Z_04.dat ann26.jac annecy.fwd annecy25_02.cov > annecy_jac.out


END=$(date "+%s")
echo "  "
echo "wall time used in h:"
echo "scale=2; (${END}-${START})/3600" | bc
