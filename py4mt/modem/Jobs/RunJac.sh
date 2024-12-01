#! /bin/bash
#OAR --name Sabancaya_Jacobian_P
#OAR --project mt-geothest
#OAR -l /nodes=4,walltime=36:00:00
#OAR -t heterogeneous

source /applis/site/guix-start.sh
refresh_guix gnu_compiler

######################################################################################################################################################################
# mpirun -np `cat $OAR_FILE_NODES|wc -l` --machinefile $OAR_NODE_FILE -mca plm_rsh_agent "oarsh"  EXECUTABLE -J MODEL_FILE DATA_FILE JACOBIAN  FWD_FILE  > OUTPUT_FILE
######################################################################################################################################################################

mpirun -np `cat $OAR_FILE_NODES|wc -l` --machinefile $OAR_NODE_FILE -mca plm_rsh_agent "oarsh"  /home/sbyrd/bin/gMod3DMTJ_MKL.x -J SABA8_P_Alpha03_prior20_NLCG_030.rho SABA8_P.dat /home/sbyrd/silenus/Sabancaya/SABA8_P.jac  SABA.fwd  > SABA8_P.out
