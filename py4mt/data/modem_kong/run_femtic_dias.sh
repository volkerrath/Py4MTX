#! /bin/bash
source /opt/intel/oneapi/setvars.sh
num_mpi_proc=8
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
mpirun -np $num_mpi_proc /home/vrath/bin/tinney_ifemtic25_abic.x > abic.log
#
END=$(date "+%s")
#
echo "  "
# echo "wall time used " $((END-START)) "s"
echo "wall time used in h:"
echo "scale=2; (${END}-${START})/3600" | bc

mail -s "FEMTIC-DABIC" vrath@cp.dias.ie <<'EOF'
Job finished.
EOF
