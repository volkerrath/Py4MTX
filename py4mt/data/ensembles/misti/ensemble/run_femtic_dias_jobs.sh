#! /bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 DIR_PREFIX ITER1 [ITER2 ...]"
    echo "Example: $0 model_number 1 3 17"
    exit 1
fi

dir_prefix=$1
shift

iters=("$@")

source /home/vrath/intel/oneapi/setvars.sh --force
export HDF5_ROOT=$HOME/local/intel
export PATH=$HDF5_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH

num_mpi_proc=18
unset OMP_NUM_THREADS

echo "Run on:"
uname -n

base_dir=$(pwd)

for iter in "${iters[@]}"; do
    d="${dir_prefix}${iter}"

    if [ ! -d "$d" ]; then
        echo "Directory not found: $d"
        continue
    fi

    echo " "
    echo "============================================================"
    echo "Starting sequential FEMTIC run in: $d"
    echo "============================================================"

    START=$(date "+%s")

    cd "$base_dir/$d" || continue

    # This mpirun is intentionally foreground-only.
    # No '&' is used, so the loop waits here until this run finishes.
    mpirun -np "$num_mpi_proc" \
        /home/vrath/bin/kayz_femtic-dabic_152_nohdf5_release.x \
        > abic.log

    run_status=$?

    cd "$base_dir" || exit 1

    END=$(date "+%s")

    echo "wall time used in h for $d:"
    echo "scale=2; (${END}-${START})/3600" | bc

    if [ "$run_status" -eq 0 ]; then
        echo "Archiving $d"
        tar -czvf "${d}.tar.gz" "$d"
    else
        echo "mpirun failed in $d with status $run_status"
        echo "Skipping archive for $d"
    fi

    echo "Finished $d before starting next run."
done

echo " "
echo "All requested sequential runs finished."
