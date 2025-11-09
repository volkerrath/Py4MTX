#! /bin/env bash
condi
conda activate MT

python3 -m numpy.f2py -c -m mt1d_aniso ./mt1d_aniso.f90
