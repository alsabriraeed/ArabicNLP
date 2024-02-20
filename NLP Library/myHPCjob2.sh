#!/bin/bash
#PBS -S /bin/bash
#PBS -N pmiweights
#PBS -P P001564001
#PBS -q Economy
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=28
cd $PBS_O_WORKDIR
nprocs=`cat $PBS_NODEFILE | wc -l`
mpirun -np $nprocs python3 pmiweights.py

