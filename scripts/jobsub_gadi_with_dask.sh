#!/bin/bash
#PBS -l storage=gdata/v10+gdata/r78+gdata/xu18+gdata/u46+scratch/r78
#PBS -P r78 
#PBS -q normal
#PBS -l walltime=5:00:00
#PBS -l mem=192GB
#PBS -l jobfs=2GB
#PBS -l ncpus=10
#PBS -l wd

module use /g/data/v10/public/modules/modulefiles --append
module load dea

QUERY_THREAD=2
MEM=192
NCPUS=10
JOBDIR=/scratch/r78/$LOGNAME/tmp

./organize_cluster.sh -q $QUERY_THREAD -c $NCPUS -m $MEM -j $JOBDIR

#run the job
datacube-stats -E c3-samples -vvv --query-workers $QUERY_THREAD --queue-size $(( QUERY_THREAD*1 )) --scheduler-file $JOBDIR/scheduler.json ndvi_climatology.yaml

#kill the scheduler
pbsdsh -n 0 -- \
bash -l -c "\
pgrep 'dask-scheduler' | xargs kill -9;\
pgrep 'dask-worker' | xargs kill -9;\
rm $JOBDIR/scheduler.json"

wait
