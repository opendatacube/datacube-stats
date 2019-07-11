#!/bin/bash
#PBS -P u46 
#PBS -q express
#PBS -l walltime=1:00:00
#PBS -l mem=256GB
#PBS -l jobfs=1GB
#PBS -l ncpus=32
#PBS -l wd

NNODES=$(cat $PBS_NODEFILE | uniq | wc -l)
NCPUS=16
JOBDIR=$PWD
DASKWORKERDIR=$JOBDIR/dask-workers
DASKSCHEDULER=$JOBDIR/scheduler.json
if [ -s $DASKSCHEDULER ]
then
    rm $DASKSCHEDULER
fi 

if [ -d $DASKWORKERDIR ]
then
    rm -fr $DASKWORKERDIR
fi
mkdir $DASKWORKERDIR

#build a dask cluster
for i in $(seq 0 $(( NNODES-1 ))); do
    mkdir $DASKWORKERDIR/$i
    if [[ $i -eq 0 ]]
    then
        pbsdsh -n $i -- \
        bash -l -c "\
        source $HOME/.bashrc; cd $JOBDIR;\
        dask-scheduler --scheduler-file $DASKSCHEDULER --local-directory $DASKWORKERDIR --no-dashboard;"& 
        pbsdsh -n $(( i+1 )) -- \
        bash -l -c "\
        source $HOME/.bashrc; cd $JOBDIR;\
        dask-worker --scheduler-file $DASKSCHEDULER --nprocs $(( NCPUS-1 )) --local-directory $DASKWORKERDIR/$i  --nthreads 1 --memory-limit 8GB --no-dashboard"& 
    else
        pbsdsh -n $(( i*NCPUS )) -- \
        bash -l -c "\
        source $HOME/.bashrc; cd $JOBDIR;\
        dask-worker --scheduler-file $DASKSCHEDULER --nprocs $NCPUS --local-directory $DASKWORKERDIR/$i --nthreads 1 --memory-limit 8GB --no-dashboard"&
    fi
done;


#run the job
datacube-stats-raijin --tile-index-file landsat_tiles.txt --query-workers 10 --queue-size 18 fc_percentile_virtual.yaml

#kill the scheduler
pbsdsh -n 0 -- \
bash -l -c "\
pgrep 'dask-scheduler' | xargs kill -9;\
rm $JOBDIR/scheduler.json"

wait
