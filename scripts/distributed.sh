#!/usr/bin/env bash
#
# Use PBSDSH to start a dask-scheduler and a bunch of dask-workers, then
# run the desired script to connect to them to run a job.
#

set -eu

env_script=${module_dest}/scripts/environment.sh
ppn=1
tpp=1
bokeh_opt="--no-bokeh"
umask=0002

while [[ $# > 0 ]]
do
    key="$1"
    case $key in
    --help)
        echo Usage: $0 --env ${env_script} --umask ${umask} --ppn ${ppn} --tpp ${tpp} --bokeh script args
        exit 0
        ;;
    --env)
        env_script="$2"
        shift
        ;;
    --umask)
        umask="$2"
        shift
        ;;
    --ppn)
        ppn="$2"
        shift
        ;;
    --tpp)
        tpp="$2"
        shift
        ;;
    --bokeh)
        bokeh_opt="--bokeh"
        ;;
    *)
    break
    ;;
    esac
shift
done

init_env="umask ${umask}; source /etc/bashrc; source ${env_script}"

echo "*** ENVIRONMENT ***"
cat ${env_script}

eval ${init_env}

SCHEDULER_NODE=`sed '1q;d' $PBS_NODEFILE`
SCHEDULER_PORT=`shuf -i 2000-65000 -n 1`
SCHEDULER_ADDR=$SCHEDULER_NODE:$SCHEDULER_PORT

# Number of worker processes on Master Node
n0ppn=$(( $ppn < $NCPUS-2 ? $ppn : $NCPUS-2 ))
n0ppn=$(( $n0ppn > 0 ? $n0ppn : 1 ))

# Run Dask Scheduler
echo pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-scheduler --port $SCHEDULER_PORT ${bokeh_opt}"
pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-scheduler --port $SCHEDULER_PORT ${bokeh_opt}"&
sleep 5s

# Start Master Node Workers
echo pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --no-nanny --nprocs ${n0ppn} --nthreads ${tpp}"
pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --no-nanny --nprocs ${n0ppn} --nthreads ${tpp}"&
sleep 0.5s

# Start Workers on Other Nodes
for ((i=NCPUS; i<PBS_NCPUS; i+=NCPUS)); do
  echo pbsdsh -n $i -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --no-nanny --nprocs ${ppn} --nthreads ${tpp}"
  pbsdsh -n $i -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --no-nanny --nprocs ${ppn} --nthreads ${tpp}"&
  sleep 0.5s
done
sleep 5s

# Run main application on Master Node
echo "*** APPLICATION ***"
echo "${@/DSCHEDULER/${SCHEDULER_ADDR}}"

# Execute $@, replacing instances of 'DSCHDULER' with the scheduler address
"${@/DSCHEDULER/${SCHEDULER_ADDR}}"
