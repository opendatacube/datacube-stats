#!/usr/bin/env bash
#
# Use PBSDSH to start a dask-scheduler and a bunch of dask-workers, then
# run the desired script to connect to them to run a job.
#

#set -eu

#env_script=${module_dest}/scripts/environment.sh
env_script=~/scripts/environment.sh
ppn=1
tpp=1
umask=0002

while [[ $# > 0 ]]
do
    key="$1"
    case $key in
    --help)
        echo Usage: $0 --env ${env_script} --umask ${umask} --ppn ${ppn} --tpp ${tpp} script args
        echo 
        echo Options:
        echo "  --ppn       Number of processors per node"
        echo "  --tpp       Number of threads per worker process"
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
SCHEDULER_ADDR=${SCHEDULER_NODE}:${SCHEDULER_PORT}

# Number of worker processes on Master Node
n0ppn=$(( $ppn < $NCPUS-2 ? $ppn : $NCPUS-2 ))
n0ppn=$(( $n0ppn > 0 ? $n0ppn : 1 ))

# Run logstash log aggregator
echo Launching Logstash Log aggregator
pbsdsh -n 0 -- /bin/bash -c "${init_env}; logstash --log.level=info -f ~/logstash.config"&
sleep 2s

echo Running Metricbeats on each node
#pbsdsh -- /bin/bash -c "${init_env}; export LOGSTASH_HOST=${SCHEDULER_NODE}; /g/data/u46/users/dra547/software/metricbeat-5.2.2-linux-x86_64/metricbeat -path.config ~" &


# Run Dask Scheduler
echo Launching Dask Scheduler
echo pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-scheduler --port $SCHEDULER_PORT"
pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-scheduler --port $SCHEDULER_PORT"&
sleep 5s

echo Starting Master Node Workers
echo pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --nprocs ${n0ppn} --nthreads ${tpp}"
pbsdsh -n 0 -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --nprocs ${n0ppn} --nthreads ${tpp}"&
sleep 0.5s

echo Starting Workers on Other Nodes
for ((i=NCPUS; i<PBS_NCPUS; i+=NCPUS)); do
  echo pbsdsh -n $i -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --nprocs ${ppn} --nthreads ${tpp}"
  pbsdsh -n $i -- /bin/bash -c "${init_env}; dask-worker $SCHEDULER_ADDR --nprocs ${ppn} --nthreads ${tpp}"&
  sleep 0.5s
done
sleep 5s

# Run main application on Master Node
echo "*** APPLICATION ***"
echo "${@/DSCHEDULER/${SCHEDULER_ADDR}}"

# Execute $@, replacing instances of 'DSCHDULER' with the scheduler address
"${@/DSCHEDULER/${SCHEDULER_ADDR}}"
