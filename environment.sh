module use /g/data/v10/public/modules/modulefiles
#module load agdc-py3-env
#module load agdc-py3-env/20170327
module load agdc-py3-prod
#module load agdc-py3-prod/1.4.1
module load otps
module load java

export LC_ALL=en_AU.UTF-8
export LANG=en_AU.UTF-8

#export DATACUBE_CONFIG_PATH=/g/data/v10/public/modules/agdc-py3-futureprod/1.2.2/datacube.conf
#export PYTHONPATH="/home/547/bxb547/ga-repository/agdc_statistics:${PYTHONPATH}"
export PYTHONPATH="/g/data/u46/users/bxb547/ga-repo/agdc_statistics:${PYTHONPATH}"
unset PYTHONNOUSERSITE

