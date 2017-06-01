module use /g/data/v10/public/modules/modulefiles
#module load agdc-py3-env
#module load agdc-py3-env/20170327
module load agdc-py3-prod
module load otps
module load java

export LC_ALL=en_AU.UTF-8
export LANG=en_AU.UTF-8

#export DATACUBE_CONFIG_PATH=/g/data/v10/public/modules/agdc-py3-futureprod/1.2.2/datacube.conf
#export PYTHONPATH="/home/547/dra547/.local/lib/python3.5/site-packages/:${PYTHONPATH}"
export PYTHONPATH="/home/547/bxb547/ga-repository/agdc_statistics:${PYTHONPATH}"
unset PYTHONNOUSERSITE

