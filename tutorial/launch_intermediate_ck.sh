#!/bin/bash
#SBATCH -A VEN114
#SBATCH -J classification_fsdp
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 08:00:00
#SBATCH -p batch
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536



source ~/miniconda3/etc/profile.d/conda.sh

module load PrgEnv-gnu
module load rocm/6.2.4
module unload darshan-runtime
module unload libfabric

eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"

# conda activate /lustre/orion/lrn036/world-shared/xf9/torch26
conda activate /lustre/orion/proj-shared/ven114/ashwinaji/miniconda3/envs/sc25

#export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/junqi/climax/rccl-plugin-rocm6/lib/:/opt/rocm-6.2.0/lib:$LD_LIBRARY_PATH

## DDStore and GPTL Timer

module use -a /lustre/orion/world-shared/lrn036/jyc/frontier/sw/modulefiles
module load libfabric/1.22.0p

export MASTER_ADDR=$(hostname -i)

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH
export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD/src:$PYTHONPATH

export ORBIT_USE_DDSTORE=0 ## 1 (enabled) or 0 (disable)

export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6

#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_117m.yaml

#time srun -n $((SLURM_JOB_NUM_NODES*8)) \
#python ./intermediate_downscaling.py ../configs/interm_8m.yaml

mkdir -p logs/${SLURM_JOB_ID}

# (1a) Setup omnistat sampling environment
ml use /autofs/nccs-svm1_sw/crusher/amdsw/modules
ml omnistat-wrapper
export OMNISTAT_VICTORIA_DATADIR=/lustre/orion/${SLURM_JOB_ACCOUNT}/world-shared/omnistat/${SLURM_JOB_ID}
# (1b) Enable data collectors and polling (1 sec interval)
${OMNISTAT_WRAPPER} usermode --start --interval 1 | tee omnistat_start.log

# (2) Run the job
for FA in "CK" "SDPA" "default"; do
  export FA_ALGO=${FA}; srun -n $((SLURM_JOB_NUM_NODES*8)) \
  python ./tutorial/intermediate_downscaling.py ./configs/interm_8m.yaml \
  > logs/${SLURM_JOB_ID}/orbit-${SLURM_JOB_ID}-${FA}.out 2> logs/${SLURM_JOB_ID}/orbit-${SLURM_JOB_ID}-${FA}.er
done

# (3) Tear-down data collection and summarize results
${OMNISTAT_WRAPPER} usermode --stopexporters
${OMNISTAT_WRAPPER} query --job ${SLURM_JOB_ID} --interval 1 --pdf omnistat-${SLURM_JOB_ID}.pdf > omnistat-${SLURM_JOB_ID}.txt
${OMNISTAT_WRAPPER} usermode --stopserver

