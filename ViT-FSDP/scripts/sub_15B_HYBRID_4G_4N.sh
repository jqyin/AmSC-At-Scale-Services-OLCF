#!/bin/bash -l
#SBATCH -J test.vit
#SBATCH -t 0:25:00
#SBATCH -N 4
#SBATCH -A stf218
#SBATCH --exclusive 
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/vit-%x-%j.out
#SBATCH -q debug

# env
source /lustre/orion/stf218/world-shared/atsaris/DEEPCAM_2022/new_env_sep2025_2/source_env.sh

CMD="python -u src/main_training.py \
     	       --model vitsmart \
     	       --model_size 15B \
     	       --strategy HYBRID_SHARD \
     	       --scaling_group_size 4 \
     	       --bs 32 \
     	       --image_size 224 \
"
echo $CMD

NUM_RANKS=`expr ${SLURM_NNODES} \* ${SLURM_NTASKS_PER_NODE}`
NUM_RANKS_PER_NODE=${SLURM_NTASKS_PER_NODE}


HOME=/tmp srun \
	  --nodes=${SLURM_NNODES} \
	  --ntasks=${NUM_RANKS} \
	  --ntasks-per-node=${NUM_RANKS_PER_NODE} \
	  -c7 \
	  --gpus-per-node=8 bash \
	  -c "source scripts/setup_fsdp.sh && $CMD"


