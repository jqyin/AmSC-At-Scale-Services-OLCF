#!/bin/bash -l
#SBATCH -J test.vit
#SBATCH -t 0:05:00
#SBATCH -N 1
#SBATCH -A 
#SBATCH --exclusive 
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/vit-%x-%j.out
#SBATCH -q debug

# env
source source_env.sh

CMD="python -u src/main_training.py \
     	       --model vitsmart \
     	       --model_size 5B \
     	       --strategy FULL_SHARD \
     	       --scaling_group_size 0 \
     	       --use_ddp \
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


