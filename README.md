# AmSC At-Scale Services - OLCF
AI and simulation software stack, baseline environments, and approaches for running on Frontier. 

# Contents

- AI Stack 
  * [LLM](#llm) -- Junqi
  * [ViT](#vit) -- Aris
  * [GNN](#gnn) -- Max
- Simulation Stack
  * [VASP](#vasp) -- Max 
  * [LSMS](#lsms) -- Junqi
  * [CFD](#cfd) -- Ramki ? 
- Baseline Environment
  * [HydraGNN](#hydragnn) -- Max
  * [LORACX](#loracx) -- Ramki 
  * [LLM Pre-Training -- FORGE](#forge) -- Junqi
  * [LLM Fine-Tuning -- TorchTitan](#torchtitan) -- Emin 
  * [Matey](#matey) -- Junqi

## LLM  
We support LLM pre-training and fine-tuning with frameworks such as DeepSpeed, Megatron, TorchTitan, TorchTune, etc, as well as model serving with vLLM, ollama, etc. 
### PyTorch
Frontier supports most LLM training and inference software libraries, and the deployment can be via either native installation (e.g., pip install) or containter (i.e., Apptainer). E.g., for rocm/6.4.2, PyTorch can be installed 
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.4
```
or 
```bash
apptainer build pytorch_rocm642.sif docker://docker.io/rocm/pytorch:rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0
```

### FlashAttention 
#### Installation 
FA2 is supported on Frontier and the upstream [repo](https://github.com/Dao-AILab/flash-attention) can be pip installed
```bash
module load PrgEnv-gnu
module load rocm
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $WRKSPC/miniconda
export PATH=$WRKSPC/miniconda/bin:$PATH
conda create --prefix $WRKSPC/miniconda/envs/fa2-env -y
source $WRKSPC/miniconda/etc/profile.d/conda.sh
conda activate $WRKSPC/miniconda/envs/fa2-env
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
git checkout v2.8.3
pip install -e .
popd
```
For latest development, please try AMD's [fork](https://github.com/ROCm/flash-attention) 
#### Backend
Standalone FA: 
- CK (default)
- Triton

PyTorch `scaled_dot_product_attention` (SDPA):
- Math
- FA
- Efficient
  
PyTorch Flex attention 
#### Performance 
- For standalone FA, use latest rocm. The built against rocm/6.3 is 1.5x faster than that with rocm/6.1 for certain inputs
  ![FA2](FlashAttention/fa2.png)

- For PyTorch SDPA, use FA or Efficient backend  
 ![SDPA](FlashAttention/sdpa.png)

## Baseline Environment

### HydraGNN
An [installation script](HydraGNN/hydragnn_installation_bash_script_frontier.sh) for setup HydraGNN on Frontier is provided. 

### FORGE
#### Step 1: Initial Environment Setup
First, load the required programming environment and ROCm modules.
```bash
module  load  PrgEnv-gnu/8.6.0
module  load  miniforge3/23.11.0-0
module  load  rocm/6.4.1
module  load  craype-accel-amd-gfx90a
export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
```
#### Step 2: Create and Activate Conda Environment
Make sure to replace the the directory in following code. It's recommended to install it in a shared lustre directory (`/lustre/orion/`) to ensure sufficient storage space.
```bash
# Create the env
conda create -p /lustre/orion/{..env_dir..}/py312 python=3.12
# Activate the env
source activate /lustre/orion/{..your_dir..}/py312
```
#### Step 3: Install Core Dependencies
1.  **Install pytorch:**
    ```bash
    pip install ninja
    pip  install  torch==2.8.0  torchvision==0.23.0  torchaudio==2.8.0  --index-url  https://download.pytorch.org/whl/rocm6.4
    ```
2.  **Install mpi4py:**
    ```bash
    MPICC="cc -shared"  pip  install  --no-cache-dir  --no-binary=mpi4py  mpi4py
    ```
3.  **Install deepspeed (DeeperSpeed fork v3.0):**
    ```bash
    cd build
    git clone https://github.com/EleutherAI/DeeperSpeed -b v3.0
    cd DeeperSpeed
    DS_BUILD_FUSED_LAMB=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1  DS_BUILD_UTILS=1 pip install .
    ```
4.  **Install other packages:** Run the following command from `forge` directory.
    ```bash
    pip install -r requirements.txt
     ```
> ⚠️  Make sure the python and pip are from conda environment.
> To check, type `which python` or `which pip` in the terminal.
> This should point to your conda environment  and not to /usr/bin/python.
#### Step 4: Build and Configure AWS OFI RCCL Plugin
This plugin is necessary for efficient communication at scale.
1.  **Clone the repository and run autogen:**
    ```bash
    mkdir build
    cd build

    rocm_version=6.4.1
    libfabric_path=/opt/cray/libfabric/1.22.0
    cd /path/to/your/workspace
    git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
    cd aws-ofi-rccl
        ./autogen.sh

        export  LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
        PLUG_PREFIX=$PWD

        CC=hipcc  CFLAGS=-I/opt/rocm-$rocm_version/include  ./configure  \
        --with-libfabric=$libfabric_path  --with-rccl=/opt/rocm-$rocm_version  --enable-trace   \
        --prefix=$PLUG_PREFIX  --with-hip=/opt/rocm-$rocm_version  --with-      mpi=$MPICH_DIR

        make
        make  install
    ```
3.  **Add the plugin to your environment:**
    Export the path to the newly built library.

    ```bash
    export LD_LIBRARY_PATH=$PLUG_PREFIX/lib:$LD_LIBRARY_PATH
    ```
> ⚠️  Make sure the to match the LD_LIBRARY_PATH in `job.sb` as well.
#### Step 5: Fused Kernels:
This will compile the fused kernels.
Make sure you have `export CXX=/opt/cray/pe/gcc-native/14/bin/g++`.
 Run the following command from the `forge` directory.
```python
python
from megatron.fused_kernels import load
load()
```
>⚠️ Important
If it fails, before recompiling, delete all the files under `megatron/fused_kernels/build` and also delete all the hip files.

### TorchTitan 
Following are an example to use TorchTitan to finetune a domain-specific Llama-3.1 model on Frontier. 
#### Environment Setup
* Install PyTorch with ROCm 6.3:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3
```
* Clone the [repo](https://github.com/pytorch/torchtitan) and install the following packages:
```bash
pip install datasets torchdata tomli tensorboard sentencepiece tiktoken blobfile tabulate ninja
```
* Download the Llama-3.1-8B tokenizer:
```python 
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original" 
```
Install FlashAttention as above. 
#### Pretraining data
For example,
* [Zyda-2](https://huggingface.co/datasets/Zyphra/Zyda-2), which is itself a cross-deduplicated and filtered combination of DCLM (3.3T), FineWeb-Edu (1.3T), Dolma (0.2T), Zyda (0.2T).
* Stack-2: the [`the-stack-v2-train-smol-ids`](https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids) subset (525B).
* [`FineMath`](https://huggingface.co/datasets/HuggingFaceTB/finemath): the `finemath-3plus` subset (34B).
#### Data loading strategy
The data loading strategy is currently as follows (implemented [here](https://github.com/eminorhan/frontier-torchtitan/blob/master/torchtitan/datasets/hf_datasets.py)):
* load individual component datasets in streaming mode (as iterable datasets)
* interleave the component datasets using `ds.interleave_datasets()`
* shuffle the combined dataset with a large buffer size (`buffer_size=100000`) and a globally shared random seed
* split the dataset across `dp` (data-parallel) ranks using `ds.split_dataset_by_node()`

The shuffle is performed once at the beginning of each training session with a fresh global random shuffling seed (due to job runtime limits on Frontier, each session takes 24 hours at most after which we checkpoint and restart again). The shuffle operation shuffles the dataset shards as well as the rows in the buffer and the large buffer size ensures that all data rows in the shard get a chance to be consumed during a ~24 hour training session.

This data loading pipeline is preferred over the one implemented in the torchtitan library ([here](https://github.com/pytorch/torchtitan/blob/main/torchtitan/datasets/hf_datasets.py)), which checkpoints a `_sample_idx` variable and attempts to skip to that idx at the beginning of the next training session, since I couldn't verify that this implementation works correctly (I observed that after resuming the checkpoint, the data loader would keep sampling some of the same data rows from the previous sessions, which should have been skipped).
### Training
The SLURM batch script in [`train_8B_n64.sh`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train_8B_n64.sh) can be used to train a Llama-3.1-8B model with a context size of 8192 tokens over 64 Frontier nodes. This script uses the training config file in [`train_configs/llama3_8b_n64.toml`](https://github.com/eminorhan/frontier-torchtitan/blob/master/train_configs/llama3_8b_n64.toml).
### Checkpoint conversions
Two utility scripts to convert checkpoints between `DCP` and `torch.save` formats are provided here. [`llama_to_dcp.py`](https://github.com/eminorhan/frontier-torchtitan/blob/master/llama_to_dcp.py) converts a checkpoint saved with `torch.save` to `DCP` format. This is useful when initially converting the original Llama-3 checkpoints into `DCP` format to continue pretraining them with the code in this repository (you will most likely need to use this only once before starting continued pretaining). You can do this as follows:
```bash
python llama_to_dcp.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```
where `INPUT_DIR` is the directory where the original checkpoint is saved (downloaded from [here](https://huggingface.co/meta-llama/Llama-3.1-8B/tree/main/original) for the 8B model) and `OUTPUT_DIR` is the directory where the `DCP` checkpoint will be saved. The bulk of this script was copied from [this PR](https://github.com/pytorch/torchtitan/commit/3247841423429faf37bdf6918204350db293e482) by [`rlsl (Rasmus)`](https://github.com/rlrs). 

For the conversion in the other direction (`DCP --> torch.save`), you can use the [`dcp_to_llama.py`](https://github.com/eminorhan/frontier-torchtitan/blob/master/dcp_to_llama.py) script like so:
```bash
python dcp_to_llama.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```
where `INPUT_DIR` now holds the `DCP` checkpoint and the `.pth` checkpoint will be saved in `OUTPUT_DIR`. You will need to do this conversion to evaluate the intermediate checkpoints. Optionally, you can also push the intermediate checkpoints (converted into `.pth` format) to huggingface by passing the argument `--push_to_hub`.
### Evaluation
After converting the checkpoints to `.pth` format, you can evaluate them on some downstram tasks using the [eval_ckpt.sh](https://github.com/eminorhan/frontier-torchtitan/blob/master/eval_ckpt.sh) script. This requires installing `torchtune` (*e.g.*, `pip install torchtune`) and `lm-evaluation-harness` (as described [here](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install)). Running an evaluation is then basically as easy as:
```bash
tune run eleuther_eval --config CONFIG_FILE
```
where `CONFIG_FILE` is the configuration file for the particular evaluation you want to run (see [eval_ckpt.sh](https://github.com/eminorhan/frontier-torchtitan/blob/master/eval_ckpt.sh) for concrete examples). The [`eval_configs`](https://github.com/eminorhan/frontier-torchtitan/tree/master/eval_configs) directory contains configuration files for some common evaluation tasks.






