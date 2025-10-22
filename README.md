# AmSC At-Scale Services - OLCF
AI and simulation software stack, baseline environments, and approaches for running on Frontier. 

# Contents

- AI Stack 
  * [LLM](#llm) -- Junqi
    + [DeepSpeed](#deepspeed)
    + [Megatron](#megatron)
    + [GPT-Neox](#gpt-neox)
    + [vLLM](#vllm)
  * [ViT](#vit) -- Aris
  * [GNN](#gnn) -- Max
- Simulation Stack
  * [VASP](#vasp) -- Max 
  * [LSMS](#lsms) -- Junqi
  * [CFD](#cfd) -- Ramki 
- Baseline Environment
  * [HydraGNN](#hydragnn) -- Max
  * [LORACX](#loracx) -- Ramki 
  * [LLM Pre-Training -- FORGE](#forge) -- Junqi
  * [ORBIT](#orbit) -- Aris
  * [Matey](#matey) -- Junqi

## LLM  
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
### Step 1: Initial Environment Setup
First, load the required programming environment and ROCm modules.
```bash
module  load  PrgEnv-gnu/8.6.0
module  load  miniforge3/23.11.0-0
module  load  rocm/6.4.1
module  load  craype-accel-amd-gfx90a
export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
```
### Step 2: Create and Activate Conda Environment
Make sure to replace the the directory in following code. It's recommended to install it in a shared lustre directory (`/lustre/orion/`) to ensure sufficient storage space.
```bash
# Create the env
conda create -p /lustre/orion/{..env_dir..}/py312 python=3.12
# Activate the env
source activate /lustre/orion/{..your_dir..}/py312
```
### Step 3: Install Core Dependencies
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
### Step 4: Build and Configure AWS OFI RCCL Plugin
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
### Step 5: Fused Kernels:
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


