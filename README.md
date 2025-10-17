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
  * [FORGE](#forge) -- Junqi
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
