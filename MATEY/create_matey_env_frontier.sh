#!/bin/bash
rocm_version="6.1.3"
module load PrgEnv-gnu
module load rocm/$rocm_version

PROJ_NAME="stf218"
## CHANGE THIS
export WRKSPC="$MEMBERWORK/${PROJ_NAME}/matey-reproducer"
mkdir -p $WRKSPC
cd $WRKSPC

# Setup Virtual Environment
echo "Setting up conda Environment"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $WRKSPC/miniconda
export PATH=$WRKSPC/miniconda/bin:$PATH
conda create --prefix $WRKSPC/miniconda/envs/matey-env -y
source $WRKSPC/miniconda/etc/profile.d/conda.sh
conda activate $WRKSPC/miniconda/envs/matey-env

# PyTorch
echo "Installing PyTorch"
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/rocm6.1
pip install timm==0.9.2


# other dependencies
pip install torchinfo adan-pytorch dadaptation wandb h5py matplotlib scipy timm
git clone https://github.com/sandialabs/exodusii
pushd exodusii
pip install .
popd

git clone https://github.com/mpi4py/mpi4py.git
pushd mpi4py
CC=$(which mpicc) CXX=$(which mpicxx) python setup.py build --mpicc=$(which mpicc)
CC=$(which mpicc) CXX=$(which mpicxx) python setup.py install
popd

#install flash attn, take ~30mins 
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
git checkout v2.6.3
pip install -e .
popd

# RCCL plugin
echo "Installing RCCL Plugin"
git clone --recursive --depth=1 https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl 
pushd aws-ofi-rccl
libfabric_path=/opt/cray/libfabric/1.22.0
./autogen.sh
export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/lib:$LD_LIBRARY_PATH
CC=cc CFLAGS=-I/opt/rocm-$rocm_version/include ./configure \
    --with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
    --prefix=$PWD --with-hip=/opt/rocm-$rocm_version 
make
make install
popd 
export LD_LIBRARY_PATH=$PWD/aws-ofi-rccl/lib:$LD_LIBRARY_PATH


#save to env file to source 
cat <<EOF > matey-env.sh
module load PrgEnv-gnu
module load rocm/$rocm_version
WRKSPC="$MEMBERWORK/${PROJ_NAME}/matey-reproducer"
export PATH=\$WRKSPC/miniconda/bin:\$PATH
source \$WRKSPC/miniconda/etc/profile.d/conda.sh
conda activate \$WRKSPC/miniconda/envs/matey-env
export LD_LIBRARY_PATH=$PWD/aws-ofi-rccl/lib:\$LD_LIBRARY_PATH
EOF



