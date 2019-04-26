export PYTHONPATH=$(pwd):${PYTHONPATH}
module load cuda/9.0
module load NCCL/2.2.12-1-cuda.9.0 # Or other NCCL versions corresponding to your loaded CUDA version
module load cudnn/v7.0-cuda.9.0
module load anaconda3/5.0.1
export C_INCLUDE_PATH=/public/apps/cuda/9.0/include/:$C_INCLUDE_PATH
export PATH=/public/apps/cuda/9.0/include/:$PATH
