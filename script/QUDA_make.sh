cmake .. -GNinja \
  -DQUDA_COVDEV=ON \
  -DQUDA_MPI=ON \
  -DQUDA_GPU_ARCH=sm_86 \
  -DQUDA_MULTIGRID=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc