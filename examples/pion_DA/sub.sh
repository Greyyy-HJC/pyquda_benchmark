#!/bin/bash

#SBATCH --job-name=pyq_da
#SBATCH --account=pion3d
#SBATCH --partition=lq2_gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4     # 4 tasks per node
#SBATCH --cpus-per-task=16      # 16 CPU cores per task
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --distribution=cyclic
#SBATCH --output=/lustre1/pion3d/jinchen/run/pyquda_pion_bench/log/pi_DA.%j.out
#SBATCH --error=/lustre1/pion3d/jinchen/run/pyquda_pion_bench/log/pi_DA.%j.err


# Set working directory
rundir=$SLURM_SUBMIT_DIR
cd $rundir

# Enable GPU support for MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# Output node information
echo -e "\n>>> SLURM_JOB_NODELIST content:"
scontrol show hostname $SLURM_JOB_NODELIST
NODES=$SLURM_JOB_NUM_NODES
TASKS=$SLURM_NTASKS
echo -e "${NODES}n*${TASKS}t\n"

# Display current time
echo -e "\n>>> Start time: $(date)"
start_time=$(date +%s)

# Load environment
source /lustre1/pion3d/jinchen/env/gpt.env
python3 --version
export PYTHONPATH=$PYTHONPATH:/lustre1/pion3d/jinchen/run/qTMD_softFF/gpt_utils:/lustre1/pion3d/jinchen/run/qTMD_softFF/gpt_utils/utils:/lustre1/pion3d/jinchen/run/qTMD_softFF/gpt_utils/qTMD
echo $PYTHONPATH

# Check CUDA configuration
echo -e "\n>>> Check nvcc:"
which nvcc
nvcc --version

# Display GPU information
echo -e "\n>>> Show GPU info:"
nvidia-smi

# Output LD_LIBRARY_PATH
echo -e "\n>>> Output LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH

# QUDA global environment
rundir="/lustre1/pion3d/jinchen/run/pyquda_pion_bench"
cd ${rundir}


export OMP_NUM_THREADS=32
export QUDA_ENABLE_TUNING=1
export QUDA_ENABLE_P2P=0
export QUDA_ENABLE_MPS=1
export QUDA_ENABLE_DEVICE_MEMORY_POOL=0

# Run Python script
# main=proton_disp.py
main=pion_DA.py
echo -e "\n>>> Run Python script ${main}"

N_conf=10

# srun -N 1 -n 4 --mpi=pmix --gpus-per-task=1 -u \
#     python3 ${main} --N_conf ${N_conf} --mpi_geometry 1.1.1.4 \
#     --mpi 1.1.1.4 --mpi_split 1.1.1.4 --grid 48.48.48.64 \
#     --shm-mpi 1 --shm 2048 \
#     --comms-sequential \
#     --accelerator-threads 16 \
#     --device-mem 26000 --comms-overlap --comms-concurent

srun -N 2 -n 8 --mpi=pmix --gpus-per-task=1 -u \
    python3 ${main} --N_conf ${N_conf} --mpi_geometry 1.1.1.8 \
    --mpi 1.1.1.8 --mpi_split 1.1.1.8 --grid 48.48.48.64 \
    --shm-mpi 1 --shm 2048 \
    --comms-sequential \
    --accelerator-threads 16 \
    --device-mem 26000 --comms-overlap --comms-concurent


# Display end time
echo -e "\n>>> End time: $(date)"
end_time=$(date +%s)

# Calculate and display total runtime
total_time=$(($end_time - $start_time))
hours=$(($total_time / 3600))
minutes=$(($total_time % 3600 / 60))
seconds=$(($total_time % 60))

echo -e "\n>>> Total runtime: ${hours}:${minutes}:${seconds}"
