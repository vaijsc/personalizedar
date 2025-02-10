#!/bin/bash -e
#SBATCH --job-name=job_gpu
#SBATCH --output=/home/binhnq9/khoigroup/ARImagen/binhnq9/scripts/slurm_outputs/slurm_%A.out
#SBATCH --error=/home/binhnq9/khoigroup/ARImagen/binhnq9/scripts/slurm_outputs/slurm_%A.err
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=80GB
#SBATCH --exclude=sdc2-hpc-dgx-a100-001
#SBATCH --mail-user=v.binhnq9@vinai.io
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


srun --container-image=/lustre/scratch/client/scratch/research/group/khoigroup/ARImagen/binhnq9/docker_images/ar_imagen_1.sqsh  --container-mounts=/lustre/scratch/client/scratch/research/group/khoigroup/ARImagen/binhnq9:/root \
    --container-workdir=/root \
    /bin/bash -c \
    "
    export HTTP_PROXY=http://proxytc.vingroup.net:9090/
    export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
    export http_proxy=http://proxytc.vingroup.net:9090/
    export https_proxy=http://proxytc.vingroup.net:9090/

    export TFHUB_CACHE_DIR=/root/cache/tfhub_modules
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export PATH="~/.local/bin:$PATH"

    source /opt/conda/bin/activate
    conda activate /root/conda/switti

    
    cd /root/code/personalizedar/switti
    
    bash train.sh

    "