#! /bin/bash
#SBATCH --job-name=img_seg_am        # Job name
#SBATCH --partition=batch             # Partition (queue) name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=120gb                     # Job memory request
#SBATCH --time=150:00:00               # Time limit hrs:min:sec
#SBATCH --output=img_seg_am.%j.out     # Standard output log
#SBATCH --error=img_seg_am.%j.err      # Standard error log
#SBATCH --cpus-per-task=45             # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuewu_mike@163.com  # Where to send mail

cd $SLURM_SUBMIT_DIR

module load Detectron2/0.3-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0

time python tile_model_seg_test.py --seed 1 --net-struct mask_rcnn_R_50_FPN_3x --gpu-use=0 1>> ./testmodel.${SLURM_JOB_ID}.out 2>> ./testmodel.${SLURM_JOB_ID}.err
