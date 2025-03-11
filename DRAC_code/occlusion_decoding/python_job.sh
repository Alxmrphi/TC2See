#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --account=def-afyshe-ab            
#SBATCH --cpus-per-task=1        
#SBATCH --mem=32G                  
#SBATCH --mail-user=jam10@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

PYTHON_SCRIPT=$1
echo $PYTHON_SCRIPT

cd ..
source venv/bin/activate
cd occlusion_decoding

module load cuda cudnn  

python $PYTHON_SCRIPT