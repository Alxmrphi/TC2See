#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=def-afyshe-ab
#SBATCH  -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=jam10@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --array=5-12,14-40   

participants=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "40")
participant_number=${participants[$SLURM_ARRAY_TASK_ID]}
echo $participant_number


PYTHON_SCRIPT=$1
echo $PYTHON_SCRIPT

cd ..
cd ..
source venv/bin/activate
cd analysis_code
cd decoding
python $PYTHON_SCRIPT -subj $participant_number 