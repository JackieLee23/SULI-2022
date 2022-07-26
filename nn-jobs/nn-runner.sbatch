#!/bin/bash
#############################################################################
#SBATCH --job-name=neural-net
#SBATCH --output=neural-net.out
#SBATCH --nodelist=node055,node056
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=500M
#SBATCH --ntasks=12

nsteps=12
script_name="arch-search.py"

#Begin job
echo "--] running job on " `hostname`
echo Begin: `date`
echo "slurm ntasks: $SLURM_NTASKS"

for i in $(seq $nsteps); do
   
   echo "running $i" 
   srun -n 1 -N 1 python $script_name $i $SLURM_CPUS_PER_TASK &> py-output.out &
   
   if (( $i % $SLURM_NTASKS == 0 )); then
      echo "Waiting..."
      wait
   fi

done
wait
echo End: `date`
