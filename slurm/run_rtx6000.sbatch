#!/bin/bash
#SBATCH -JSlurmPythonExample                    # Job name
#SBATCH --account=gts-agarg35                   # charge account
#SBATCH -N1 --gres=gpu:RTX_6000:1               # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=12G                       # Memory per core
#SBATCH -t8:00:00                               # Duration of the job (8 hours)
#SBATCH -q embers                               # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=albertwilcox@gatech.edu     # E-mail address for notifications
#SBATCH --output=slurm_out/Report-%A.out
cd $HOME/p-agarg35-0/albert/PRIMO        # Change to working directory

conda activate quest

# srun python scripts/main.py --multirun hydra/launcher=joblib hydra.launcher.n_jobs=3 algo=coql algo.eta=3 algo.tau=3e-4 algo.tau_coql=3e-5 wandb.group=january-debug-1-post-ref exp_name=hope_this_works env=walker2d-medium-v2 algo.trainer.eval_freq=10 general.seed=0,1,2 env.max_action=3 algo.agent.use_alternate_loss=false

echo "Running the following command:"
echo $@

srun $@