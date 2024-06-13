salloc -A gts-agarg35 -N1 --mem-per-gpu=12G -q embers -t8:00:00 --gres=gpu:RTX_6000:1

cd $HOME/p-agarg35-0/albert/quest        # Change to working directory

conda activate quest



salloc -A gts-agarg35 -N1 --mem-per-gpu=12G -q embers -t8:00:00 --gres=gpu:V100:1

cd $HOME/p-agarg35-0/albert/quest        # Change to working directory

conda activate quest




salloc -A gts-agarg35 -N1 --mem-per-gpu=12G -q embers -t8:00:00 --gres=gpu:A100:1

cd $HOME/p-agarg35-0/albert/quest        # Change to working directory

conda activate quest
