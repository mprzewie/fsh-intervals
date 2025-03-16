#!/bin/bash
#SBATCH --job-name=hyper-few-shot
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=student
#SBATCH --cpus-per-task=8

export NEPTUNE_PROJECT='adam.piotr.w/IntervalMAML'
export NEPTUNE_API_TOKEN='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZGIyOWYyOS03YWNlLTQ1MWYtYTQ5Yi0wYjBhZWU5MDhhZmEifQ=='

source activate intervalhypernet

#export ARGSPATH="args.json"
export MODELPATH="save/checkpoints/cross_char/Conv4_interval_hmaml_5way_1shot/best_model.tar"

#echo "ARGSPATH=$ARGSPATH"
echo "MODELPATH=$MODELPATH"

#echo "MODEL ARGUMENTS:"
#echo $(python parse_args.py)


python hypershot_uncertainty.py --hm_weight_set_num_test 0 --es_threshold 10 --dataset cross_char --num_classes 4112 --train_n_way 5 \
         --seed 1 --method interval_hmaml --stop_epoch 64 --model Conv4 --hm_enhance_embeddings True --hm_use_class_batch_input \
   --lr_scheduler multisteplr --n_shot 1 --hm_maml_warmup --lr 0.01 --hm_maml_warmup_epochs 50 --hm_maml_warmup_switch_epochs 500 \
  --hn_head_len 3 --hn_hidden_size 512 --milestones 51 550 --hm_weight_set_num_train 5 \
  --hm_eps 0 --hm_eps_pump_epochs 10 --hm_eps_pump_value 8e-4 --hm_radius_eps_warmup_epochs 500 --hm_worst_case_loss_multiplier 12 \
  --hn_eps 0.00001 --hn_eps_pump_epochs 2000 --hn_eps_pump_value 0.0001  #$(python parse_args.py)
