#!/bin/bash
############################## ./grid_search.sh #############################################
# IMPORTANT: rename output_dir to output_dir=f"{model_name}-lora_{RANK}_{ALPHA}_{DROPOUT}",
# otherwise dropouts will override
# Define the array of rank values
rank_list=(64 128 256 512 1024)
dropout_list=(0.05 0.1 0.15 0.2 0.25)

# Loop through each rank_k value
for rank in "${rank_list[@]}"
do
  alpha=$((2*rank))  # Set alpha as twice the value of rank
  for dropout in "${dropout_list[@]}"
  do
    # Execute the python script with the current rank_k value and sublayer_name
    nohup python3 medsam_train_native.py --rank "$rank" --dropout "$dropout" --alpha "$alpha" &
  done
done
############################################################################################
