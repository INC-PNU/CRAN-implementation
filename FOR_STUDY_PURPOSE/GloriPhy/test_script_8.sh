#!/bin/bash

gpu_id=${2:-0}
root_path_gloriphy="$(pwd)/GLoRiPHY_source"
root_path_baseline="$(pwd)/Baseline"
sim_data_dir="$1/simulation"
real_data_dir="$1/real_world"
nodes=(F H)

# Hardcode variables for SF8
sf=8
num_packets=100

echo "========================================"
echo " Starting SF8 testing for GLoRiPHY... "
echo "========================================"

# GLoRiPHY - Simulation (SF8)
python $root_path_gloriphy/main.py \
  --root_path "$root_path_gloriphy" \
  --train_denoiseGen \
  --data_dir "$sim_data_dir/test_SF_$sf" \
  --num_packets $num_packets \
  --checkpoint_dir testing/test_SF${sf}_sim \
  --load checkpoints/denoiseGen/simulation/SF${sf} \
  --load_symbol_conformer checkpoints/AWGN_filter/SF${sf}/best_conformer_jgj.pkl \
  --sf $sf \
  --sim_data \
  --test_mode \
  --transformer_encoder_dim 256 \
  --free_gpu_id "$gpu_id"

python $root_path_gloriphy/offset_correction.py \
  --data_dir "$sim_data_dir/test_SF_$sf" \
  --sf $sf \
  --num_packets $num_packets \
  --sim_data

# GLoRiPHY - Real World (Nodes D-H, defaults to SF8 models)
for node in "${nodes[@]}"; do
  python $root_path_gloriphy/main.py \
    --root_path "$root_path_gloriphy" \
    --train_denoiseGen \
    --data_dir "$real_data_dir/test_Node_$node" \
    --checkpoint_dir testing/test_Node_$node \
    --load_symbol_conformer checkpoints/AWGN_filter/SF8/best_conformer_jgj.pkl \
    --load checkpoints/denoiseGen/real_world/SF8 \
    --test_mode \
    --real_data \
    --free_gpu_id "$gpu_id"

  python $root_path_gloriphy/offset_correction.py \
    --data_dir "$real_data_dir/test_Node_$node" \
    --real_data
done

echo "========================================"
echo " Starting SF8 testing for NELoRa... "
echo "========================================"

# NELoRa - Simulation (SF8)
python $root_path_baseline/main_baseline.py \
  --root_path "$root_path_baseline" \
  --data_dir "$sim_data_dir/test_SF_${sf}_corrected" \
  --sf $sf \
  --num_packets $num_packets \
  --sim_data \
  --checkpoint_dir testing/test_SF${sf}_sim \
  --load checkpoints/simulation/SF${sf} \
  --test_mode \
  --free_gpu_id "$gpu_id"

# NELoRa - Real World (Nodes D-H, defaults to SF8 models)
for node in "${nodes[@]}"; do
  python $root_path_baseline/main_baseline.py \
    --root_path "$root_path_baseline" \
    --data_dir "$real_data_dir/test_Node_${node}_corrected" \
    --real_data \
    --num_packets 200 \
    --checkpoint_dir testing/test_Node_$node \
    --load checkpoints/real-world/SF8 \
    --test_mode \
    --free_gpu_id "$gpu_id"
done

echo "========================================"
echo " Plotting Results... "
echo "========================================"

# Plot results
python $root_path_gloriphy/plot_results.py