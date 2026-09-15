dataset="human"
model="S2IPLLM"
# nohup python -u calculate_params_flop.py -C conf/$dataset/$model.json > run.out 2>&1 &
python -u calculate_params_flop.py -C conf/$dataset/$model.json