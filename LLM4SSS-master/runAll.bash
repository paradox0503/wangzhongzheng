dataset="human"
for model in "AutoTimes" "GPT4SSS" "S2IPLLM" "TimeLLM" "UniTime"; do
    nohup python -u LLM4SSSsummary_run.py -C conf/$dataset/$model.json > run.out 2>&1 &
done