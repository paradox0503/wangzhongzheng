for folder in human deep1B astro F5 F10 sald; do
    mkdir -p conf/$folder
done

for folder in GPT4SSS TimeLLM AutoTimes UniTime S2IPLLM TimeMixer MyLLM4SSS1 MyLLM4SSS2 MyLLM4SSS3 MyLLM4SSS4 MyLLM4SSS5 MyLLM4SSS6 MyLLM4SSS7 MyLLM4SSS8 MyLLM4SSS9 MyLLM4SSS10 MyLLM4SSS11; do
    for sub in human deep1B astro F5 F10 sald; do
        for subsub in log model sample save; do
            mkdir -p example/$folder/$sub/$subsub
        done
    done
done

for folder in GPT4SSS TimeLLM AutoTimes UniTime S2IPLLM TimeMixer MyLLM4SSS1 MyLLM4SSS2 MyLLM4SSS3 MyLLM4SSS4 MyLLM4SSS5 MyLLM4SSS6 MyLLM4SSS7 MyLLM4SSS8 MyLLM4SSS9 MyLLM4SSS10 MyLLM4SSS11; do
    for sub in human deep1B astro F5 F10 sald; do
        mkdir -p BSF_Data/$folder/$sub
        mkdir -p BSF_Tightness/$folder/$sub
    done
done