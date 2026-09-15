import numpy as np
from collections import Counter


dataset = "F5"
model = "AutoTimes"
data_path = "/mnt/data/user_liangzhiyu/wangzhongzheng/LLM4SSSsummary/BSF_Data/"
data_pos = f"{data_path}{model}/{dataset}/origin_data.bin"

data_num = 1000
len_series = 256
total_length = data_num * len_series

k = 6

data = np.fromfile(data_pos, dtype=np.float32, count=total_length)
# data: [data_num * len_series]
data_series = data.reshape(data_num, len_series)
# data_series: [data_num, len_series]


def find_top_k_periods(series, k=3, max_period=256/2, min_period=256/16):
    n = len(series)
    fft_coeffs = np.fft.fft(series)
    amplitudes = np.abs(fft_coeffs)

    freqs = np.fft.fftfreq(n) * n  # 转换为实际频率单位
    positive_mask = freqs > 0      # 筛选正频率
    freqs = freqs[positive_mask]
    amplitudes = amplitudes[positive_mask]

    # 计算周期
    periods = np.ceil(n / freqs).astype(int)

    # 过滤周期大于 max_period 的项
    valid_mask = (periods <= max_period) & (periods >= min_period)
    freqs = freqs[valid_mask]
    amplitudes = amplitudes[valid_mask]
    periods = periods[valid_mask]

    # 重新排序并取前k个
    sorted_indices = np.argsort(amplitudes)[::-1]
    top_k_indices = sorted_indices[:k]

    return periods[top_k_indices], freqs[top_k_indices], amplitudes[top_k_indices]

period_counter = Counter()

for seires in data_series:
    periods, freqs, amplitudes = find_top_k_periods(seires, k)
    print(periods, freqs, amplitudes)
    period_counter.update(periods)
    
print("Period Frequency Count:")
for period, count in period_counter.items():
    print(f"Period: {period}, Count: {count}")
    
    
top_k_periods = period_counter.most_common(k)

# 输出出现次数最多的 k 个周期及其计数
print(f"Top {k} Periods by Frequency:")
for period, count in top_k_periods:
    print(f"Period: {period}, Count: {count}")