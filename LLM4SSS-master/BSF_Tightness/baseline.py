from scipy.stats import norm
import numpy as np
from collections import defaultdict


np.set_printoptions(threshold=np.inf)


def baseline_sample(model, dataset, data_num, len_series, win_size, sax_seg, bucket_size, sample_num)->list:
    # Get data
    data_pos = f"BSF_Data/{model}/{dataset}/origin_data.bin"
    total_length = data_num * len_series
    data = np.fromfile(data_pos, dtype=np.float32, count=total_length)   # data: [total_length]
    data_series = data.reshape(data_num, len_series)   # data: [data_num, len_series]
    
    # Get avg std
    mean = np.mean(data)
    std = np.std(data)
    
    # Get Sax_Segment
    if sax_seg > 0:
        quantiles = [i / sax_seg for i in range(1, sax_seg)]
        boundaries = [norm.ppf(q) for q in quantiles]
        boundaries = [mean + std * b for b in boundaries]
    else:
        raise ValueError("sax_seg must be > 0")
    
    # window summarize
    data_series = data_series.reshape(data_num, len_series // win_size, win_size)   # data: [data_num, len_series // win_size, win_size]
    data_avg = data_series.mean(axis=-1)   # data_avg: [data_num, len_series // win_size]
    
    # Get belonging
    flat_data = data_avg.ravel()   # flat_data: [total_length]
    indices = np.searchsorted(boundaries, flat_data, side='right')   # shape: [total_length]
    segment_ids = indices.reshape(data_num, len_series // win_size)   # segment_ids: [data_num, len_series // win_size]
    
    # Get bucket
    win_num = len_series // win_size
    sum_num = win_num * (sax_seg - 1) + 1
    sum_series = segment_ids.sum(axis=-1)   # sum_series: [data_num]
    
    buckets = [[] for _ in range(sum_num)]
    for idx, sum_val in enumerate(sum_series):
        buckets[sum_val].append(idx)
    merged_buckets = [sum(buckets[i:i+bucket_size], []) for i in range(0, len(buckets), bucket_size)]
    
    # get sample
    sampled_indices = []
    for bucket in merged_buckets:
        bucket_len = len(bucket)
        if bucket_len == 0:
            continue
        sample_size = int((bucket_len / data_num) * sample_num)
        sample_size = min(sample_size, bucket_len)
        sampled = np.random.choice(bucket, size=sample_size, replace=False)
        sampled_indices.extend(sampled)

    while len(sampled_indices) < sample_num:
        for bucket in merged_buckets:
            if len(bucket) == 0:
                continue
            remaining = list(set(bucket) - set(sampled_indices))
            if remaining:
                sampled_indices.append(np.random.choice(remaining))
            if len(sampled_indices) >= sample_num:
                break
            
    return sampled_indices


baseline_sample("AutoTimes", "human", 1_000_000, 256, 8, 16, 16, 10000)
