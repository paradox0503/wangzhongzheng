import numpy as np
import faiss
from tqdm import tqdm
import os
import sys

root_dir = "/mnt/data/user_liangzhiyu/wangzhongzheng/LLM4SSSsummary/"
sys.path.append(root_dir)
from BSF_Tightness.baseline import baseline_sample


model_selected = "TimeMixer" 
dataset_selected = "sald"

tightness_type = "std_tightness"

print("model:", model_selected, "+ dataset:", dataset_selected)

dataset_config = {
    "human": (256, 16),
    "deep1B": (96, 16),
    "astro": (256, 16),
    "sald": (128, 16),
    "F5": (256, 16),
    "F10": (256, 16)
}

if dataset_selected in dataset_config:
    len_series, len_reduce = dataset_config[dataset_selected]
else:
    raise ValueError("Invalid dataset selected.")

data_num = 1_000_000
query_num = 1_000

data_pos = f"BSF_Data/{model_selected}/{dataset_selected}/origin_data.bin"
query_pos = f"BSF_Data/{model_selected}/{dataset_selected}/origin_query.bin"

# nearest
reduce_data_pos = f"BSF_Data/{model_selected}/{dataset_selected}/reduce_data.bin"
approIndex_series = f"BSF_Data/{model_selected}/{dataset_selected}/origin_query.bin"
exactIndex_path = f"BSF_Tightness/{model_selected}/{dataset_selected}/exactIndex.txt"

# split
source_path = f"BSF_Tightness/{model_selected}/{dataset_selected}/approSeries.bin"
target_paths = [
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approSeries_1.bin",
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approSeries_5.bin",
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approSeries_10.bin",
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approSeries_50.bin",
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approSeries_100.bin"
    ]
whole_size = 5_000
slice_size = 1_000

# getIndex
approSeries_paths = target_paths
approIndex_paths = [
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approIndex_1.txt",
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approIndex_5.txt",
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approIndex_10.txt",
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approIndex_50.txt",
    f"BSF_Tightness/{model_selected}/{dataset_selected}/approIndex_100.txt"
    ]

node_nums = [1, 5, 10, 50, 100]


def nearest():
    data = np.fromfile(data_pos, dtype=np.float32).reshape(-1, len_series)
    queries = np.fromfile(approIndex_series, dtype=np.float32).reshape(-1, len_series)

    index = faiss.IndexFlatL2(len_series)
    index.add(data)

    batch_size = 100
    num_batches = (len(queries) + batch_size - 1) // batch_size

    nearest_indices = []
    for i in tqdm(range(0, len(queries), batch_size), desc="Queries", unit="batch"):
        batch_queries = queries[i:i+batch_size]
        _, batch_nearest_indices = index.search(batch_queries, 1)  # k=1
        nearest_indices.extend(batch_nearest_indices.flatten())

    np.savetxt(exactIndex_path, np.array(nearest_indices), fmt="%d")

    print(f"Processing complete. Results saved to {exactIndex_path}")
    
    
def split():
    sequences = []
    for idx in range(0, whole_size):
        sequence = np.fromfile(source_path, dtype=np.float32, count=len_reduce, offset=4 * len_reduce * idx)
        sequences.append(sequence)
    for i in range(0, 5):
        temp = np.concatenate(sequences[slice_size * i:slice_size * (i + 1)])
        temp.tofile(target_paths[i])
        

def getIndex():
    for i in range(0, 5):
        approSeries_path = approSeries_paths[i]
        data_seq = np.memmap(reduce_data_pos, dtype=np.float32, mode='r', shape=(data_num, len_reduce))
        query_seq = np.fromfile(approSeries_path, dtype=np.float32, count=query_num * len_reduce).reshape(query_num, len_reduce)
        approIndex_path = approIndex_paths[i]

        match_indice = []
        for query_idx, query in enumerate(query_seq):
            match = np.all(data_seq==query, axis=1)
            match_index = np.where(match)[0][0]
            if match_index.size > 0:
                # print(f"Series {query_idx} found at index {match_index}.")
                match_indice.append(match_index)
            else:
                raise ValueError(f"Series {query_idx} failed to find the index.")

        with open(approIndex_path, 'w') as f:
            f.write("\n".join(map(str, match_indice)) + "\n")
                

def tightness():
    exact_indice = []
    with open(exactIndex_path, 'r') as file:
            for line in file:
                index = int(line.strip())
                exact_indice.append(index)

    origin_data = np.fromfile(data_pos, dtype=np.float32).reshape(-1, len_series)
    origin_query = np.fromfile(query_pos, dtype=np.float32).reshape(-1, len_series)

    # baseline sample
    if tightness_type == "std_tightness":
        sample_num = 1000
        sample_indice = baseline_sample(model_selected, dataset_selected, data_num, len_series, 8, 16, 16, sample_num)
        sample = origin_data[sample_indice]
    
    for appro_index_path, node_num in zip(approIndex_paths, node_nums):
        appro_indice = []
        with open(appro_index_path, 'r') as file:
            for line in file:
                index = int(line.strip())
                appro_indice.append(index)
        
        # different tightness type
        
        all_tightness = []
        for i in range(0, query_num):
            query = origin_query[i]
            exact = origin_data[exact_indice[i]]
            appro = origin_data[appro_indice[i]]
            dis1 = np.linalg.norm(query - exact)
            dis2 = np.linalg.norm(query - appro)
            
            if tightness_type == "std_tightness":
                dis_sample = np.array([np.linalg.norm(query - sample[i]) for i in range(sample_num)])
                dis3 = dis_sample.mean(axis=0)
                tightness = (dis3 - dis2) / (dis3 - dis1)
            elif tightness_type == "tightness":
                tightness = dis1 / dis2
            
            all_tightness.append(tightness)
            
        tightness_mean = np.mean(all_tightness)
        print(f"{tightness_mean}")
        
        
def main():
    if not os.path.exists(exactIndex_path):
        print("Searching nearest...")
        nearest()
        
    if not os.path.exists(approSeries_paths[0]):
        print("Spliting...")
        split()
        
    if not os.path.exists(approIndex_paths[0]):
        print("Getting index...")
        getIndex()
        
    tightness()
    
    
if __name__ == "__main__":
    main()