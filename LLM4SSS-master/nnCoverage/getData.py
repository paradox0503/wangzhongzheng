import sys
import argparse
import torch
import numpy as np
from torch import nn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
import pywt
import os
from tqdm import tqdm
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir + "/../"
sys.path.append(root_dir)

from utils.conf import Configuration

from model.GPT4SSS import GPT4SSS
from model.TimeLLM import TimeLLM
from model.AutoTimes import AutoTimes
from model.UniTime import UniTime
from model.S2IPLLM import S2IPLLM
from model.TimeMixer import TimeMixer

from model.MyLLM4SSS1 import MyLLM4SSS1
from model.MyLLM4SSS2 import MyLLM4SSS2
from model.MyLLM4SSS3 import MyLLM4SSS3
from model.MyLLM4SSS4 import MyLLM4SSS4
from model.MyLLM4SSS5 import MyLLM4SSS5
from model.MyLLM4SSS6 import MyLLM4SSS6
from model.MyLLM4SSS7 import MyLLM4SSS7
from model.MyLLM4SSS8 import MyLLM4SSS8
from model.MyLLM4SSS9 import MyLLM4SSS9
from model.MyLLM4SSS10 import MyLLM4SSS10
from model.MyLLM4SSS11 import MyLLM4SSS11


origin_data_path = "./nnCoverage/data/origin_data.bin"
origin_query_path = "./nnCoverage/data/origin_query.bin"
reduce_data_path = "./nnCoverage/data/reduce_data.bin"
reduce_query_path = "./nnCoverage/data/reduce_query.bin"

train_data_path = "./example"

max_data_size = 10_000_000
data_size = 20000
max_query_size = 10000
query_size = 1000
train_size = 200000

batch_size = 100

dim_reduction = "LLM"  # "LLM", "PCA", "SVD", "PAA", "DWT", "RP"
print(f"Dimensionality reduction method: {dim_reduction}")

def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters')
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    args = parser.parse_args(argv[1:])
    conf = Configuration(args.confpath)

    # model and reduction method
    model_selected = conf.getEntry("model_selected")

    device = conf.getEntry("device")
    selected_devices = conf.getEntry("GPUs")
    dataset_selected = conf.getEntry("dataset_selected")
    model_path = f"./example/{model_selected}/{dataset_selected}/model/example_model.pth"

    len_series = conf.getEntry("len_series")
    len_reduce = conf.getEntry("len_reduce")

    data_path = conf.getEntry("data_path")
    data_pos = data_path + dataset_selected + "/data.bin"
    query_pos = data_path + dataset_selected + "/query.bin"

    train_pos = f"{train_data_path}/{model_selected}/{dataset_selected}/sample/train.bin"

    if dim_reduction == "LLM":
        model_classes = {
            "GPT4SSS": GPT4SSS,
            "TimeLLM": TimeLLM,
            "AutoTimes": AutoTimes,
            "UniTime": UniTime,
            "S2IPLLM": S2IPLLM,
            "TimeMixer": TimeMixer,
            "MyLLM4SSS1": MyLLM4SSS1,
            "MyLLM4SSS2": MyLLM4SSS2,
            "MyLLM4SSS3": MyLLM4SSS3,
            "MyLLM4SSS4": MyLLM4SSS4,
            "MyLLM4SSS5": MyLLM4SSS5,
            "MyLLM4SSS6": MyLLM4SSS6,
            "MyLLM4SSS7": MyLLM4SSS7,
            "MyLLM4SSS8": MyLLM4SSS8,
            "MyLLM4SSS9": MyLLM4SSS9,
            "MyLLM4SSS10": MyLLM4SSS10,
            "MyLLM4SSS11": MyLLM4SSS11,
        }

        if model_selected in model_classes:
            model = model_classes[model_selected](conf).to(device)
        else:
            raise ValueError("Unsupported LLM model.")

        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=selected_devices)

    def getTestData(data_pos, query_pos, data_size, query_size):
        data_indices = np.random.randint(0, max_data_size, size=data_size, dtype=np.int64)
        query_indices = np.random.randint(0, max_query_size, size=query_size, dtype=np.int64)

        origin_data, origin_query = [], []
        for index in data_indices:
            sequence = np.fromfile(data_pos, dtype=np.float32, count=len_series, offset=4*len_series*index)
            if not np.isnan(np.sum(sequence)):
                origin_data.append(sequence)
        for index in query_indices:
            sequence = np.fromfile(query_pos, dtype=np.float32, count=len_series, offset=4*len_series*index)
            if not np.isnan(np.sum(sequence)):
                origin_query.append(sequence)

        return np.array(origin_data, dtype=np.float32), np.array(origin_query, dtype=np.float32)

    origin_data_np, origin_query_np = getTestData(data_pos, query_pos, data_size, query_size)
    train_data2, _ = getTestData(data_pos, query_pos, train_size, 0)
    origin_data_np.tofile(origin_data_path)
    origin_query_np.tofile(origin_query_path)
    print("Data and query generated successfully.")

    if dim_reduction == "LLM":
        origin_data = torch.from_numpy(origin_data_np).reshape(-1, batch_size, len_series).to(device)
        origin_query = torch.from_numpy(origin_query_np).reshape(-1, batch_size, len_series).to(device)

        reduce_data, reduce_query = [], []
        for batch in tqdm(origin_data, desc="LLM Reducing Data"):
            with torch.no_grad():
                reduce_data.append(model(batch).cpu().numpy())
        for batch in tqdm(origin_query, desc="LLM Reducing Query"):
            with torch.no_grad():
                reduce_query.append(model(batch).cpu().numpy())

        reduce_data = np.vstack(reduce_data)
        reduce_query = np.vstack(reduce_query)

    else:
        origin_data = origin_data_np.reshape(-1, len_series)
        origin_query = origin_query_np.reshape(-1, len_series)

        train_data = torch.tensor(np.fromfile(train_pos, dtype=np.float32))
        train_data = train_data.reshape(-1, len_series)

        if dim_reduction == "PCA":
            reducer = PCA(n_components=len_reduce)
        elif dim_reduction == "SVD":
            reducer = TruncatedSVD(n_components=len_reduce)
        elif dim_reduction == "RP":
            reducer = GaussianRandomProjection(n_components=len_reduce)
        elif dim_reduction == "PAA":
            def paa(series, segments):
                return np.mean(series.reshape(-1, segments), axis=1)
            reduce_data = np.array([paa(x, len_series // len_reduce) for x in tqdm(origin_data, desc="PAA Reducing Data")])
            reduce_query = np.array([paa(x, len_series // len_reduce) for x in tqdm(origin_query, desc="PAA Reducing Query")])
            reducer = None
        elif dim_reduction == "DWT":
            def dwt_reduction(series, target_len):
                coeffs = pywt.wavedec(series, 'db1', level=None)
                flattened = np.concatenate(coeffs)
                return flattened[:target_len] if len(flattened) >= target_len else np.pad(flattened, (0, target_len - len(flattened)))
            reduce_data = np.array([dwt_reduction(x, len_reduce) for x in tqdm(origin_data, desc="DWT Reducing Data")])
            reduce_query = np.array([dwt_reduction(x, len_reduce) for x in tqdm(origin_query, desc="DWT Reducing Query")])
            reducer = None
        else:
            raise ValueError(f"Unsupported reduction method: {dim_reduction}")

        if reducer:
            print("Fitting reduction model...")
            reducer.fit(train_data2)
            reduce_data = reducer.transform(origin_data)
            reduce_query = reducer.transform(origin_query)

    reduce_data, reduce_query = reduce_data.reshape(-1, len_reduce), reduce_query.reshape(-1, len_reduce)
    reduce_data.tofile(reduce_data_path)
    reduce_query.tofile(reduce_query_path)


if __name__ == '__main__':
    main(sys.argv)
