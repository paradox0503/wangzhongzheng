import sys
import argparse
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os

# import torchvision
# torchvision.disable_beta_transforms_warning()
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

root_dir = "/mnt/data/user_liangzhiyu/wangzhongzheng/LLM4SSSsummary/"
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


torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# configure
max_data_size = 10_000_000
data_size = 1_000_000

max_query_size = 10_000
query_size = 1_000

batch_size1 = 500
batch_size2 = 100


def ensure_dir_exists(file_path):
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_binary_data(file_path, indices, len_series):
    data = np.zeros((len(indices), len_series), dtype=np.float32)
    with open(file_path, 'rb') as f:
        for i, index in enumerate(indices):
            f.seek(4 * len_series * index)
            data[i] = np.frombuffer(f.read(4 * len_series), dtype=np.float32)
    return torch.tensor(data, dtype=torch.float32)


def main(argv):
    print("Run!!!")
    parser = argparse.ArgumentParser(description='Command-line parameters')
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    args = parser.parse_args(argv[1:])
    conf = Configuration(args.confpath)

    model_selected = conf.getEntry("model_selected")
    device = conf.getEntry("device")
    selected_devices = conf.getEntry("GPUs")

    dataset_selected = conf.getEntry("dataset_selected")

    model_path = f"./example/{model_selected}/{dataset_selected}/model/example_model.pth"

    data_path = conf.getEntry("data_path")

    data_pos = data_path + dataset_selected + "/data.bin"
    query_pos = data_path + dataset_selected + "/query.bin"

    origin_data_pos = f"./BSF_Data/{model_selected}/{dataset_selected}/origin_data.bin"
    origin_query_pos = f"./BSF_Data/{model_selected}/{dataset_selected}/origin_query.bin"
    reduce_data_pos = f"./BSF_Data/{model_selected}/{dataset_selected}/reduce_data.bin"
    reduce_query_pos = f"./BSF_Data/{model_selected}/{dataset_selected}/reduce_query.bin"

    len_series = conf.getEntry("len_series")
    len_reduce = conf.getEntry("len_reduce")
    len_reduce = conf.getEntry("len_reduce")

    data_indices = np.random.randint(0, max_data_size, size=data_size, dtype=np.int64)
    query_indices = np.random.randint(0, max_query_size, size=query_size, dtype=np.int64)

    origin_data = load_binary_data(data_pos, data_indices, len_series)
    origin_query = load_binary_data(query_pos, query_indices, len_series)

    origin_data = origin_data.view(-1, batch_size1, len_series).to(device, non_blocking=True)
    origin_query = origin_query.view(-1, batch_size2, len_series).to(device, non_blocking=True)

    model_map = {
        "GPT4SSS": GPT4SSS,
        "TimeLLM": TimeLLM,
        "AutoTimes": AutoTimes,
        "UniTime": UniTime,
        "S2IPLLM": S2IPLLM,
        "TimeMixer": TimeMixer,
        "MyLLM4SSS1": MyLLM4SSS1,
        "MyLLM4SSS2": MyLLM4SSS2,
    }
    model = model_map[model_selected](conf).to(device)

    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=selected_devices)

    mask = torch.ones((1, len_series)).to(device) if model_selected == "UniTime" else None


    # Start Function Processing Data
    def process_batches(data, reduce_pos, origin_pos, batch_size):
        reduce_batches = np.empty((data.shape[0] * batch_size, len_reduce), dtype=np.float32)
        original_batches = np.empty((data.shape[0] * batch_size, len_series), dtype=np.float32)

        stream = torch.cuda.Stream()
        with torch.no_grad(), torch.cuda.stream(stream):
            for i, batch in enumerate(tqdm(data, desc="Batches", unit="batch")):
                original_batches[i * batch_size : (i + 1) * batch_size] = batch.cpu().numpy()
                reduce_batch = model(batch, mask) if mask is not None else model(batch)
                reduce_batches[i * batch_size : (i + 1) * batch_size] = reduce_batch.cpu().numpy()

        original_batches.tofile(origin_pos)
        reduce_batches.tofile(reduce_pos)
    # End Function Processing Data


    print("Start Processing Data...")
    process_batches(origin_data, reduce_data_pos, origin_data_pos, batch_size1)
    process_batches(origin_query, reduce_query_pos, origin_query_pos,batch_size2)
    print("Processing Completed!")


if __name__ == '__main__':
    main(sys.argv)
