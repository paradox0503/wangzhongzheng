import os
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader     # pyright:ignore
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler     # pyright:ignore
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count

from utils.conf import Configuration
from utils.sample import getSamples
from utils.sample import TSData

from utils.loss import ScaledL2Loss
from utils.loss import ExpScaledL2Loss
from utils.loss import LogScaledL2Loss
from utils.loss import Umap

from model.GPT4SSS import GPT4SSS
from model.TimeLLM import TimeLLM
from model.AutoTimes import AutoTimes
from model.UniTime import UniTime
from model.S2IPLLM import S2IPLLM
from model.TimeMixer import TimeMixer
from model.UniTS import UniTS
from model.TimeMoE import TimeMoE
   
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


class Experiment:
    def __init__(self, conf:Configuration):
        # Configuration
        self.conf = conf
        self.device = self.conf.getEntry("device")
        self.epoch_max = self.conf.getEntry("epoch_max")
        self.model_path = self.conf.getEntry("model_path")
        self.batch_size = self.conf.getEntry("batch_size")
        self.log_path = self.conf.getEntry("log_path")
        self.model_selected = self.conf.getEntry("model_selected")
        if self.model_selected == "UniTime":
            self.mask_rate = self.conf.getEntry("mask_rate")
        self.loss_method = "ScaledL2Loss"

        # Early Stopping
        self.patience1 = 10
        self.patience2 = 10
        self.best_val_error = float('inf')
        self.epochs_without_improvement = 0
        self.delta = 0.0005
        self.val_error_history = []

        if self.model_selected in ('UniTS', 'TimeMoE'):
            os.makedirs(os.path.dirname(self.log_path) or '.', exist_ok=True)

        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s',
            filename = self.log_path,
            filemode = "w"
        )


    def run(self) -> None:
        self.setup()

        self.epoch = 0
        # self.validate()

        while self.epoch < self.epoch_max:
            self.epoch += 1

            self.train()
                
            if self.epoch == 1:
                break

        # self.test()


    def setup(self) -> None:
        # print("Setup experiment...")
        self.len_series = self.conf.getEntry("len_series")
        self.len_reduce = self.conf.getEntry("len_reduce")
        
        model_selected = self.conf.getEntry("model_selected")
        
        model_classes = {
            "GPT4SSS": GPT4SSS,
            "TimeLLM": TimeLLM,
            "AutoTimes": AutoTimes,
            "UniTime": UniTime,
            "S2IPLLM": S2IPLLM,
            "TimeMixer": TimeMixer,
            "UniTS": UniTS,
            "TimeMoE": TimeMoE,
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
            "MyLLM4SSS11": MyLLM4SSS11
        }
        if model_selected in model_classes:
            self.model = model_classes[model_selected](self.conf).to(self.device)
        else:
            raise ValueError("Unsupported LLM model.")

        model_structure = str(self.model)
        logging.info(f"Model Structure:\n{model_structure}")

        logging.info("Experiment Configuration:")
        for key, value in self.conf.confLoaded.items():
            logging.info(f"{key}: {value}")

        if os.path.exists(f"{self.model_path}example_model.pth"):
            logging.info("Model loading...")
            checkpoint = torch.load(f"{self.model_path}example_model.pth", weights_only=True)
            self.model.load_state_dict(checkpoint)
        else:
            logging.info("Model initializing...")
            self.model = self.InitModel(self.model)

        logging.info(f"Loss method: {self.loss_method}")
        if self.loss_method == "ScaledL2Loss":
            self.loss_calculator = ScaledL2Loss(self.len_series, self.len_reduce)
        elif self.loss_method == "ExpScaledL2Loss":
            self.loss_calculator = ExpScaledL2Loss(self.len_series, self.len_reduce)
        elif self.loss_method == "LogScaledL2Loss":
            self.loss_calculator = LogScaledL2Loss(self.len_series, self.len_reduce)
        elif self.loss_method == "Umap":
            self.loss_calculator = Umap(self.len_series, self.len_reduce)
        else:
            raise ValueError("Invalid loss method selected.")

        self.error_calculator = ScaledL2Loss(self.len_series, self.len_reduce)

        self.optimizer = AdamW(self.model.parameters(), lr = 1e-4, weight_decay = 0.01)
        self.scaler = GradScaler()

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda = self.lr_lambda)

        # multi-GPUs
        if torch.cuda.device_count() > 1:
            selected_device = self.conf.getEntry("GPUs")
            logging.info(f"Using {len(selected_device)} GPUs")
            self.model = nn.DataParallel(self.model, device_ids=selected_device)


    def InitModel(self, model):
        if model is None:
            raise ValueError("The `model` passed to `InitModel` is None.")

        if not isinstance(model, torch.nn.Module):
            raise TypeError("The `model` passed to `InitModel` is not an instance of `torch.nn.Module`.")

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Apply initialization to all submodules except certain ones
        for name, module in model.named_modules():
            if 'llm' in name:
                continue
            init_weights(module)
        return model


    # For UniTime
    def random_mask(self):
        mask = torch.rand((self.batch_size, self.len_series))
        # mask: (batch_size, len_series)
        mask[mask < self.mask_rate] = 0  # masked
        mask[mask >= self.mask_rate] = 1  # remained
        # mask: (batch_size, len_series)
        return mask


    def train(self) -> None:
        logging.info(f'epoch: {self.epoch}, start training')

        if self.loss_method == "Umap":
            batch = batch.to(self.device)

            batch_reduce = self.model(batch)   # (batch_size, len_reduce)
            loss = self.loss_calculator(batch, batch_reduce)

            self.scaler.scale(loss).backward()      # pyright:ignore
            self.scaler.step(self.optimizer)
            self.scaler.update()

        elif self.loss_method in ["ScaledL2Loss", "ExpScaledL2Loss", "LogScaledL2Loss"]:
            self.optimizer.zero_grad()
            
            inputs = torch.randn(self.batch_size, self.len_series).to(self.device)  # dummy data for AMP
            
            if self.model_selected == "UniTime":
                mask = self.random_mask().to(self.device)
                inputs = inputs.masked_fill(mask==0, 0)   # (batch_size, len_series)
                inputs_union = (inputs, mask)
                flops = FlopCountAnalysis(self.model, (inputs_union,))
                params = parameter_count(self.model)
                print(f"Total FLOPs: {flops.total() / 1e9:.2f} G")
                print(f"Total Params: {sum(params.values()) / 1e6:.2f} M")
            else:
                flops = FlopCountAnalysis(self.model, inputs)
                params = parameter_count(self.model)
                print(f"Total FLOPs: {flops.total() / 1e9:.2f} G")
                print(f"Total Params: {sum(params.values()) / 1e6:.2f} M")
