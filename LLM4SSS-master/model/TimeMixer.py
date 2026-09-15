import torch
from torch import nn

from utils.conf import Configuration
from model.Decompose import series_decomp


class TimeMixer(nn.Module):
    def __init__(self, conf: Configuration):
        super(TimeMixer, self).__init__()
        
        self.conf = conf
        self.len_series = conf.getEntry('len_series')
        self.len_reduce = conf.getEntry('len_reduce')
        self.down_sampling_num = conf.getEntry('down_sampling_num')
        self.sampling_win_size = conf.getEntry('sampling_win_size')
        self.d_model = conf.getEntry('d_model')
        self.d_ff = conf.getEntry('d_ff')
        self.enc_layers = conf.getEntry('enc_layers')
        
        self.enc_embeder = DataEmbedding(1, self.d_model)
        self.pdm_blocks = nn.ModuleList([DecomposableMixer(self.conf) for _ in range(self.enc_layers)])
        self.projection_layers = nn.ModuleList([
            nn.Linear(self.len_series // (self.sampling_win_size ** i), self.len_reduce) 
            for i in range(self.down_sampling_num + 1)
        ])
        self.dim_project_layer = nn.Linear(self.d_model, 1)
        
        
    def forward(self, x):
        # x: [batch_size, len_series]
        
        x = x.unsqueeze(-1)
        # x: [batch_size, len_series, 1]
        x_list = self.multi_scale(x)
        # x_list: [down_sampling_num + 1, batch_size, len_series_i, 1]
        
        enc_out_list = []
        for x in x_list:
            # x: [batch_size, len_series_i, 1]
            x = self.enc_embeder(x)
            # x: [batch_size, len_series_i, d_model]
            enc_out_list.append(x)
        # enc_out_list: [down_sampling_num + 1, batch_size, len_series_i, d_model]
        
        for i in range(self.enc_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        # enc_out_list: [down_sampling_num + 1, batch_size, len_series_i, d_model]
        dec_out_list = self.multi_scale_mixer(x_list, enc_out_list)
        # dec_out_list: [down_sampling_num + 1, batch_size, len_reduce]
        result = sum(torch.stack(dec_out_list, dim=0))
        # result: [batch_size, len_reduce]
        
        return result
    
    
    def multi_scale_mixer(self, x_list, enc_out_list):
        # x_list: [down_sampling_num + 1, batch_size, len_series_i, 1]
        # enc_out_list: [down_sampling_num + 1, batch_size, len_series_i, d_model]
        
        dec_out_list = []
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            # enc_out: [batch_size, len_series_i, d_model]
            dec_out = self.projection_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            # dec_out: [batch_size, len_reduce, d_model]
            dec_out = self.dim_project_layer(dec_out)
            # dec_out: [batch_size, len_reduce, 1]
            dec_out = dec_out.reshape(-1, self.len_reduce)
            # dec_out: [batch_size, len_reduce]
            dec_out_list.append(dec_out)
        # dec_out_list: [down_sampling_num + 1, batch_size, len_reduce]
        
        return dec_out_list
        
    
    def multi_scale(self, x):
        # x: (batch_size, len_series, 1)
        
        down_pool = nn.AvgPool1d(self.sampling_win_size, stride=self.sampling_win_size, padding=0)
        
        x = x.permute(0, 2, 1)
        # x: [batch_size, 1, len_series]
        
        x_sampling_list = []
        x_sampling_list.append(x.permute(0, 2, 1))
        # x: [1, batch_size, len_series, 1]
        for _ in range(self.down_sampling_num):
            temp = down_pool(x)
            x_sampling_list.append(temp.permute(0, 2, 1))
            x = temp
        # x_sampling_list: [down_sampling_num + 1, batch_size, len_series_i, 1]
        
        x = x_sampling_list
        # x: [down_sampling_num + 1, batch_size, len_series_i, 1]
        return x
        
        
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return self.dropout(x)
        
        
class DecomposableMixer(nn.Module):
    def __init__(self, conf: Configuration):
        super(DecomposableMixer, self).__init__()
        
        self.conf = conf
        self.d_model = conf.getEntry('d_model')
        self.d_ff = conf.getEntry('d_ff')
        self.down_sampling_num = conf.getEntry('down_sampling_num')
        self.dropout = conf.getEntry('dropout')
        self.trend_win = conf.getEntry('trend_win')
        
        self.decomposition = series_decomp(self.trend_win)
        
        self.multi_scale_season_mixer = MultiScaleSeasonMixer(conf)
        self.multi_scale_trend_mixer = MultiScaleTrendMixer(conf)
        
        self.outlayer = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.d_model)
        )
        
    
    def forward(self, enc_out_list):
        # enc_out_list: [down_sampling_num + 1, batch_size, len_series_i, d_model]
        
        length_list = []
        for enc_out in enc_out_list:
            # enc_out: [batch_size, len_series_i, d_model]
            length_list.append(enc_out.shape[1])
        # length_list: [down_sampling_num + 1]
        
        season_list = []
        trend_list = []
        for enc_out in enc_out_list:
            # enc_out: [batch_size, len_series_i, d_model]
            season, trend = self.decomposition(enc_out)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        # season_list: [down_sampling_num + 1, batch_size, d_model, len_series_i]
        # trend_list: [down_sampling_num + 1, batch_size, d_model, len_series_i]
            
        out_season_list = self.multi_scale_season_mixer(season_list)
        # out_season_list: [down_sampling_num + 1, batch_size, len_series_i, d_model]
        out_trend_list = self.multi_scale_trend_mixer(trend_list)
        # out_trend_list: [down_sampling_num + 1, batch_size, len_series_i, d_model]
        
        out_list = []
        for enc_out, out_season, out_trend, length in zip(enc_out_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            # out: [batch_size, len_series_i, d_model]
            out = out + self.outlayer(out)
            # out : [batch_size, len_series_i, d_model]
            out_list.append(out)
        # out_list: [down_sampling_num + 1, batch_size, len_series_i, d_model]
        
        return out_list
            

class MultiScaleSeasonMixer(nn.Module):
    def __init__(self, conf: Configuration):
        super(MultiScaleSeasonMixer, self).__init__()
            
        self.conf = conf
        self.len_series = conf.getEntry('len_series')
        self.down_sampling_num = conf.getEntry('down_sampling_num')
        self.sampling_win_size = conf.getEntry('sampling_win_size')
        
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.len_series // (self.sampling_win_size ** i),
                          self.len_series // (self.sampling_win_size ** (i + 1))),
                nn.GELU(),
                nn.Linear(self.len_series // (self.sampling_win_size ** (i + 1)),
                          self.len_series // (self.sampling_win_size ** (i + 1)))
            ) for i in range(self.down_sampling_num)
        ])
    
    
    def forward(self, season_list):
        # season_list: [down_sampling_num + 1, batch_size, d_model, len_series_i]
        
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        # out_season_list: [1, batch_size, len_series_i, d_model]
        
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            # out_low_res: [batch_size, len_series_i, d_model]
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        # out_season_list: [down_sampling_num + 1, batch_size, len_series_i, d_model]
        
        return out_season_list
        
        
class MultiScaleTrendMixer(nn.Module):
    def __init__(self, conf: Configuration):
        super(MultiScaleTrendMixer, self).__init__()
        
        self.conf = conf
        self.len_series = conf.getEntry('len_series')
        self.down_sampling_num = conf.getEntry('down_sampling_num')
        self.sampling_win_size = conf.getEntry('sampling_win_size')
        
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.len_series // (self.sampling_win_size ** (i + 1)),
                          self.len_series // (self.sampling_win_size ** i)),
                nn.GELU(),
                nn.Linear(self.len_series // (self.sampling_win_size ** i),
                          self.len_series // (self.sampling_win_size ** i))
            ) for i in reversed(range(self.down_sampling_num))
        ])
        
        
    def forward(self, trend_list):
        # trend_list: [down_sampling_num + 1, batch_size, d_model, len_series_i]
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
            
        out_trend_list.reverse()
        
        return out_trend_list