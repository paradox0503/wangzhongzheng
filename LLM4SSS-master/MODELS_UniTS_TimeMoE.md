# UniTS / TimeMoE-50M 接入说明

两个模型沿用 `model/<模型名>.py`、`conf/<数据集>/<模型名>.json`、
`example/<模型名>/<数据集>/` 的结构，输入 `[batch_size, len_series]`，
输出 `[batch_size, len_reduce]`。继续使用当前距离损失训练降维表示，
不是运行原论文的预测或分类评测任务。
两个适配器都会将原始序列的均值和标准差与骨干特征拼接后送入投影层，
保留标准化去除的水平和幅度信息，供原始序列距离学习使用。

已补齐 human、F5、F10、astro、deep1B、sald 共 12 个配置，保留各数据集
原有长度及采样数量。新配置默认单卡 `cuda:0`、`GPUs: [0]`、批大小 64；
实际批大小需按服务器显存调整。

## UniTS

- 官方源码：https://github.com/mims-harvard/UniTS
- 将官方仓库放在本项目旁边，或修改 `units_repo_path`。适配器直接加载
  `<units_repo_path>/models/UniTS.py`，不会将其 `utils` 加入 Python 搜索路径。
- 在实际运行的服务器上安装官方依赖（包括 PyTorch、timm）；当前机器未安装。
- 默认维度 128、3 层、8 个注意力头、10 个提示 token，patch_len=stride=16。
- `units_checkpoint` 默认为空，表示从头训练官方 UniTS 结构，**不代表已加载预训练模型**。
  使用预训练权重时填写本地 checkpoint 文件路径，模型维度、层数和 patch 长度
  必须与权重一致。支持原始 state_dict，以及 `state_dict` / `model` 包装和 `module.` 前缀。
- 预训练多任务模型中的数据集专属 token 可能不匹配新任务，这些 token 保持新初始化；
  共享 patch embedding、位置编码、所有 backbone blocks 和分类特征头必须完整匹配，
  否则明确报错，不会静默忽略不兼容的骨干权重。
- 默认 `freeze_backbone: false`，训练骨干、提示 token 和投影层。
  若设为 true，需提供预训练权重；此时只训练投影层，新任务 token 也会被冻结。
- 数据经过官方 tokenize、classification prompt、backbone 和 CLS 特征头，
  再通过线性投影得到降维向量；未使用类别 logits 作为特征。

## TimeMoE（base / 50M）

- 指定权重：https://huggingface.co/Maple728/TimeMoE-50M
- 官方实现：https://github.com/Time-MoE/Time-MoE
- `time_moe_path` 指向下载完整的本地模型目录，应包含权重、config.json 和
  Hugging Face 仓库提供的全部自定义 Python 模型文件。
- 默认 `local_files_only: true`，运行时从本地加载。若服务器需要在线加载，
  可改成 false，并将 `time_moe_path` 设为 `Maple728/TimeMoE-50M`。
- 使用官方 `AutoModelForCausalLM` 自定义模型加载方式（`trust_remote_code=True`），
  再取内部 decoder 特征。依赖 Transformers 版本需兼容官方 TimeMoE；
  50M 的官方配置记录版本为 4.40.1，新版本兼容性未经本地验证。
- 校验 50M 的结构参数，避免误用 200M 等模型；不裁剪预训练层数。
- 输入逐序列标准化，提取最后一个因果 token 的隐藏表示，再投影到 `len_reduce`。
  默认冻结骨干，仅训练投影层；`freeze_backbone: false` 可开启骨干微调，
  但仍只使用本项目距离损失，未加入官方预测损失或 MoE 路由辅助损失。
- 使用 float32 加载以兼容当前训练器的 AMP；关闭 KV cache，使用 eager attention，
  无需额外安装 FlashAttention。

## 运行方式（在数据和环境准备好的服务器上）

从 `LLM4SSS-master` 目录执行，先修改 JSON 中的数据和模型路径：

原始数据使用平铺文件名，保留服务器现有文件，无需重命名。例如：

```json
"data_path": "/data/user_jialinhan/data_big/",
"dataset_selected": "astro"
```

训练读取 `/data/user_jialinhan/data_big/astro-dataset.bin`，导出脚本还会读取
同目录下的 `astro-query.bin`。目录末尾的斜杠可省略。
`deep1B` 配置自动对应文件前缀 `deep1b`，其余数据集名称原样用作文件前缀。
该规则应用于 `utils/sample.py`、`BSF_Data/genData.py` 和 `nnCoverage/getData.py`。

```bash
python -u LLM4SSSsummary_run.py -C conf/human/UniTS.json
python -u LLM4SSSsummary_run.py -C conf/human/TimeMoE.json
```

也可将原 `run.bash` 中的 model 改成 `UniTS` 或 `TimeMoE`。
训练入口会为新模型建立输出目录；`mkdir.bash` 也已加入两个模型。
新模型在训练、验证阶段分别切换 train/eval 模式；数据导出时使用 eval 模式，
避免 dropout 导致同一序列导出不同向量。单卡和 DataParallel 均保存无包装前缀的权重。
参数/FLOP 统计、`BSF_Data/genData.py`、`nnCoverage/getData.py` 均已注册两个名称。
这些分析脚本仍沿用原来的数据路径及运行要求；FLOP 工具对动态路由等算子的
统计可能不完整，需在实际环境查看 unsupported operator 提示。

原始 `*-dataset.bin` 和 `*-query.bin` 仅被读取。每次运行仍会覆盖配置指定的抽样文件、索引、
日志和同名训练 checkpoint；应为不同实验使用不同输出路径。

## 验证范围

按要求未运行测试、训练、推理或模型下载，也未安装依赖。
仅核对官方接口、代码、配置与改动差异；运行兼容性及模型效果尚未实测。
