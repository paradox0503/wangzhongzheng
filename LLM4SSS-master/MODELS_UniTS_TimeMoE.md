# UniTS / TimeMoE-50M 接入说明

两个模型沿用 `model/<模型名>.py`、`conf/<数据集>/<模型名>.json`、
`example/<模型名>/<数据集>/` 的结构，输入 `[batch_size, len_series]`，
输出 `[batch_size, len_reduce]`。继续使用当前距离损失训练降维表示，
不是运行原论文的预测或分类评测任务。
两个适配器都使用原始序列分块，不额外标准化或拼接统计量。
TimeMoE 对齐 AutoTimes 的编码、骨干、解码及投影流程；UniTS 对齐
GPT4SSS 的分块特征、骨干、展平及投影流程。保留各自骨干，不保证相同耗时或效果。

已补齐 origin、F5、seismic、astro、deep1b、sald 共 12 个配置，保留各数据集
原有长度及采样数量。新配置默认单卡 `cuda:0`、`GPUs: [0]`、批大小 64；
实际批大小需按服务器显存调整。

## UniTS

- 官方源码：https://github.com/mims-harvard/UniTS
- 将官方仓库放在本项目旁边，或修改 `units_repo_path`。适配器直接加载
  `<units_repo_path>/models/UniTS.py`，不会将其 `utils` 加入 Python 搜索路径。
- 在实际运行的服务器上安装官方依赖（包括 PyTorch、timm）；当前机器未安装。
- 默认维度 128、3 层、8 个注意力头、10 个提示 token。
  分块与同数据集 AutoTimes 一致：sald/deep1b 为 8，其余为 16。
- `units_checkpoint` 默认为空，表示从头训练官方 UniTS 结构，**不代表已加载预训练模型**。
  使用预训练权重时填写本地 checkpoint 文件路径，模型维度、层数和 patch 长度
  必须与权重一致。支持原始 state_dict，以及 `state_dict` / `model` 包装和 `module.` 前缀。
- 预训练多任务模型中的数据集专属 token 可能不匹配新任务，这些 token 保持新初始化；
  共享 patch embedding、位置编码和所有 backbone blocks 必须完整匹配，
  否则明确报错，不会静默忽略不兼容的骨干权重。
- 默认 `freeze_backbone: false`，训练骨干、提示 token 和投影层。
  若设为 true，需提供预训练权重；此时只训练投影层，新任务 token 也会被冻结。
- 数据经过官方 patch embedding、位置编码和 backbone，取所有数据 token
  展平后线性投影。不再调用 tokenize 的逐序列标准化，也不再计算 CLS token/分类头。
  保留 10 个提示前缀，因为官方动态 MLP 的固定输入权重需要它们。
  128 点输入为 16 个数据 token 加 10 个提示 token；96 点为 12+10。
- 新投影层与旧 checkpoint 不兼容，任务权重目录改为 `model_patched/`。
  加载官方预训练 checkpoint 时，patch_len/维度/层数仍必须与权重一致；
  默认 `units_checkpoint` 为空的实验仍是随机初始化，不能描述为预训练 UniTS。

## TimeMoE（base / 50M）

- 指定权重：https://huggingface.co/Maple728/TimeMoE-50M
- 官方实现：https://github.com/Time-MoE/Time-MoE
- `time_moe_path` 指向下载完整的本地模型目录，应包含权重、config.json 和
  Hugging Face 仓库提供的全部自定义 Python 模型文件。
- 默认 `local_files_only: true`，运行时从本地加载。可恢复的导出要求先下载固定模型
  快照并将 `time_moe_path` 指向本地目录，避免远程模型更新后续写到旧向量文件。
- 使用官方 `AutoModelForCausalLM` 自定义模型加载方式（`trust_remote_code=True`），
  再取内部 decoder 特征。依赖 Transformers 版本需兼容官方 TimeMoE；
  50M 的官方配置记录版本为 4.40.1，新版本兼容性未经本地验证。
- 校验 50M 的结构参数，避免误用 200M 等模型；不裁剪预训练层数。
- 输入处理与 AutoTimes 对齐：原始序列 → `unfold(win_size, stride)` →
  项目已有的 `Mlp` 编码器 → 官方 decoder 的 `inputs_embeds` → `Mlp` 解码器
  → 展平到 `len_series` → 线性投影到 `len_reduce`。
  绕过官方逐点输入 embedding，但保留全部预训练 decoder 层；这是本项目的
  分块适配实验，不是官方逐点输入的零样本预测。
- 六份配置的窗口、步长、MLP 和 dropout 设置与同数据集 AutoTimes 一致。
  `sald` 为 128/8=16 tokens，`deep1b` 为 96/8=12 tokens；
  `origin`、`F5`、`astro`、`seismic` 为 256/16=16 tokens。
  要求 `win_size == stride > 0` 且长度整除窗口，避免漏掉尾部或改变样本数。
- 默认冻结骨干，训练编码器、解码器与投影层。冻结不会切断编码器的梯度；
  `freeze_backbone: false` 可开启骨干微调，
  但仍只使用本项目距离损失，未加入官方预测损失或 MoE 路由辅助损失。
- 使用 float32 加载以兼容训练器与导出 autocast；关闭 KV cache。
  默认 `attn_implementation: eager`，无需安装 FlashAttention。
  官方模型不支持 SDPA，不能直接填写 `sdpa`；服务器已安装兼容 FlashAttention2
  时可显式选择 `--attn-implementation flash_attention_2`，其兼容性和收益需实测。
  导出时该后端必须搭配 CUDA BF16/FP16；FP32 会明确报错，避免官方后端暗中转换精度。

### 2026-10-30 分块版本迁移

旧版逐点模型的任务 checkpoint 不兼容新的编码器、解码器和投影层。
新配置使用 `model_patched/` 保存任务权重，原 `model/` 保留；若使用 checkpoint
模式需重新训练。`--weights pretrained` 与 AutoTimes 一样包含未训练的任务层，
不能当作已经训练好的距离表示。

正在运行的进程不会自动切换新代码。切换后必须换一个新的导出目录；
不要续写旧版 `.partial`，也不要混用新旧 dataset/query 向量。
从项目根目录导出，例如：

```bash
python -u export_embeddings.py -C conf/sald/TimeMoE.json --weights pretrained --batch-size 128 --output-dir example/TimeMoE/embeddings_patched_v2
```

使用 `runall.bash` 时也将 `output_dir` 改为新目录。比较速度时保持相同设备、
batch size、精度和数据量；token 数减少 8～16 倍不等于实测耗时减少相同倍数。

### 推理选项与断点续跑

TimeMoE/UniTS 的配置现在默认 `inference_precision: auto`：支持 BF16 的 CUDA
GPU 使用 autocast BF16，其他 GPU/CPU 保持 FP32。参数和输出文件仍为 float32。
`--precision fp32` 可作对照；`--precision bf16/fp16` 为显式选择，半精度可能改变
向量数值和检索效果，应对比同批数据后再决定。AutoTimes/GPT4SSS 默认仍为 FP32，
可使用相同 CLI 选项公平比较。导出统一关闭未用的 KV cache、attention/hidden-state 输出。

默认新模型输出到 `example/<model>/embeddings_patched_v2/`，显式 `--output-dir`
优先。导出目录内新增 `<dataset>-export.json`，记录源文件大小/mtime、配置、seed、
实际精度、导出行数策略、代码哈希和本地权重大小/mtime；改变这些内容必须使用新目录。
无身份记录的旧文件也会被拒绝续写，不会覆盖或删除。大文件使用元数据识别，
不是全量内容校验；替换数据/权重时不要人为保留旧大小与mtime。
TimeMoE 可恢复导出强制使用本地权重目录。

每个输出使用操作系统文件锁，进程退出后锁自动释放；小型 `.lock` 文件保留，
无需删除。两份输出已完成且身份一致时，跳过模型加载和推理。
不得用不同代码/精度重复生成 dataset 与 query 后混合使用。

保留服务器的导出策略：所有 dataset 默认只导出前 100,000,000 个完整行，
query 导出全部完整行。`--max-dataset-rows N` 可调整上限，`N <= 0` 表示不限量；
它不影响 query。源文件末尾不足一行的数据会给出警告并跳过，原文件不修改。
续跑和已完成判断使用同一行数上限；修改上限也需要新输出目录。

服务器可先用 batch 128 跑新目录，观察 rows/s 和显存，再在相同配置下尝试 256/512。
没有 GPU 实测，不能预设最大 batch 或承诺加速倍数。示例（从项目根目录执行）：

```bash
python -u export_embeddings.py -C conf/sald/TimeMoE.json --batch-size 256 --precision auto
python -u export_embeddings.py -C conf/sald/UniTS.json --batch-size 256 --precision auto
# FP32 对照必须用另一目录；batch 256 不是自动测得的最优值。
python -u export_embeddings.py -C conf/sald/TimeMoE.json --batch-size 256 --precision fp32 --output-dir example/TimeMoE/embeddings_fp32_v2
```

## 运行方式（在数据和环境准备好的服务器上）

从 `LLM4SSS-master` 目录执行，先修改 JSON 中的数据和模型路径：

原始数据使用平铺文件名，保留服务器现有文件，无需重命名。例如：

```json
"data_path": "/data/user_jialinhan/data_big/",
"dataset_selected": "astro"
```

训练读取 `/data/user_jialinhan/data_big/astro-dataset.bin`，导出脚本还会读取
同目录下的 `astro-query.bin`。目录末尾的斜杠可省略。
配置目录和 dataset 名称统一为小写 `deep1b`（Linux 区分大小写）。
读原始数据时仍兼容旧配置中的 `deep1B`，对应文件前缀 `deep1b`。
已有 `example/`、`BSF_Data/`、`BSF_Tightness/` 下的大写 `deep1B` 结果未迁移或删除；
新版小写路径不会自动复用这些旧结果，复用历史分析时需明确选择原路径。
该规则应用于 `utils/sample.py`、`BSF_Data/genData.py` 和 `nnCoverage/getData.py`。

合并服务器提交时保留了服务器 GPU 分配及脚本选择：`runall.bash` 默认运行
TimeMoE 的 sald/seismic；原 `runAll.bash` 使用不与小写脚本冲突的名称
`runAll_models.bash`。`run2.bash` 保留；运行日志和 PID 不纳入源码。

```bash
python -u LLM4SSSsummary_run.py -C conf/origin/UniTS.json
python -u LLM4SSSsummary_run.py -C conf/origin/TimeMoE.json
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

初始接入未运行测试、训练、推理或模型下载，也未安装依赖。
2026-10-30 首轮 TimeMoE 修正仅运行配置检查及语法检查，七个张量测试因无 PyTorch 跳过。
后续 UniTS 对齐、混合精度和断点身份修正按用户要求仅做静态检查，没有运行新测试。
服务器测试覆盖分块、末批、梯度、精度选择和断点身份；UniTS 测试使用官方源码，
若路径不同需设置 `UNITS_REPO_PATH`。服务器可运行：

```bash
python -m unittest discover -s tests -v
```

TimeMoE 的替身测试不等同于真实预训练权重验证；真实权重兼容性、GPU 吞吐和效果仍需实测。
