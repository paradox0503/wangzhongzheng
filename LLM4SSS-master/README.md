这个项目的代码结构可以概括为“单入口 + 统一实验框架 + 多模型实现 + 多实验脚本”。

**需要做的**
- 具体环境已经丢失了但是没有刁钻的版本需求。主要就是：
    torch
    numpy
    transformers
    scikit-learn
- 总共是六个数据集，其中`human`就是`RandWalk`数据集。
- 修改`conf/`目录下的json配置文件，修改其中的数据集路径，其他的可以完全不变。
- 执行`run.bash`即可训练模型，训练得到的模型参数会保留在`example`目录下，例如`./example/GPT4SSS/human/model/example_model.pth`。

**代码入口**
- 主入口是 [LLM4SSSsummary_run.py]，只做两件事：读配置、启动实验。
- 命令行脚本在 [run.bash]，修改模型与数据集即可执行。

**核心目录结构（按代码职责）**
- `utils/`
- [utils/conf.py]：JSON 配置读取。
- [utils/expe.py]：实验主流程（setup/train/validate/test）。
- [utils/sample.py]：样本抽样与 `Dataset`。
- [utils/loss.py]：损失函数与误差计算。

- `model/`
- 各模型文件按“一个模型一个文件”组织，如 `GPT4SSS.py`、`TimeLLM.py`、`AutoTimes.py`、`UniTime.py`、`S2IPLLM.py`、`TimeMixer.py`。按照要求只需要测`AutoTimes`和`UniTime`。
- `MyLLM4SSS1~11.py` 是多版自定义模型迭代，没有什么用。
- 公共组件在 `embed.py`、`mlp.py`、`Decompose.py`、`prompt.py`、`unitimegpt2.py`。

- `conf/`
- 按数据集分子目录（`human/`、`F5/`、`F10/`、`astro/`、`deep1B/`、`sald/`），这里需要接数据集的目录，目录下需要有`train.bin`、`val.bin`、`test.bin`。
- 每个 JSON 对应一个“数据集+模型”的实验配置（如 `conf/human/GPT4SSS.json`）。

**数据集**
- 需要数据集（`human/`、`F5/`、`F10/`、`astro/`、`deep1B/`、`sald/`），这里需要在`conf/`里的 json 文件中修改数据集的目录，目录下需要有`train.bin`、`val.bin`、`test.bin`。

**实验与分析相关目录**
- `BSF_Data/`：数据生成脚本（`genData.py`）。
- `BSF_Tightness/`：指标/紧致性分析脚本（`baseline.py`、`tightness.py`）及结果文件。
- `nnCoverage/`：覆盖率相关脚本（`getData.py`、`nnCoverage.py`）。
- `figure/`：画图脚本与结果图（`draw.py`、`extract.py`、`union_draw.py`）。

**结构特点**
- 优点：分层清晰，`conf` + `model` + `utils` 解耦，便于横向比较多模型。
- 风格：偏研究代码组织，结果与中间产物目录（如 `BSF_Tightness/figure/`）也放在仓库内。
