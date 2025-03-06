<div align="center">
<a href="https://github.com/SIA-IDE/BearLLM">
<img src="https://raw.githubusercontent.com/SIA-IDE/BearLLM/refs/heads/main/docs/images/logo.svg" width="200" alt="logo"/>
</a>
<h1>BearLLM: A Prior Knowledge-Enhanced Bearing Health Management Framework with Unified Vibration Signal Representation</h1>

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/Pytorch-latest-orange"></a>
<a href="https://arxiv.org/abs/2408.11281"><img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv-B31B1B"></a>
<a href="https://huggingface.co/datasets/SIA-IDE/MBHM"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-🤗-FFFDF5"></a>
<a href="https://github.com/SIA-IDE/BearLLM"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SIA-IDE/BearLLM"></a>
</div>

<h4 align="center">
    <p>
        <a href="https://github.com/SIA-IDE/BearLLM/blob/main/README.md">English</a> ｜
        <b>简体中文</b>
    </p>
</h4>

## 🔥 新闻
- **[2025-03-06]** 🌟 完整的数据集和代码现已正式开源！
- **[2024-12-11]** ⏫ 我们正在努力将 BearLLM 的代码开源，敬请期待！
- **[2024-12-10]** 🎉 BearLLM 论文已被第三十九届 AAAI 人工智能会议（[AAAI-25](https://aaai.org/conference/aaai/aaai-25/)）接收。
- **[2024-08-21]** 📝 BearLLM 论文的预印本已发布在 arXiv。详情请查看[论文页面](https://arxiv.org/abs/2408.11281)。

## 📅 待办
- [ ] 完善相关注释和文档。
- [x] 上传完整的 BearLLM Demo代码。
- [x] 上传 MBHM 数据集的健康管理语料库。
- [x] 整理 BearLLM 的预训练和微调代码。
- [x] 整理 BearLLM 的分类网络代码及其他对比模型的代码。
- [x] 上传 MBHM 数据集中的振动信号部分。

## 📚 介绍
[MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM) 数据集是首个专为轴承健康管理研究设计的多模态数据集，分为两个部分：振动信号和健康管理语料库。振动信号及其工况信息来自 9 个公开数据集，并在持续更新和改进中。成千上万的工况设置为识别模型带来了更高的挑战，同时也更好地模拟了真实应用场景。

[BearLLM](https://github.com/SIA-IDE/BearLLM) 是一个先验知识增强的轴承健康管理框架，具备统一的振动信号表示。该框架将待测信号转换至频域，以便更有效地识别相较于无故障状态下振动信号的频谱差异。通过对齐振动信号与故障语义嵌入，我们利用低计算开销的微调语言模型，实现了各类健康管理任务的统一自然语言响应。实验结果表明，该框架在成千上万种工况下均能取得领先性能。

## 💻 依赖

代码基于 Python 3.12 实现，所需的依赖包列在 `requirements.txt` 文件中。你可以使用以下命令安装所需的依赖：

```bash
conda create --name bearllm python=3.12
conda activate bearllm
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 下载 Demo 数据 / 使用你自己的数据

首先，你需要下载 [MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM/tree/main) 数据集中的 `demo_data.json`。
对于中国大陆用户，你可以使用 [镜像链接](https://hf-mirror.com/datasets/SIA-IDE/MBHM/tree/main) 加速下载：

或者，你也可以按照相同的格式构建自己的测试数据:  
`instruction`: 健康管理任务文本指令。  
`vib_data`: 需要识别的振动信号数据，需要时间长度为1s。  
`ref_data`: 作为参考的无故障振动信号数据，需要时间长度为1s。

```json
{
    "instruction": "xxx.",
    "vib_data": [1.0, 0.0, 1.0, ...],
    "ref_data": [1.0, 0.0, 1.0, ...],
}
```

### 2. 下载权重

你可以在 [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main) 下载 Qwen2.5-1.5B 的预训练权重（所有文件）。  
此外，你还需要下载 [BearLLM](https://huggingface.co/SIA-IDE/BearLLM/tree/main) 的权重（所有文件）。

### 3. 组织文件

建议将权重和测试数据按以下结构组织：

```
BearLLM/
├── qwen_weights/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   └── 其他文件...
├── bearllm_weights/
│   ├── vibration_adapter.pth
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── mbhm_dataset/
    └── demo_data.json 
```

### 4. 运行代码

首先复制`.env.example`文件复制为`.env`，并修改其中的数据路径。  
然后，你可以使用以下命令运行代码：
```bash
python run_demo.py
```

## ⚙️ 开发

### 1. 下载数据集

首先，你需要下载 [MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM/tree/main) 数据集中的以下文件。
对于中国大陆用户，你可以使用 [镜像链接](https://hf-mirror.com/datasets/SIA-IDE/MBHM/tree/main) 加速下载：

- `data.hdf5`: 包含了振动信号数据集。
- `corpus.json`: 包含了健康管理语料库。
- `metadata.sqlite`: 包含了数据集的元数据信息。

### 2. 下载权重

你可以在 [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main) 下载 Qwen2.5-1.5B 的预训练权重。

### 3. 修改环境变量

将 `.env.example` 文件复制为 `.env`，并修改其中的数据路径。

### 4. 预训练和微调模型

根据 `src/pre_training.py` 进行 FCN 预训练。  
再根据 `src/fine_tuning.py` 进行微调。

## 📖 引用

如果您在研究中使用了本研究，请引用以下论文：

```
@misc{peng2024bearllmpriorknowledgeenhancedbearing,
      title={BearLLM: A Prior Knowledge-Enhanced Bearing Health Management Framework with Unified Vibration Signal Representation}, 
      author={Haotian Peng and Jiawei Liu and Jinsong Du and Jie Gao and Wei Wang},
      year={2024},
      eprint={2408.11281},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.11281}, 
}
```
