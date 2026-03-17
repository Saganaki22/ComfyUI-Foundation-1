<div align="center">
  <img src="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/Charts/banner.PNG" alt="Foundation-1 Banner" width="100%">
</div>

<div align="center">
  <h1>ComfyUI-Foundation-1</h1>

  <p>
    ComfyUI 自定义节点<br>
    <b>Foundation-1 — 音乐制作的结构化文本转采样扩散模型</b>
  </p>
  <p>
    <a href="https://huggingface.co/RoyalCities/Foundation-1"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-模型-blue' alt="HF Model"></a>
    <a href="https://github.com/Stability-AI/stable-audio-tools"><img src="https://img.shields.io/badge/GitHub-stable--audio--tools-green" alt="GitHub"></a>
    <a href="#license"><img src="https://img.shields.io/badge/许可证-Stability%20AI%20社区-yellow" alt="License"></a>
    <img src="https://img.shields.io/badge/显存-8GB%2B%20推荐-orange" alt="VRAM">
    <img src="https://img.shields.io/badge/速度-%7E7--8秒%20(RTX%203090)-brightgreen" alt="Speed">
  </p>
</div>

---

## 概述

**Foundation-1** 是一个用于音乐制作的结构化文本转采样扩散模型。它能够理解乐器身份、音色、效果、音乐记谱、BPM、小节数和调式作为独立的可组合控制——实现精确、可预测的音乐循环合成。

本 ComfyUI 封装提供原生节点集成：
- **结构化提示词** — 乐器、音色、效果和记谱标签
- **节拍同步生成** — 根据 BPM 和小节数自动计算时长
- **调式感知合成** — 支持完整的西方调式
- **原生进度条** — 支持中断操作

> **介绍视频：** [观看 Foundation-1 概述和设计理念](https://www.youtube.com/watch?v=O2iBBWeWaL8)

---

## 特性

- 结构化文本转采样 — 从结构化文本提示生成音乐循环
- 节拍同步时长 — 根据 BPM 和小节数自动计算时长
- 24 种调式 — 完整的西方调式支持（大调和小调）
- 原生 ComfyUI 集成 — AUDIO 输出、进度条、中断支持
- 优化性能 — 支持 SDPA、FlashAttention 2、SageAttention
- 智能自动下载 — 首次使用时自动从 HuggingFace 下载模型权重
- 智能缓存 — 可选在运行之间将模型卸载到 CPU 内存

---

## 系统要求

- **显卡：** NVIDIA 显卡，**8GB 显存最低要求**（必须支持 CUDA）
  - 典型显存占用：生成时约 **~7GB**
  - 生成速度：**~20 it/s**（每秒迭代次数），使用默认采样器
- **CPU/MPS：** 不支持 — Foundation-1 使用 Flash Attention，仅支持 CUDA
- **Python：** 3.10+
- **CUDA：** 11.8+
- **Flash Attention：** 必需（PyTorch 2.0+ SDPA 自带）
- **SageAttention：** 可选但推荐（已在 2.2.0 版本测试）

> [!WARNING]
> **仅支持 CUDA**
> 
> Foundation-1 需要 CUDA。模型架构使用 Flash Attention 进行滑动窗口注意力计算，没有 CPU 回退方案。不支持 CPU 和 MPS 设备。
>
> **注意力要求：**
> - **最低要求：** Flash Attention 2（PyTorch 2.0+ SDPA 后端内置）
> - **推荐：** SageAttention 2.2.0+ 以获得更好性能

---

## 安装

<details>
<summary><b>点击展开安装方法</b></summary>

### 方法 1：ComfyUI Manager（推荐）

1. 打开 ComfyUI Manager
2. 搜索 "Foundation-1"
3. 点击安装
4. 重启 ComfyUI

### 方法 2：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-Foundation-1.git
cd ComfyUI-Foundation-1
pip install -r requirements.txt
```

> **注意：** 依赖项在 ComfyUI 启动时自动安装。有关安装内容和原因，请参阅下方的[依赖详情](#依赖详情)部分。

---

## 依赖详情

<details>
<summary><b>点击展开依赖安装详情</b></summary>

所有依赖项在 ComfyUI 启动时自动安装。您**无需**手动运行 `pip install`。

### 已包含在 ComfyUI 中
这些包通常已在 ComfyUI 环境中存在：
```
torch
torchaudio      # Foundation-1 音频处理必需
numpy
safetensors
transformers    # T5 文本编码器
huggingface_hub # 模型下载
```

如果由于某种原因缺少 `torchaudio`，请手动安装：
```bash
pip install torchaudio
```

### 正常 pip 安装（自动安装）
这些包正常安装：
```
einops>=0.7.0
alias-free-torch>=0.0.6
ema-pytorch>=0.2.3
einops-exts>=0.0.3
```

### 使用 --no-deps 特殊安装
这些包需要特殊处理：

| 包名 | 安装命令 | 原因 |
|------|----------|------|
| `stable-audio-tools` | `pip install stable-audio-tools --no-deps` | 避免 `pandas==2.0.2`，该版本没有 Python 3.13 wheel，从源码构建会失败 |
| `k-diffusion` | `pip install k-diffusion==0.1.1 --no-deps --target ./k_diffusion_files/` | 安装到私有目录，避免与 ComfyUI 捆绑的 k_diffusion 冲突以及 `clip→pkg_resources` 导入链问题 |

### 不要手动安装的内容

> [!WARNING]
> **请勿运行以下命令：**
> ```bash
> pip install stable-audio-tools      # 错误 - 会拉取 pandas==2.0.2
> pip install k-diffusion             # 错误 - 会与 ComfyUI 版本冲突
> ```
>
> 这些在启动时使用正确的标志自动处理。

### 可选包
- `sageattention` — 手动安装以获得更好性能：`pip install sageattention`

</details>

### 安装 SageAttention（推荐）

```bash
pip install sageattention
```

已在 SageAttention 2.2.0 版本测试通过。

</details>

---

## 快速开始

### 基本工作流

1. **添加模型加载器**
   - 添加 `Foundation-1 Model Loader` 节点
   - 首次使用时自动从 [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1) 下载模型
   - 选择注意力类型（auto/sdpa/flash/sageattention）

2. **添加生成器**
   - 添加 `Foundation-1 Generate` 节点
   - 连接加载器的模型输出
   - 输入标签：`Synth Lead, Warm, Bright, Melody`
   - 选择 BPM、小节数和调式

3. **运行！**
   - 执行工作流
   - 音频输出可连接其他 ComfyUI 音频节点

---

## 节点参考

### Foundation-1 Model Loader

加载 Foundation-1 检查点并准备生成。

**输入：**
| 参数 | 类型 | 描述 |
|------|------|------|
| `model` | 下拉菜单 | Foundation-1 检查点（首次运行自动下载） |
| `attention` | 下拉菜单 | 注意力机制：`auto`、`sdpa`、`flash_attention_2`、`sageattention` |

**输出：**
| 输出 | 类型 | 描述 |
|------|------|------|
| `model` | FOUNDATION1_MODEL | 加载的模型，用于生成器节点 |

---

### Foundation-1 Generate

生成节拍同步的音乐循环。

**输入：**
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model` | FOUNDATION1_MODEL | — | 连接模型加载器 |
| `tags` | STRING | `Synth Lead, Warm, ...` | 乐器、音色、效果、记谱标签 |
| `bpm` | 下拉菜单 | `140 BPM` | 速度（100-150 BPM 选项） |
| `bars` | 下拉菜单 | `8 Bars` | 循环长度（4 或 8 小节） |
| `key` | 下拉菜单 | `E minor` | 调式（24 个选项） |
| `steps` | INT | 250 | 扩散步数（10-500） |
| `cfg_scale` | FLOAT | 7.0 | 无分类器引导（1.0-15.0） |
| `seed` | INT | 0 | 生成种子 |
| `sampler_type` | 下拉菜单 | `dpmpp-3m-sde` | 扩散采样器 |
| `sigma_min` | FLOAT | 0.3 | 最小噪声级别 |
| `sigma_max` | FLOAT | 500.0 | 最大噪声级别 |
| `unload_after_generate` | BOOLEAN | False | 生成后卸载到 CPU 内存 |
| `torch_compile` | BOOLEAN | False | 启用 torch.compile（首次运行较慢） |

**输出：**
| 输出 | 类型 | 描述 |
|------|------|------|
| `audio` | AUDIO | 生成的音频波形 |
| `prompt` | STRING | 完整的组合提示词 |

---

## 提示词标签

<details>
<summary><b>点击展开标签参考</b></summary>

Foundation-1 使用结构化标签精确控制生成。标签应描述：
- **乐器** — 如 `Synth Lead`、`Piano`、`Guitar`、`Drums`
- **音色** — 如 `Warm`、`Bright`、`Dark`、`Rich`、`Clean`
- **效果** — 如 `Reverb`、`Delay`、`Distortion`、`Chorus`
- **记谱** — 如 `Arp`、`Chord`、`Melody`、`Bassline`
- **特征** — 如 `Spacey`、`Intimate`、`Wide`、`Thick`

**示例提示词：**
```
Synth Lead, Warm, Wide, Bright, Clean, Melody
Piano, Soft, Intimate, Reverb, Chord Progression
Drums, Punchy, Tight, Kick, Snare, Hi-Hat
Bass, Deep, Sub, Rolling, Groove
```

> **注意：** BPM、小节数和调式通过下拉菜单控制 — 不要在标签字段中包含它们。

</details>

### 📋 完整标签参考

完整支持的标签列表，请参阅 **[标签参考表 (Master Tag Reference)](./Master_Tag_Reference.md)**。

### 标签分布图表

<details>
<summary><b>点击展开标签分布图表</b></summary>

<div align="center">
  <img src="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/Charts/subfamilites_pie.PNG" alt="子类乐器分布" width="80%">
  <p><em>乐器子类覆盖范围</em></p>
</div>

<div align="center">
  <img src="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/Charts/timbre_tags_pie.PNG" alt="音色标签分布" width="80%">
  <p><em>音色描述符覆盖范围</em></p>
</div>

<div align="center">
  <img src="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/Charts/fx_pie.PNG" alt="效果标签分布" width="80%">
  <p><em>效果描述符覆盖范围</em></p>
</div>

</details>

---

## 调式

<details>
<summary><b>点击展开支持的调式</b></summary>

**大调：**
C major, C# major, D major, Eb major, E major, F major, F# major, G major, Ab major, A major, Bb major, B major

**小调：**
C minor, C# minor, D minor, D# minor, E minor, F minor, F# minor, G minor, G# minor, A minor, Bb minor, B minor

</details>

---

## 时长计算

时长根据 BPM 和小节数自动计算：

```
时长（秒）= round(小节数 x 4 / BPM x 60)
```

**示例：**
| BPM | 小节 | 时长 |
|-----|------|------|
| 100 | 8 | 19秒 |
| 120 | 4 | 8秒 |
| 140 | 8 | 14秒 |
| 150 | 4 | 6秒 |

**最大时长：** 20 秒（模型限制）

---

## 文件结构

```
ComfyUI/
├── models/
│   └── stable_audio/
│       └── Foundation-1/              # 自动下载
│           ├── Foundation_1.safetensors
│           └── model_config.json
└── custom_nodes/
    └── ComfyUI-Foundation-1/
        ├── __init__.py
        ├── nodes/
        │   ├── __init__.py
        │   ├── loader_node.py
        │   ├── generate_node.py
        │   └── model_cache.py
        ├── k_diffusion_files/         # 私有 k-diffusion 安装
        ├── pyproject.toml
        ├── requirements.txt
        └── README.md
```

---

## 参数详解

<details>
<summary><b>点击展开参数详情</b></summary>

| 参数 | 描述 | 推荐值 |
|------|------|--------|
| **attention** | 注意力机制 | `auto`（有 SageAttention 则用，否则 SDPA） |
| **steps** | 扩散步数 | `250`（训练默认），`100-150` 更快 |
| **cfg_scale** | 无分类器引导 | `7.0`（训练默认），`6-8` 平衡 |
| **sampler_type** | 扩散采样器 | `dpmpp-3m-sde`（推荐，质量最佳），`k-dpm-fast`（更快，需要更少步数） |
| **sigma_min** | 最小噪声级别 | `0.3`（默认） |
| **sigma_max** | 最大噪声级别 | `500.0`（默认） |
| **unload_after_generate** | 卸载到 CPU 内存 | `True` 可在运行之间释放显存 |
| **torch_compile** | torch.compile 优化 | `True`（首次慢，后续更快） |

</details>

---

## 故障排除

<details>
<summary><b>点击展开故障排除指南</b></summary>

### 模型无法下载？

手动从 [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1) 下载：
```bash
pip install -U huggingface_hub
huggingface-cli download RoyalCities/Foundation-1 --local-dir ComfyUI/models/stable_audio/Foundation-1
```

只需要这两个文件：
- `Foundation_1.safetensors`（~3GB 模型权重）
- `model_config.json`（模型配置）

### 依赖安装失败？

`__init__.py` 会在启动时自动安装依赖。如果失败，请手动安装：

**正常 pip 安装：**
```bash
pip install einops>=0.7.0
pip install alias-free-torch
pip install ema-pytorch
pip install einops-exts
```

**使用 `--no-deps` 特殊安装（必须！）：**

这些包必须使用 `--no-deps` 安装，否则会破坏 ComfyUI 环境：

```bash
# stable-audio-tools --no-deps 避免 pandas==2.0.2（没有 Python 3.13 wheel）
pip install stable-audio-tools --no-deps

# k-diffusion 必须安装到私有文件夹（避免与 ComfyUI 捆绑版本冲突）
pip install k-diffusion==0.1.1 --no-deps --target ComfyUI/custom_nodes/ComfyUI-Foundation-1/k_diffusion_files/
```

> [!WARNING]
> **请勿运行：**
> ```bash
> pip install stable-audio-tools    # 错误 - 会拉取 pandas==2.0.2
> pip install k-diffusion           # 错误 - 会与 ComfyUI 冲突
> ```

### k_diffusion_files/ 文件夹是什么？

`k_diffusion_files/` 文件夹由自动安装器自动创建。它包含 `k-diffusion` 的私有副本，在运行时通过 `importlib` 加载——这可以防止与 ComfyUI 自带的 `k_diffusion` 冲突，并避免 `clip→pkg_resources` 导入链问题。

如果此文件夹丢失或损坏，节点会在下次启动时自动重新下载 `k-diffusion==0.1.1`。

### 显存不足？

- 启用 `unload_after_generate=True` 卸载到 CPU 内存
- 减少 `steps`（100-150 仍有良好效果）
- 关闭其他 GPU 应用程序

### 生成速度慢？

- 安装 SageAttention：`pip install sageattention`
- 启用 `torch_compile=True`（首次运行较慢，后续更快）
- 使用 `dpmpp-2m-sde` 采样器（比 `dpmpp-3m-sde` 稍快）

### k_diffusion 冲突？

Foundation-1 将 k-diffusion 安装到私有目录（`k_diffusion_files/`）以避免与 ComfyUI 捆绑版本冲突。请勿手动将 k-diffusion 安装到 site-packages。

</details>

---

## 🔗 重要链接

### 🤗 HuggingFace
- **模型：** [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1)

### 📄 代码
- **推理引擎：** [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)

### 🌐 社区
- **介绍视频：** [Foundation-1 概述](https://www.youtube.com/watch?v=O2iBBWeWaL8)

---

## 📄 许可证

<a name="license"></a>

本模型采用 **Stability AI 社区许可证** 授权：
- ✅ **非商业用途** — 允许
- ✅ **有限商业用途** — 年收入低于 100 万美元的实体
- ⚠️ **收入超过 100 万美元** — 请参阅仓库许可证文件了解完整条款

来自 [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1) 的模型权重受相同许可证约束

---

## ⚠️ 使用声明

Foundation-1 旨在用于音乐制作、创意应用和合法目的。请负责任地、合乎道德地使用。我们对任何非法使用不承担任何责任。请遵守您当地关于生成内容的法律

---

<div align="center">
    <b>音乐制作的结构化文本转采样扩散模型</b>
</div>
