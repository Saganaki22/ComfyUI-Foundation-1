<div align="center">
  <img src="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/Charts/banner.PNG" alt="Foundation-1 Banner" width="100%">
</div>

<div align="center">
  <h1>ComfyUI-Foundation-1</h1>

  <p>
    ComfyUI custom nodes for<br>
    <b>Foundation-1 — Structured Text-to-Sample Diffusion for Music Production</b>
  </p>
  <p>
    <a href="https://huggingface.co/RoyalCities/Foundation-1"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF Model"></a>
    <a href="https://github.com/Stability-AI/stable-audio-tools"><img src="https://img.shields.io/badge/GitHub-stable--audio--tools-green" alt="GitHub"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-Stability%20AI%20Community-yellow" alt="License"></a>
    <img src="https://img.shields.io/badge/VRAM-8GB%2B%20Recommended-orange" alt="VRAM">
    <img src="https://img.shields.io/badge/Speed-%7E7--8s%20(RTX%203090)-brightgreen" alt="Speed">
  </p>
</div>

---

## Overview

**Foundation-1** is a structured text-to-sample diffusion model for music production. It understands instrument identity, timbre, FX, musical notation, BPM, bar count, and key as separate composable controls — enabling precise, predictable synthesis of musical loops.

This ComfyUI wrapper provides native node-based integration with:
- **Structured prompting** with instrument, timbre, FX, and notation tags
- **Tempo-synced generation** with BPM and bar count controls
- **Key-aware synthesis** with full western key support
- **Native progress bars** and interruption support

> **Companion Video:** [Watch the Foundation-1 overview and design philosophy](https://www.youtube.com/watch?v=Foundation1Video)

---

## Features

- Structured Text-to-Sample — Generate musical loops from structured text prompts
- Tempo-Synced Duration — Automatic duration calculation from BPM and bar count
- 24 Musical Keys — Full western key support (major and minor)
- Native ComfyUI Integration — AUDIO noodle outputs, progress bars, interruption support
- Optimized Performance — Support for SDPA, FlashAttention 2, SageAttention
- Smart Auto-Download — Model weights auto-downloaded from HuggingFace on first use
- Smart Caching — Optional model offloading to CPU RAM between runs

---

## Requirements

- **GPU:** NVIDIA GPU with **8GB VRAM minimum** (CUDA required)
  - Typical VRAM usage: **~7GB** during generation
  - Generation speed: **~20 it/s** (iterations per second) with default sampler
- **CPU/MPS:** Not supported — Foundation-1 uses Flash Attention which is CUDA-only
- **Python:** 3.10+
- **CUDA:** 11.8+
- **Flash Attention:** Required (comes with PyTorch 2.0+ SDPA)
- **SageAttention:** Optional but recommended (tested on 2.2.0)

> [!NOTE]
> **Attention Requirements:**
> - **Minimum:** Flash Attention 2 (built into PyTorch 2.0+ SDPA backend)
> - **Recommended:** SageAttention 2.2.0+ for better performance

---

## Installation

<details>
<summary><b>Click to expand installation methods</b></summary>

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "Foundation-1"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-Foundation-1.git
cd ComfyUI-Foundation-1
pip install -r requirements.txt
```

> **Note:** Dependencies are auto-installed at startup. See the [Dependency Details](#dependency-details) section below for what gets installed and why.

---

## Dependency Details

<details>
<summary><b>Click to expand dependency installation details</b></summary>

All dependencies are automatically installed at ComfyUI startup. You do **not** need to run `pip install` manually.

### Already included in ComfyUI
These packages are typically already present in ComfyUI environments:
```
torch
torchaudio      # Required for Foundation-1 audio processing
numpy
safetensors
transformers    # T5 text encoder
huggingface_hub # Model downloads
```

If `torchaudio` is missing for some reason, install it manually:
```bash
pip install torchaudio
```

### Normal pip installs (auto-installed)
These packages are installed normally:
```
einops>=0.7.0
alias-free-torch>=0.0.6
ema-pytorch>=0.2.3
einops-exts>=0.0.3
```

### Special installs with --no-deps
These packages require special handling:

| Package | Install Command | Reason |
|---------|-----------------|--------|
| `stable-audio-tools` | `pip install stable-audio-tools --no-deps` | Avoids `pandas==2.0.2` which has no Python 3.13 wheel and fails to build from source |
| `k-diffusion` | `pip install k-diffusion==0.1.1 --no-deps --target ./k_diffusion_files/` | Installed to private directory to avoid conflicts with ComfyUI's bundled k_diffusion and the `clip->pkg_resources` import chain issue |

### What NOT to install manually

> [!WARNING]
> **Do NOT run these commands:**
> ```bash
> pip install stable-audio-tools      # WRONG - will pull pandas==2.0.2
> pip install k-diffusion             # WRONG - will conflict with ComfyUI's version
> ```
>
> These are handled automatically at startup with the correct flags.

### Optional packages
- `sageattention` — Install manually for better performance: `pip install sageattention`

</details>

### Installing SageAttention (Recommended)

```bash
pip install sageattention
```

Tested and working with SageAttention 2.2.0.

</details>

---

## Quick Start

### Basic Workflow

1. **Add Model Loader**
   - Add `Foundation-1 Model Loader` node
   - Model auto-downloads from [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1) on first use
   - Select attention type (auto/sdpa/flash/sageattention)

2. **Add Generator**
   - Add `Foundation-1 Generate` node
   - Connect model output from loader
   - Enter tags: `Synth Lead, Warm, Bright, Melody`
   - Select BPM, bars, and key

3. **Run!**
   - Execute the workflow
   - Audio output ready for ComfyUI audio nodes

---

## Node Reference

### Foundation-1 Model Loader

Loads a Foundation-1 checkpoint and prepares it for generation.

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | dropdown | Foundation-1 checkpoint (auto-downloaded on first run) |
| `attention` | dropdown | Attention mechanism: `auto`, `sdpa`, `flash_attention_2`, `sageattention` |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| `model` | FOUNDATION1_MODEL | Loaded model for generator node |

---

### Foundation-1 Generate

Generates a tempo-synced musical loop.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | FOUNDATION1_MODEL | — | Connect from Model Loader |
| `tags` | STRING | `Synth Lead, Warm, ...` | Instrument, timbre, FX, notation tags |
| `bpm` | dropdown | `140 BPM` | Tempo (100-150 BPM options) |
| `bars` | dropdown | `8 Bars` | Loop length (4 or 8 bars) |
| `key` | dropdown | `E minor` | Musical key (24 options) |
| `steps` | INT | 250 | Diffusion steps (10-500) |
| `cfg_scale` | FLOAT | 7.0 | Classifier-free guidance (1.0-15.0) |
| `seed` | INT | 0 | Generation seed |
| `sampler_type` | dropdown | `dpmpp-3m-sde` | Diffusion sampler |
| `sigma_min` | FLOAT | 0.3 | Minimum noise level |
| `sigma_max` | FLOAT | 500.0 | Maximum noise level |
| `unload_after_generate` | BOOLEAN | False | Offload to CPU RAM after generation |
| `torch_compile` | BOOLEAN | False | Enable torch.compile (first run slower) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| `audio` | AUDIO | Generated audio waveform |
| `prompt` | STRING | Full assembled prompt |

---

## Prompt Tags

<details>
<summary><b>Click to expand tag reference</b></summary>

Foundation-1 uses structured tags for precise control over generation. Tags should describe:
- **Instrument** — e.g., `Synth Lead`, `Piano`, `Guitar`, `Drums`
- **Timbre** — e.g., `Warm`, `Bright`, `Dark`, `Rich`, `Clean`
- **FX** — e.g., `Reverb`, `Delay`, `Distortion`, `Chorus`
- **Notation** — e.g., `Arp`, `Chord`, `Melody`, `Bassline`
- **Character** — e.g., `Spacey`, `Intimate`, `Wide`, `Thick`

**Example prompts:**
```
Synth Lead, Warm, Wide, Bright, Clean, Melody
Piano, Soft, Intimate, Reverb, Chord Progression
Drums, Punchy, Tight, Kick, Snare, Hi-Hat
Bass, Deep, Sub, Rolling, Groove
```

> **Note:** BPM, Bars, and Key are controlled via dropdowns — do not include them in the tags field.

</details>

### 📋 Full Tag Reference

For the complete list of supported tags, see the **[Master Tag Reference Sheet](./Master_Tag_Reference.md)**.

### Tag Distribution Charts

<div align="center">
  <img src="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/Charts/subfamilites_pie.PNG" alt="Sub-Family Distribution" width="80%">
  <p><em>Instrument Sub-Family Coverage</em></p>
</div>

<div align="center">
  <img src="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/Charts/timbre_tags_pie.PNG" alt="Timbre Tag Distribution" width="80%">
  <p><em>Timbre Descriptor Coverage</em></p>
</div>

<div align="center">
  <img src="https://huggingface.co/RoyalCities/Foundation-1/resolve/main/Charts/fx_pie.PNG" alt="FX Tag Distribution" width="80%">
  <p><em>FX Descriptor Coverage</em></p>
</div>

---

## Musical Keys

<details>
<summary><b>Click to expand supported keys</b></summary>

**Major Keys:**
C major, C# major, D major, Eb major, E major, F major, F# major, G major, Ab major, A major, Bb major, B major

**Minor Keys:**
C minor, C# minor, D minor, D# minor, E minor, F minor, F# minor, G minor, G# minor, A minor, Bb minor, B minor

</details>

---

## Duration Calculation

Duration is automatically calculated from BPM and bars:

```
duration (seconds) = round(bars x 4 / BPM x 60)
```

**Examples:**
| BPM | Bars | Duration |
|-----|------|----------|
| 100 | 8 | 19s |
| 120 | 4 | 8s |
| 140 | 8 | 14s |
| 150 | 4 | 6s |

**Maximum duration:** 20 seconds (model limit)

---

## File Structure

```
ComfyUI/
├── models/
│   └── stable_audio/
│       └── Foundation-1/              # Auto-downloaded
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
        ├── k_diffusion_files/         # Private k-diffusion install
        ├── pyproject.toml
        ├── requirements.txt
        └── README.md
```

---

## Parameters Explained

<details>
<summary><b>Click to expand parameter details</b></summary>

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **attention** | Attention mechanism | `auto` (SageAttention if available, else SDPA) |
| **steps** | Diffusion steps | `250` (training default), `100-150` for faster results |
| **cfg_scale** | Classifier-free guidance | `7.0` (training default), `6-8` for balance |
| **sampler_type** | Diffusion sampler | `dpmpp-3m-sde` (recommended, best quality), `k-dpm-fast` (fastest, needs fewer steps) |
| **sigma_min** | Min noise level | `0.3` (default) |
| **sigma_max** | Max noise level | `500.0` (default) |
| **unload_after_generate** | Offload to CPU RAM | `True` to free VRAM between runs |
| **torch_compile** | torch.compile optimization | `True` (first run slow, subsequent faster) |

</details>

---

## Troubleshooting

<details>
<summary><b>Click to expand troubleshooting guide</b></summary>

### Model Not Downloading?

Manually download from [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1):
```bash
pip install -U huggingface_hub
huggingface-cli download RoyalCities/Foundation-1 --local-dir ComfyUI/models/stable_audio/Foundation-1
```

### Missing Dependencies?

All dependencies are auto-installed at startup. If auto-install fails:
```bash
cd ComfyUI/custom_nodes/ComfyUI-Foundation-1
pip install -r requirements.txt
```

Common missing packages:
- `sageattention` — for optimized attention (`pip install sageattention`)

### Out of Memory?

- Enable `unload_after_generate=True` to offload to CPU RAM
- Reduce `steps` (100-150 still gives good results)
- Close other GPU applications

### Slow Generation?

- Install SageAttention: `pip install sageattention`
- Enable `torch_compile=True` (first run is slower, subsequent runs faster)
- Use `dpmpp-2m-sde` sampler (slightly faster than `dpmpp-3m-sde`)

### k_diffusion Conflicts?

Foundation-1 installs k-diffusion to a private directory (`k_diffusion_files/`) to avoid conflicts with ComfyUI's bundled version. Never install k-diffusion to site-packages manually.

</details>

---

## 🔗 Important Links

### 🤗 HuggingFace
- **Model:** [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1)

### 📄 Code
- **Inference Engine:** [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)

### 🌐 Community
- **Companion Video:** [Foundation-1 Overview](https://www.youtube.com/watch?v=Foundation1Video)

---

## 📄 License

<a name="license"></a>

This model is licensed under the **Stability AI Community License**:
- ✅ **Non-commercial use** — permitted
- ✅ **Limited commercial use** — entities with annual revenues below USD $1M
- ⚠️ **Revenue exceeding USD $1M** — refer to the repository license file for full terms

Model weights from [RoyalCities/Foundation-1](https://huggingface.co/RoyalCities/Foundation-1) are subject to the same license.

---

## ⚠️ Usage Disclaimer

Foundation-1 is intended for music production, creative applications, and legitimate purposes. Please use responsibly and ethically. We do not hold any responsibility for any illegal usage. Please refer to your local laws regarding generated content.

---

<div align="center">
    <b>Structured Text-to-Sample Diffusion for Music Production</b>
</div>
