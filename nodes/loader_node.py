"""Foundation1ModelLoader — loads a Foundation-1 checkpoint from disk.

Scans ComfyUI/models/stable_audio/ for pairs of
  <name>.safetensors + model_config.json
and exposes them as a dropdown.

If no models are found on first run, the node automatically downloads
Foundation-1 from HuggingFace (RoyalCities/Foundation-1) into the
correct models directory.  Directory resolution uses 4 fallback
strategies so it works on any ComfyUI installation layout.

The model is cached in model_cache so the generate node can resume it
from CPU without a full disk reload when unload_after_generate=True.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

try:
    import folder_paths
    _HAS_FOLDER_PATHS = True
except ImportError:
    _HAS_FOLDER_PATHS = False

try:
    import comfy.model_management as mm
    _HAS_MM = True
except ImportError:
    _HAS_MM = False

from .model_cache import (
    apply_attention,
    get_cache_key,
    get_cached_model,
    set_cached_model,
    unload_model,
)

logger = logging.getLogger("Foundation1")

# ---------------------------------------------------------------------------
# HuggingFace download config
# ---------------------------------------------------------------------------

_HF_REPO_ID   = "RoyalCities/Foundation-1"
_HF_FILES     = ["Foundation_1.safetensors", "model_config.json"]
_DOWNLOAD_SUB = "stable_audio/Foundation-1"   # subfolder under models/


# ---------------------------------------------------------------------------
# Directory resolution  (4 independent fallbacks)
# ---------------------------------------------------------------------------

def _resolve_models_dir() -> Path:
    """Return the path to ComfyUI's models/ directory using 4 fallbacks.

    Fallback order
    ──────────────
    1. folder_paths.models_dir  — ComfyUI's own registry (most reliable when
                                  the node is running inside ComfyUI).
    2. __file__ navigation      — nodes/ → node/ → custom_nodes/ → ComfyUI/ → models/
                                  Works regardless of ComfyUI's Python path setup.
    3. COMFYUI_PATH env var     — user-set environment variable pointing at the
                                  ComfyUI root.  Appends /models automatically.
    4. COMFYUI_MODELS_DIR env   — user-set environment variable pointing directly
                                  at the models/ directory.

    Returns the first candidate that already exists on disk.
    If none exist yet the first candidate is returned so callers can
    create it with mkdir(parents=True).
    """
    candidates: list[Path] = []

    # 1. ComfyUI folder_paths
    if _HAS_FOLDER_PATHS:
        candidates.append(Path(folder_paths.models_dir).resolve())

    # 2. Filesystem navigation from this file
    #    __file__ = .../ComfyUI/custom_nodes/<node>/nodes/loader_node.py
    #    .parent×4 = ComfyUI root
    candidates.append(
        Path(__file__).resolve().parent.parent.parent.parent / "models"
    )

    # 3. COMFYUI_PATH environment variable
    env_root = os.environ.get("COMFYUI_PATH", "").strip()
    if env_root:
        candidates.append(Path(env_root) / "models")

    # 4. COMFYUI_MODELS_DIR environment variable
    env_models = os.environ.get("COMFYUI_MODELS_DIR", "").strip()
    if env_models:
        candidates.append(Path(env_models))

    for c in candidates:
        if c.is_dir():
            logger.debug(f"Models dir resolved via fallback: {c}")
            return c

    # None exist — return the most authoritative candidate for mkdir
    best = candidates[0] if candidates else Path("models")
    logger.debug(f"Models dir not found on disk yet — will create: {best}")
    return best


def _stable_audio_dir() -> Path:
    """Return ComfyUI/models/stable_audio/ as a Path."""
    return _resolve_models_dir() / "stable_audio"


# ---------------------------------------------------------------------------
# HuggingFace auto-download
# ---------------------------------------------------------------------------

def _download_foundation1() -> bool:
    """Download Foundation_1.safetensors + model_config.json from HuggingFace.

    Uses huggingface_hub which ships with transformers and is always
    available in a standard ComfyUI environment.

    Returns True if all files are present after the attempt.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed — cannot auto-download. "
            "Install it with: pip install huggingface_hub"
        )
        return False

    dest_dir: Path = _resolve_models_dir() / _DOWNLOAD_SUB
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Auto-downloading Foundation-1 from "
        f"https://huggingface.co/{_HF_REPO_ID}"
    )
    logger.info(f"Destination: {dest_dir}")

    all_ok = True
    for filename in _HF_FILES:
        dest_file = dest_dir / filename
        if dest_file.exists():
            logger.info(f"Already present — skipping: {filename}")
            continue

        logger.info(f"Downloading: {filename}  (this may take a while for the weights file)")
        try:
            hf_hub_download(
                repo_id=_HF_REPO_ID,
                filename=filename,
                local_dir=str(dest_dir),
            )
            logger.info(f"Downloaded: {filename}")
        except Exception as e:
            logger.error(f"Failed to download '{filename}': {e}")
            all_ok = False

    if all_ok:
        logger.info("Foundation-1 download complete.")
    else:
        logger.warning(
            "One or more files failed to download. "
            f"You can download them manually from "
            f"https://huggingface.co/{_HF_REPO_ID} "
            f"and place them in: {dest_dir}"
        )

    return all_ok


# ---------------------------------------------------------------------------
# Checkpoint scanning
# ---------------------------------------------------------------------------

def _do_scan() -> list:
    """Walk stable_audio/ and return pairs of .safetensors + model_config.json."""
    root = _stable_audio_dir()
    results = []

    if not root.is_dir():
        return results

    for dirpath, _dirs, files in os.walk(str(root)):
        safetensors = [f for f in files if f.endswith(".safetensors")]
        if not safetensors:
            continue
        has_config = "model_config.json" in files
        for sf in safetensors:
            rel = os.path.relpath(dirpath, str(root)).replace("\\", "/")
            label = sf if rel == "." else f"{rel}/{sf}"
            results.append({
                "label": label,
                "checkpoint": os.path.join(dirpath, sf),
                "config": os.path.join(dirpath, "model_config.json") if has_config else None,
            })

    return results


def _scan_checkpoints() -> list:
    """Return available checkpoints, auto-downloading if none are found."""
    results = _do_scan()

    if not results:
        logger.info(
            "No Foundation-1 models found in models/stable_audio/. "
            "Attempting auto-download from HuggingFace..."
        )
        _download_foundation1()
        results = _do_scan()   # rescan after download attempt

    return results


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(choice: str) -> str:
    """Resolve 'auto' and validate explicit device choices."""
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if choice == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        return "cpu"

    if choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available — falling back to CPU.")
            return "cpu"

    return choice


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Foundation1ModelLoader:
    """Loads a Foundation-1 checkpoint and its model_config.json.

    On first run, if no model files are found in
    ComfyUI/models/stable_audio/ the node automatically downloads
    Foundation-1 from HuggingFace (RoyalCities/Foundation-1).

    The loaded model is cached; it reloads from disk only when the
    checkpoint, device, or attention type changes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        pairs = _scan_checkpoints()
        labels = (
            [p["label"] for p in pairs]
            if pairs
            else ["Download failed — check console and place files manually"]
        )

        return {
            "required": {
                "model": (labels, {
                    "tooltip": (
                        "Foundation-1 .safetensors checkpoint. "
                        "If no model is found it will be downloaded automatically "
                        "from huggingface.co/RoyalCities/Foundation-1 on first run."
                    ),
                }),
                "attention": (["auto", "sdpa", "flash_attention_2", "sageattention"], {
                    "default": "auto",
                    "tooltip": (
                        "'auto' uses SageAttention if installed, otherwise SDPA. "
                        "'sdpa' enables all PyTorch SDPA backends (uses Flash internally if available). "
                        "'flash_attention_2' forces the Flash SDP backend only (CUDA only). "
                        "'sageattention' monkey-patches F.sdpa with SageAttention "
                        "(CUDA only — requires: pip install sageattention). "
                        "Changing this setting unloads and reloads the model."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("FOUNDATION1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/Foundation-1"
    DESCRIPTION = (
        "Loads a Foundation-1 model from a local .safetensors checkpoint. "
        "Auto-downloads from HuggingFace on first run if no model is present. "
        "The model is cached and only reloads when the checkpoint, device, "
        "or attention type changes."
    )

    def load_model(
        self,
        model: str,
        attention: str,
    ) -> Tuple[Dict[str, Any]]:

        # ── CUDA is required (Flash Attention hardcoded in model architecture) ──
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Foundation-1 requires CUDA. "
                "The model architecture uses Flash Attention which is CUDA-only."
            )
        resolved_device = "cuda"
        logger.info(f"Device: cuda")

        # ── Locate checkpoint + config ─────────────────────────────────────
        pairs = _scan_checkpoints()
        pair = next((p for p in pairs if p["label"] == model), None)

        if pair is None:
            raise ValueError(
                f"Checkpoint '{model}' not found. "
                "Check the console — auto-download may have failed. "
                f"Download manually from https://huggingface.co/{_HF_REPO_ID} "
                f"and place files in: {_resolve_models_dir() / _DOWNLOAD_SUB}"
            )

        checkpoint_path: str = pair["checkpoint"]
        config_path: Optional[str] = pair["config"]

        if not config_path or not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"model_config.json not found alongside {checkpoint_path}. "
                "Both files must be in the same directory."
            )

        # ── Check cache ────────────────────────────────────────────────────
        cache_key = get_cache_key(checkpoint_path, resolved_device, attention)
        cached_data, cached_key = get_cached_model()

        if cached_data is not None and cached_key == cache_key:
            logger.info(f"Using cached Foundation-1 model ({model}).")
            return (cached_data,)

        # ── Cache miss — evict old model first ─────────────────────────────
        if cached_data is not None:
            logger.info("Cache key changed — unloading previous model.")
            unload_model()

        # ── Load config ────────────────────────────────────────────────────
        with open(config_path, "r", encoding="utf-8") as f:
            model_config: Dict[str, Any] = json.load(f)

        sample_rate: int = model_config.get("sample_rate", 44100)
        sample_size: int = model_config.get("sample_size", 882000)

        logger.info(
            f"Config: {sample_rate} Hz, "
            f"max {sample_size / sample_rate:.1f}s"
        )

        # ── Apply attention ────────────────────────────────────────────────
        applied_attention = apply_attention(attention, resolved_device)
        if applied_attention != attention and attention != "auto":
            logger.warning(
                f"Attention '{attention}' unavailable on '{resolved_device}' "
                f"— using '{applied_attention}'."
            )

        # ── Build model architecture ───────────────────────────────────────
        logger.info("Building model architecture from config...")
        from stable_audio_tools.models.factory import create_model_from_config
        audio_model = create_model_from_config(model_config)

        # ── Load weights ───────────────────────────────────────────────────
        logger.info(f"Loading weights: {os.path.basename(checkpoint_path)}")
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device="cpu")

        # Strip EMA prefix if present (use_ema=true in training config)
        ema_keys = [k for k in state_dict if k.startswith("ema_model.")]
        if ema_keys:
            logger.info(f"EMA weights detected ({len(ema_keys)} keys) — stripping prefix.")
            state_dict = {
                k[len("ema_model."):]: v
                for k, v in state_dict.items()
                if k.startswith("ema_model.")
            }

        missing, unexpected = audio_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {len(missing)} (first 3: {missing[:3]})")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

        # ── dtype: upcast fp16 → fp32 for CPU (fp16 ops unreliable on CPU) ─
        if resolved_device == "cpu":
            logger.info("CPU device — upcasting fp16 weights to float32.")
            audio_model = audio_model.float()

        # ── Move to device and eval ────────────────────────────────────────
        audio_model.eval().to(resolved_device)

        logger.info(
            f"Foundation-1 ready — device={resolved_device}, "
            f"attention={applied_attention}, {sample_rate} Hz"
        )

        model_data: Dict[str, Any] = {
            "model": audio_model,
            "config": model_config,
            "device": resolved_device,
            "sample_rate": sample_rate,
            "sample_size": sample_size,
            "checkpoint_path": checkpoint_path,
            "attention": applied_attention,
        }

        set_cached_model(model_data, cache_key, keep_in_vram=True)
        return (model_data,)

    @classmethod
    def IS_CHANGED(cls, model: str, attention: str):
        """Re-execute only when checkpoint or attention changes."""
        return hash((model, attention))
