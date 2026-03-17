"""Foundation-1 model cache, attention management, and ComfyUI memory integration.

This module is the single source of truth for the loaded model state.
It hooks into comfy.model_management.soft_empty_cache at import time so
ComfyUI's native 'Free Memory' button also clears our CPU-offloaded model.
"""

import gc
import logging
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("Foundation1")

# ---------------------------------------------------------------------------
# Cache state
# ---------------------------------------------------------------------------

_cached_model_data: Optional[Dict[str, Any]] = None
_cached_key: Tuple = ()

# When True the soft_empty_cache hook will NOT evict the model.
# Set to False when unload_after_generate=True so ComfyUI's 'Free Memory'
# also clears the CPU-resident copy.
_keep_in_vram: bool = True

# True once the model has been moved to CPU via offload_to_cpu().
_offloaded_to_cpu: bool = False

# Track whether SageAttention is currently monkey-patched onto
# torch.nn.functional.scaled_dot_product_attention.
_sage_patched: bool = False


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def get_cache_key(checkpoint_path: str, device: str, attention: str) -> Tuple:
    return (checkpoint_path, device, attention)


# ---------------------------------------------------------------------------
# Cache accessors
# ---------------------------------------------------------------------------

def get_cached_model() -> Tuple[Optional[Dict[str, Any]], Tuple]:
    return _cached_model_data, _cached_key


def set_cached_model(
    model_data: Dict[str, Any],
    key: Tuple,
    keep_in_vram: bool = True,
) -> None:
    global _cached_model_data, _cached_key, _keep_in_vram, _offloaded_to_cpu
    _cached_model_data = model_data
    _cached_key = key
    _keep_in_vram = keep_in_vram
    _offloaded_to_cpu = False


def is_offloaded() -> bool:
    return _offloaded_to_cpu


# ---------------------------------------------------------------------------
# Attention management
# ---------------------------------------------------------------------------

def apply_attention(attention_type: str, device: str) -> str:
    """Configure the requested attention backend.

    Returns the attention type that was actually applied, which may differ
    from the requested type if a fallback was needed.

    Restores the original F.scaled_dot_product_attention before any new
    attention setting is applied, so switching types always starts clean.
    """
    global _sage_patched

    # Always restore the original SDPA first to avoid double-patching
    # or leaving a stale SageAttention patch when switching to sdpa/flash.
    if _sage_patched:
        import torch.nn.functional as F
        if hasattr(F, "_f1_original_sdpa"):
            F.scaled_dot_product_attention = F._f1_original_sdpa
            del F._f1_original_sdpa
        _sage_patched = False
        logger.debug("Restored original F.scaled_dot_product_attention.")

    # Non-CUDA devices only support SDPA — warn and fall back.
    if device != "cuda":
        if attention_type in ("sageattention", "flash_attention_2"):
            logger.warning(
                f"'{attention_type}' requires CUDA but device is '{device}'. "
                "Using sdpa."
            )
        # Reset PyTorch SDPA backends to safe defaults on non-CUDA devices.
        # (These calls are no-ops on CPU/MPS but harmless.)
        _set_sdpa_all_backends(True)
        return "sdpa"

    # ── CUDA path ──────────────────────────────────────────────────────────

    # Auto: probe sage → then fall through to sdpa (which uses flash if available)
    resolved = attention_type
    if attention_type == "auto":
        try:
            import sageattention  # noqa: F401
            resolved = "sageattention"
        except ImportError:
            resolved = "sdpa"  # PyTorch SDPA uses Flash as a backend automatically

    if resolved == "sageattention":
        try:
            from sageattention import sageattn
            import torch.nn.functional as F
            F._f1_original_sdpa = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = sageattn
            _sage_patched = True
            logger.info("Attention: SageAttention active (monkey-patched F.sdpa).")
        except ImportError:
            logger.warning(
                "SageAttention not installed — install with: pip install sageattention. "
                "Falling back to sdpa."
            )
            resolved = "sdpa"

    if resolved == "flash_attention_2":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            logger.info("Attention: Flash Attention (SDPA flash-only backend) active.")
        except Exception as e:
            logger.warning(f"Failed to enable Flash Attention: {e}. Using sdpa.")
            _set_sdpa_all_backends(True)
            resolved = "sdpa"

    if resolved == "sdpa":
        _set_sdpa_all_backends(True)
        logger.info("Attention: SDPA (all backends enabled — uses Flash if available).")

    return resolved


def _set_sdpa_all_backends(enabled: bool) -> None:
    """Enable or disable all PyTorch SDPA backends uniformly."""
    try:
        torch.backends.cuda.enable_flash_sdp(enabled)
        torch.backends.cuda.enable_mem_efficient_sdp(enabled)
        torch.backends.cuda.enable_math_sdp(enabled)
    except Exception:
        pass  # Not on CUDA or older PyTorch — safe to ignore


# ---------------------------------------------------------------------------
# CPU offload / resume
# ---------------------------------------------------------------------------

def offload_to_cpu() -> None:
    """Move the model from VRAM to system RAM.

    The model stays in memory so the next resume() is much faster than
    a cold reload from disk. VRAM is freed immediately after the move.
    """
    global _offloaded_to_cpu

    if _cached_model_data is None:
        return
    if _offloaded_to_cpu:
        logger.debug("Model already on CPU — skipping offload.")
        return

    try:
        _cached_model_data["model"].to("cpu")
        _offloaded_to_cpu = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Foundation-1 offloaded to CPU. VRAM freed.")
    except Exception as e:
        logger.warning(f"CPU offload failed: {e}")


def resume_to_device(device: str) -> None:
    """Move the model from CPU back to the target device before generation."""
    global _offloaded_to_cpu

    if _cached_model_data is None or not _offloaded_to_cpu:
        return

    try:
        _cached_model_data["model"].to(device)
        _cached_model_data["device"] = device
        _offloaded_to_cpu = False
        logger.info(f"Foundation-1 resumed to {device}.")
    except Exception as e:
        logger.warning(f"Resume to {device} failed: {e}")


# ---------------------------------------------------------------------------
# Full unload
# ---------------------------------------------------------------------------

def unload_model() -> None:
    """Remove the model from memory entirely (both VRAM and RAM)."""
    global _cached_model_data, _cached_key, _keep_in_vram, _offloaded_to_cpu

    if _cached_model_data is None:
        return

    logger.info("Unloading Foundation-1 from memory...")
    try:
        del _cached_model_data["model"]
    except Exception:
        pass
    del _cached_model_data

    _cached_model_data = None
    _cached_key = ()
    _keep_in_vram = True
    _offloaded_to_cpu = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Foundation-1 unloaded.")


# ---------------------------------------------------------------------------
# ComfyUI memory management hook
# ---------------------------------------------------------------------------

def _hook_comfy_model_management() -> None:
    """Patch comfy.model_management.soft_empty_cache so ComfyUI's native
    'Free Memory' / 'Unload Models' button also clears our cache.

    Behaviour:
    - If _keep_in_vram is True  (unload_after_generate=False): leave it alone.
      The user explicitly chose to keep the model loaded.
    - If _keep_in_vram is False (unload_after_generate=True): model is on CPU,
      fully unload it so system RAM is also freed.
    """
    try:
        import comfy.model_management as mm
        _original = mm.soft_empty_cache

        def _patched(*args, **kwargs):
            if not _keep_in_vram:
                unload_model()
            return _original(*args, **kwargs)

        mm.soft_empty_cache = _patched
        logger.debug("Hooked comfy.model_management.soft_empty_cache.")
    except Exception:
        pass  # Not running inside ComfyUI — no-op.


# Install the hook the moment this module is first imported.
_hook_comfy_model_management()
