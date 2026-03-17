"""ComfyUI custom nodes for Foundation-1.

Foundation-1 is a structured text-to-sample diffusion model for music
production. It understands instrument identity, timbre, FX, musical
notation, BPM, bar count, and key as separate composable controls.

Nodes registered:
  - Foundation1ModelLoader  — loads a local .safetensors checkpoint
  - Foundation1Generate     — generates a tempo-synced musical loop

Model files go in:  ComfyUI/models/stable_audio/<any-subfolder>/
Required files:     Foundation_1.safetensors  +  model_config.json
"""

__version__ = "0.1.0"

import warnings

# Silence upstream deprecation warnings from PyTorch and torchsde
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*")
warnings.filterwarnings("ignore", message="Should have tb<=t1 but got tb=")

import importlib
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Directory containing THIS file (ComfyUI-Foundation-1/)
_NODE_DIR = Path(__file__).parent.resolve()

# Private install target for k-diffusion files.
# Installed with --target so it lands here, NOT in site-packages.
# This directory is never added to sys.path — files are loaded with importlib.
_KDIFF_TARGET = _NODE_DIR / "k_diffusion_files"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("Foundation1")
logger.propagate = False

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[Foundation1] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# pip helper
# ---------------------------------------------------------------------------

def _pip_install(spec: str) -> bool:
    """Install a package using the currently running Python interpreter.

    --prefer-binary is added to every install so pip always downloads a
    pre-built wheel instead of building from source.  This avoids the
    pkg_resources / setuptools breakage that occurs when building older
    packages (e.g. pandas 2.0.x) from source in embedded Python 3.13+.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--prefer-binary"] + spec.split()
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"Installed: {spec}")
            importlib.invalidate_caches()
            return True
        logger.error(f"pip failed for '{spec}':\n{result.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"pip timed out for: {spec}")
        return False
    except Exception as e:
        logger.error(f"pip error for '{spec}': {e}")
        return False


# ---------------------------------------------------------------------------
# torch safety net
# ---------------------------------------------------------------------------

def _restore_torch() -> None:
    """Re-install a CUDA torch build if a dependency downgraded it."""
    try:
        import torch
        if "+cu" in torch.__version__:
            return
        logger.warning(
            f"torch {torch.__version__} is NOT a CUDA build. Restoring..."
        )
    except ImportError:
        return
    _pip_install("torch torchaudio --index-url https://download.pytorch.org/whl/cu128")


# ---------------------------------------------------------------------------
# k_diffusion sys.modules registration
# ---------------------------------------------------------------------------

def _register_comfy_k_diffusion() -> None:
    """Register ComfyUI's k_diffusion as the top-level 'k_diffusion' module.

    ComfyUI ships its own k_diffusion at comfy/k_diffusion/.  We register
    it in sys.modules under the bare name 'k_diffusion' so that any later
    `import k_diffusion` (e.g. from stable-audio-tools' sampling.py) gets
    ComfyUI's version from the cache instead of searching sys.path and
    potentially finding a site-packages installation with incompatible
    dependencies (clip, evaluation, pkg_resources, etc.).

    This must run before stable-audio-tools is ever imported.
    """
    if "k_diffusion" in sys.modules:
        return  # already registered

    try:
        import comfy.k_diffusion as _comfy_kd
        sys.modules["k_diffusion"] = _comfy_kd
        logger.debug("Registered comfy.k_diffusion as top-level k_diffusion.")
    except ImportError:
        logger.warning(
            "Could not import comfy.k_diffusion — "
            "k_diffusion module registration skipped."
        )


# ---------------------------------------------------------------------------
# k-diffusion private install (--target, not site-packages)
# ---------------------------------------------------------------------------

def _ensure_kdiff_files() -> bool:
    """Install k-diffusion 0.1.1 to our private node directory.

    Uses pip --target so the files land in _KDIFF_TARGET (inside this
    custom node's folder) and are NEVER added to sys.path or site-packages.
    We only ever load specific files (external.py, sampling.py) from there
    using importlib — we never `import k_diffusion` from this target.

    This avoids the k_diffusion/__init__.py → evaluation.py → clip →
    pkg_resources import chain that breaks on embedded Python 3.13+.
    """
    ext_path = _KDIFF_TARGET / "k_diffusion" / "external.py"
    sampling_path = _KDIFF_TARGET / "k_diffusion" / "sampling.py"

    if ext_path.is_file() and sampling_path.is_file():
        logger.debug("k-diffusion shim files already present.")
        return True

    logger.info(
        f"Installing k-diffusion 0.1.1 to private directory {_KDIFF_TARGET} ..."
    )
    _KDIFF_TARGET.mkdir(parents=True, exist_ok=True)

    ok = _pip_install(
        f"k-diffusion==0.1.1 --no-deps --target {_KDIFF_TARGET}"
    )
    if not ok:
        logger.error(
            "Failed to install k-diffusion shim files. "
            "Audio generation will fail."
        )
        return False

    importlib.invalidate_caches()

    if not (ext_path.is_file() and sampling_path.is_file()):
        logger.error(
            f"k-diffusion installed but expected files are missing in {_KDIFF_TARGET}."
        )
        return False

    logger.info("k-diffusion shim files ready.")
    return True


# ---------------------------------------------------------------------------
# Dependency auto-install
# ---------------------------------------------------------------------------

# (import_name, pip_spec)
#
# torch / torchaudio / numpy / safetensors / transformers / huggingface_hub
# are already present in ComfyUI portable — not listed here.
#
# k_diffusion is handled separately via _ensure_kdiff_files() — see above.
#
# stable-audio-tools is installed with --no-deps to avoid pandas==2.0.2
# (no Python 3.13 wheel; building from source hits missing pkg_resources).
_REQUIRED = [
    ("einops",           "einops>=0.7.0"),
    ("stable_audio_tools", "stable-audio-tools --no-deps"),
    ("alias_free_torch", "alias-free-torch"),
    ("ema_pytorch",      "ema-pytorch"),
    ("einops_exts",      "einops-exts"),
]


def _ensure_dependencies() -> bool:
    """Auto-install missing packages. Returns True when all are importable."""

    # ── Step 1: register ComfyUI's k_diffusion before anything else runs ──
    # This must happen before any package that does `import k_diffusion`.
    _register_comfy_k_diffusion()

    # ── Step 2: standard package installs ─────────────────────────────────
    all_ok = True
    any_installed = False
    failed: list = []

    for import_name, pip_spec in _REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            logger.warning(f"'{import_name}' not found — installing: {pip_spec}")
            if _pip_install(pip_spec):
                any_installed = True
                try:
                    __import__(import_name)
                except ImportError:
                    logger.error(
                        f"Installed '{pip_spec}' but '{import_name}' still "
                        "cannot be imported. Please restart ComfyUI."
                    )
                    failed.append(pip_spec)
                    all_ok = False
            else:
                failed.append(pip_spec)
                all_ok = False

    # ── Step 4: install k-diffusion to private target dir ─────────────────
    if not _ensure_kdiff_files():
        failed.append("k-diffusion==0.1.1 (private target)")
        all_ok = False

    if any_installed:
        _restore_torch()

    if not all_ok:
        manual = "\n".join(
            f"  {sys.executable} -m pip install {s}" for s in failed
        )
        logger.error(
            "Auto-install failed for some packages. "
            "Install manually then restart ComfyUI:\n" + manual
        )

    return all_ok


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

if _ensure_dependencies():
    try:
        from .nodes.loader_node import Foundation1ModelLoader
        from .nodes.generate_node import Foundation1Generate

        NODE_CLASS_MAPPINGS = {
            "Foundation1ModelLoader": Foundation1ModelLoader,
            "Foundation1Generate":    Foundation1Generate,
        }
        NODE_DISPLAY_NAME_MAPPINGS = {
            "Foundation1ModelLoader": "Foundation-1 Model Loader",
            "Foundation1Generate":    "Foundation-1 Generate",
        }

        logger.info(
            f"Registered {len(NODE_CLASS_MAPPINGS)} nodes (v{__version__}): "
            + ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values())
        )

    except Exception as e:
        logger.error(f"Failed to register nodes: {e}", exc_info=True)
else:
    logger.warning(
        "Foundation-1 nodes not registered — "
        "fix dependency errors above and restart ComfyUI."
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
