"""Foundation1Generate — generate a musical loop with Foundation-1.

Accepts a FOUNDATION1_MODEL from Foundation1ModelLoader and a set of
musical parameters, assembles the final prompt, calculates the correct
loop duration, and runs the diffusion sampler.

Progress is reported step-by-step via ComfyUI's native ProgressBar using
the k-diffusion callback that stable-audio-tools passes through to each
sampler (dpmpp-3m-sde, dpmpp-2m-sde, k-heun, …).

After generation the model can be offloaded to CPU
(unload_after_generate=True) to free VRAM while keeping weights in RAM
for a faster next run. ComfyUI's native 'Free Memory' button will also
clear the CPU-resident copy when this is enabled.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from comfy.utils import ProgressBar
    _HAS_PBAR = True
except ImportError:
    _HAS_PBAR = False

try:
    import comfy.model_management as mm
    _HAS_MM = True
except ImportError:
    _HAS_MM = False

from .model_cache import is_offloaded, offload_to_cpu, resume_to_device

logger = logging.getLogger("Foundation1")

# ---------------------------------------------------------------------------
# k_diffusion shim paths
# ---------------------------------------------------------------------------
# k-diffusion 0.1.1 is installed by __init__.py into this private directory
# using pip --target (NOT site-packages, NOT on sys.path).  We load only the
# two files we need via importlib — the package __init__.py (which has the
# evaluation→clip→pkg_resources issue) is never imported.
# ---------------------------------------------------------------------------

# nodes/generate_node.py → nodes/ → ComfyUI-Foundation-1/
_NODE_ROOT = Path(__file__).parent.parent.resolve()
_KDIFF_TARGET = _NODE_ROOT / "k_diffusion_files"

_real_k_sampling: Optional[object] = None   # cached after first load
_external_injected: bool = False             # injected once per session


def _load_file_as_module(name: str, path: Path) -> Optional[object]:
    """Load a single .py file as a module using importlib (no __init__ triggered)."""
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Set __package__ so relative imports (e.g. `from . import utils`) inside
    # the file resolve against k_diffusion already in sys.modules (ComfyUI's
    # version), which is API-compatible.
    mod.__package__ = "k_diffusion"
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        logger.warning(f"Could not load {path.name}: {e}")
        return None


def _inject_k_external() -> None:
    """Inject VDenoiser into k_diffusion.external (once per session).

    Loads external.py from our private k-diffusion install and attaches it
    as k_diffusion.external on whichever k_diffusion is in sys.modules.
    ComfyUI's k_diffusion never uses .external so nothing breaks.
    """
    global _external_injected
    if _external_injected:
        return

    import k_diffusion as _kd   # safe: __init__.py registered ComfyUI's version
    if hasattr(_kd, "external"):
        _external_injected = True
        return

    ext_path = _KDIFF_TARGET / "k_diffusion" / "external.py"
    mod = _load_file_as_module("k_diffusion.external", ext_path)
    if mod is None:
        logger.error(
            "k_diffusion/external.py not found — "
            f"expected at {ext_path}. Restart ComfyUI to trigger re-install."
        )
        return

    sys.modules["k_diffusion.external"] = mod
    _kd.external = mod
    _external_injected = True
    logger.info("k_diffusion.external injected (VDenoiser available).")


def _load_real_k_sampling() -> Optional[object]:
    """Load k_diffusion/sampling.py from our private install (cached).

    This is the unmodified k-diffusion sampling, needed because ComfyUI's
    version expects its own ModelPatcher API which Foundation-1 doesn't use.
    """
    global _real_k_sampling
    if _real_k_sampling is not None:
        return _real_k_sampling

    sampling_path = _KDIFF_TARGET / "k_diffusion" / "sampling.py"
    mod = _load_file_as_module("k_diffusion._real_sampling", sampling_path)
    if mod is None:
        logger.error(
            "k_diffusion/sampling.py not found — "
            f"expected at {sampling_path}. Restart ComfyUI to trigger re-install."
        )
        return None

    _real_k_sampling = mod
    logger.debug("Real k_diffusion.sampling loaded from private install.")
    return mod

# ---------------------------------------------------------------------------
# Prompt building constants
# ---------------------------------------------------------------------------

# All 24 western keys — enharmonic equivalents included where common
KEYS = [
    # Major
    "C major", "C# major", "D major", "Eb major", "E major", "F major",
    "F# major", "G major", "Ab major", "A major", "Bb major", "B major",
    # Minor
    "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "Bb minor", "B minor",
]

BPM_OPTIONS = [
    "100 BPM", "110 BPM", "120 BPM", "128 BPM",
    "130 BPM", "140 BPM", "150 BPM",
]

BARS_OPTIONS = ["4 Bars", "8 Bars"]

SAMPLER_OPTIONS = [
    "dpmpp-3m-sde",   # Recommended — best quality / speed balance
    "dpmpp-2m-sde",
    "dpmpp-2m",
    "k-heun",
    "k-dpmpp-2s-ancestral",
    "k-dpm-2",
    "k-dpm-fast",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_duration(bars: int, bpm: int) -> int:
    """Return loop duration in whole seconds.

    Formula: bars × 4 beats / BPM × 60 s/min
    Matches the seconds_total values used in the model's training demos:
      8 bars @ 100 BPM → 19 s   (19.2 rounded)
      8 bars @ 140 BPM → 14 s   (13.71 rounded)
      4 bars @ 150 BPM →  6 s   ( 6.4  rounded)
    """
    return round(bars * 4 / bpm * 60)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Foundation1Generate:
    """Generates a tempo-synced musical loop with Foundation-1.

    Connect the 'model' output of Foundation1ModelLoader to this node.
    Fill in your instrument / timbre / FX / notation tags in the 'tags'
    field and select BPM, bars, and key from the dropdowns — the node
    assembles the final prompt and calculates the correct duration for you.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FOUNDATION1_MODEL", {
                    "tooltip": "Connect to Foundation1ModelLoader.",
                }),
                "tags": ("STRING", {
                    "multiline": True,
                    "default": "Synth Lead, Warm, Wide, Bright, Clean, Melody",
                    "tooltip": (
                        "Instrument, timbre, FX, and notation tags separated by commas. "
                        "See the Tag Reference Sheet. "
                        "Do NOT put BPM, Bars, or Key here — use the dropdowns below."
                    ),
                }),
                "bpm": (BPM_OPTIONS, {
                    "default": "140 BPM",
                    "tooltip": "Tempo of the generated loop.",
                }),
                "bars": (BARS_OPTIONS, {
                    "default": "8 Bars",
                    "tooltip": "Loop length in 4/4 bars.",
                }),
                "key": (KEYS, {
                    "default": "E minor",
                    "tooltip": "Musical key for the generated loop.",
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "tooltip": (
                        "Diffusion steps. "
                        "Higher = better quality but slower. "
                        "100–250 is the practical range. "
                        "The training demo used 250."
                    ),
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": (
                        "Classifier-free guidance scale. "
                        "Higher = stronger prompt adherence, less variation. "
                        "6–8 recommended. Training demo used 7."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Generation seed. Use ComfyUI's seed controls to randomise.",
                }),
                "sampler_type": (SAMPLER_OPTIONS, {
                    "default": "dpmpp-3m-sde",
                    "tooltip": (
                        "Diffusion sampler. "
                        "'dpmpp-3m-sde' recommended for quality. "
                        "'dpmpp-2m-sde' is slightly faster."
                    ),
                }),
                "sigma_min": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.001,
                    "tooltip": "Minimum noise level for k-diffusion schedule. Default 0.3.",
                }),
                "sigma_max": ("FLOAT", {
                    "default": 500.0,
                    "min": 10.0,
                    "max": 1000.0,
                    "step": 10.0,
                    "tooltip": "Maximum noise level for k-diffusion schedule. Default 500.",
                }),
                "unload_after_generate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "After generation, move the model from VRAM to CPU RAM. "
                        "Frees VRAM while keeping weights in memory for a faster "
                        "next run (avoids a full disk reload). "
                        "When enabled, ComfyUI's native 'Free Memory' button will "
                        "also clear the CPU copy."
                    ),
                }),
                "torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Compile the model with torch.compile before the first generation. "
                        "The first run after enabling will be slower (compilation warmup). "
                        "Every subsequent run in the same session will be faster. "
                        "Compiled state is cached — toggling off requires a model reload. "
                        "Requires PyTorch 2.0+. CUDA only. Disable if you see errors."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio/Foundation-1"
    DESCRIPTION = (
        "Generates a tempo-synced musical sample loop using Foundation-1. "
        "Connect Foundation1ModelLoader → model input."
    )

    def generate(
        self,
        model: Dict[str, Any],
        tags: str,
        bpm: str,
        bars: str,
        key: str,
        steps: int,
        cfg_scale: float,
        seed: int,
        sampler_type: str,
        sigma_min: float,
        sigma_max: float,
        unload_after_generate: bool,
        torch_compile: bool,
    ) -> Tuple[Dict[str, Any]]:

        self._check_interrupt()

        # ── Validate ───────────────────────────────────────────────────────
        if not tags.strip():
            raise ValueError("'tags' cannot be empty.")

        # ── Parse dropdowns ────────────────────────────────────────────────
        bpm_int  = int(bpm.split()[0])
        bars_int = int(bars.split()[0])

        # ── Duration / sample_size ─────────────────────────────────────────
        duration_s  = _calc_duration(bars_int, bpm_int)
        audio_model = model["model"]
        device      = model["device"]
        sample_rate = model["sample_rate"]
        sample_size = duration_s * sample_rate

        logger.info(
            f"Duration: {duration_s}s  "
            f"({bars_int} bars @ {bpm_int} BPM)  "
            f"sample_size={sample_size}"
        )

        # ── Build prompt ───────────────────────────────────────────────────
        tags_clean  = tags.strip().rstrip(",").rstrip()
        full_prompt = f"{tags_clean}, {bars}, {bpm}, {key}"
        logger.info(f"Prompt: {full_prompt}")
        logger.info(f"Seed: {seed}")

        # ── Resume from CPU if previously offloaded ────────────────────────
        if is_offloaded():
            logger.info(f"Model is on CPU — resuming to {device}...")
            resume_to_device(device)

        # ── torch.compile ─────────────────────────────────────────────────
        # Compiled state is stored in model_data so it survives across calls
        # without recompiling. Toggling off requires reloading the model
        # (cache key change) since we can't un-compile a wrapped model.
        if torch_compile and device == "cuda" and not model.get("_compiled", False):
            try:
                logger.info(
                    "Compiling model with torch.compile (one-time warmup — "
                    "this run will be slower, subsequent runs faster)..."
                )
                model["model"] = torch.compile(
                    audio_model,
                    mode="reduce-overhead",   # best for repeated inference
                    fullgraph=False,          # allow graph breaks for safety
                )
                audio_model = model["model"]
                model["_compiled"] = True
                logger.info("torch.compile done.")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed ({e}) — running without compilation."
                )
        elif torch_compile and model.get("_compiled", False):
            audio_model = model["model"]   # use already-compiled model
            logger.debug("Using cached compiled model.")
        elif torch_compile and device != "cuda":
            logger.warning("torch.compile is only supported on CUDA — skipping.")

        # ── TF32 — free speed-up on Ampere+ (RTX 30xx / 40xx) ─────────────
        # Allows matmul and conv ops to use TF32 precision in CUDA kernels —
        # same VRAM footprint, measurably faster on Ampere+ GPUs.
        # PyTorch 2.9 renamed the API; try new first, fall back to old.
        if device == "cuda":
            try:
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision  = "tf32"
            except AttributeError:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32        = True

        # ── Progress bar ───────────────────────────────────────────────────
        # steps   = diffusion steps (one callback per k-diffusion step)
        # +1 slot = latent decode pass that follows sampling
        total_pbar_steps = steps + 1
        pbar = ProgressBar(total_pbar_steps) if _HAS_PBAR else None

        def _step_callback(data: dict) -> None:
            """Called by the k-diffusion sampler at every denoising step."""
            if pbar is None:
                return
            # k-diffusion passes 'i' as 0-based step index
            step = data.get("i", 0)
            pbar.update_absolute(step + 1, total_pbar_steps)

        # ── Generate ───────────────────────────────────────────────────────
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        conditioning = [{
            "prompt": full_prompt,
            "seconds_start": 0,
            "seconds_total": duration_s,
        }]

        logger.info("Generating...")
        audio_tensor: torch.Tensor

        # ── k_diffusion shim ───────────────────────────────────────────────
        # 1. Inject external.py (VDenoiser) into k_diffusion.external
        # 2. Swap ComfyUI's patched sampling.py with the real k-diffusion one
        _inject_k_external()

        import k_diffusion as _K
        _comfy_sampling  = _K.sampling
        _real_sampling   = _load_real_k_sampling()
        if _real_sampling is not None:
            _K.sampling = _real_sampling
        else:
            logger.warning(
                "Real k_diffusion.sampling unavailable — generation may fail."
            )

        try:
            self._check_interrupt()

            # inference_mode is stronger than no_grad — it disables autograd
            # history entirely, reducing memory overhead and running faster.
            with torch.inference_mode():
                audio_tensor = generate_diffusion_cond(
                    audio_model,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    conditioning=conditioning,
                    sample_size=sample_size,
                    seed=seed,
                    device=device,
                    # passed through **sampler_kwargs into each k-diff sampler
                    sampler_type=sampler_type,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    callback=_step_callback,
                )

            # Decode step — mark progress bar complete
            if pbar is not None:
                pbar.update_absolute(total_pbar_steps, total_pbar_steps)

        finally:
            # Always restore ComfyUI's k_diffusion.sampling regardless of
            # success, error, or user cancellation.
            _K.sampling = _comfy_sampling
            if unload_after_generate:
                offload_to_cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Format output ──────────────────────────────────────────────────
        # ComfyUI AUDIO: {"waveform": Tensor[batch, channels, samples], "sample_rate": int}
        if audio_tensor.dim() == 2:      # [channels, samples]
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 1:    # [samples]
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

        audio_tensor = audio_tensor.contiguous().cpu().float()

        logger.info(
            f"Done — shape={list(audio_tensor.shape)}, "
            f"{sample_rate} Hz"
        )

        return ({"waveform": audio_tensor, "sample_rate": sample_rate},)

    def _check_interrupt(self) -> None:
        if _HAS_MM:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                raise

    @classmethod
    def IS_CHANGED(
        cls,
        model,
        tags,
        bpm,
        bars,
        key,
        steps,
        cfg_scale,
        seed,
        sampler_type,
        sigma_min,
        sigma_max,
        unload_after_generate,
        torch_compile,
    ):
        # Hash all generation parameters so ComfyUI caches the result when
        # nothing changes. When the user enables ComfyUI's 'randomize seed',
        # the seed value changes → hash changes → node re-executes.
        return hash((tags, bpm, bars, key, steps, cfg_scale, seed,
                     sampler_type, sigma_min, sigma_max, torch_compile))
