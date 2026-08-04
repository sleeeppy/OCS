"""Pick an accelerator, and make code that hardcodes ``cuda`` land on it.

see-through was written against CUDA and says so in eight places, all of them in
``common/utils/inference_utils.py``:

    layerdiff_pipeline.vae.to(dtype=torch.bfloat16, device='cuda')
    ... trans_vae, unet, text_encoder, text_encoder_2 the same
    layerdiff_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
    marigold_pipeline.to(device='cuda', dtype=torch.bfloat16)
    marigold_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)

Everything downstream reads the device back off the modules -- ``self.unet.device``,
``vae.device``, ``self.text_encoder.device`` -- so those eight sites are the whole
story. Move the modules somewhere else and the rest of the pipeline follows.

`external/see-through` is a pinned submodule, so this patches at runtime instead
of editing it: `git submodule update` would revert a local edit, and carrying a
fork of a 12 GB-weight research repo to change one string is a bad trade.

The patch is deliberately narrow. It rewrites a *requested* CUDA device to the
target and touches nothing else -- a `.to('cpu')` stays on CPU, a `.to(dtype=...)`
with no device is untouched. Anything already device-agnostic keeps working.

Apple Silicon notes, all measured on an M5 Pro / 24 GB with torch 2.8.0:

- bf16 matmul, conv2d, group_norm and scaled_dot_product_attention all work, and
  ``torch.Generator(device='mps')`` seeds ``randn`` reproducibly. The dtype
  see-through asks for needs no downgrade.
- ``torch.mps.recommended_max_memory()`` is 17.8 GB here, not the full 24 GB.
  see-through's README puts the plain path at 12-16 GB at 1280, so it fits, but
  not with much room -- drop ``--resolution`` first if it does not.
- ``torch.cuda.empty_cache()`` and ``torch.cuda.manual_seed_all()``, which
  see-through calls unguarded, are already no-ops when CUDA is absent. Left alone.
- Group offloading is **required**, not optional. Without it a 1280 run peaked at
  a 35 GB physical footprint and 12 GB of swap on a 24 GB machine. See
  :func:`should_group_offload`.
"""

from __future__ import annotations

import os
from typing import Any

_PATCHED = False


def select_device(prefer: str | None = None) -> "object":
    """Best available accelerator as a ``torch.device``.

    ``OCS_TORCH_DEVICE`` overrides everything, which is the escape hatch for
    forcing ``cpu`` when a driver bug shows up mid-run.
    """
    import torch

    requested = prefer or os.environ.get("OCS_TORCH_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device: "object") -> dict:
    """Human-facing facts about a device, for the health endpoint and setup logs."""
    import torch

    info: dict[str, Any] = {"torch": torch.__version__, "device": str(device)}
    kind = getattr(device, "type", str(device))

    if kind == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 1
        )
    elif kind == "mps":
        info["name"] = "Apple Silicon (MPS)"
        # The MPS allocator's own ceiling, which is what a run has to fit inside
        # -- lower than the machine's RAM, so reporting total RAM would mislead.
        info["memory_gb"] = round(torch.mps.recommended_max_memory() / 2**30, 1)
    else:
        info["name"] = "CPU"
        info["memory_gb"] = None

    info["group_offload_supported"] = should_group_offload(device, requested=True)
    return info


def should_group_offload(device: "object", requested: bool) -> bool:
    """Whether to forward see-through's ``group_offload`` flag to this device.

    Group offloading parks parameter blocks in CPU memory and pages them in per
    forward pass, trading speed for headroom. see-through leans on it to fit 1280
    into 8-12 GB of VRAM.

    On MPS it is not optional, it is **required**. Measured on an M5 Pro / 24 GB
    at ``--resolution 1280``: without offloading the process reached a 35 GB peak
    physical footprint and drove 12 GB of swap, which shows up as the diffusion
    loop sitting at step 0/30 with the CPU idle -- thrashing, not computing.
    Unified memory is why: the MPS allocator's 17.8 GB budget plus SDXL weights
    plus 1280 activations does not fit in 24 GB, and offloading is what keeps
    only one block resident at a time.

    diffusers' implementation does work on MPS -- verified end to end on a real
    ``UNet2DConditionModel``, finite bf16 output -- with one condition, which
    :func:`redirect_cuda_to` enforces: ``use_stream`` must stay ``False``. The
    streamed path builds a pinned-memory dict, and ``Tensor.pin_memory()`` fails
    on this machine ("Attempted to set the storage of a tensor on device cpu to a
    storage on different device mps:0"). see-through never passes ``use_stream``,
    so the default already avoids it.

    CPU is still refused -- there is nothing to offload to.
    """
    if not requested:
        return False
    return getattr(device, "type", str(device)) in ("cuda", "mps")


#: Largest ``--resolution`` that fits on MPS, and why.
#:
#: The binding constraint is not the weights, it is one self-attention call.
#: MPS has no memory-efficient SDPA kernel: it materialises the score matrix, so
#: cost grows with the square of the token count. Measured directly, bf16, 10
#: heads, ``driver_allocated_memory`` after a single
#: ``scaled_dot_product_attention``:
#:
#:     resolution   latent tokens   allocated
#:            768           9,216      4.2 GB
#:           1024          16,384     12.6 GB
#:           1280          25,600     25.4 GB
#:
#: ``torch.mps.recommended_max_memory()`` is 17.8 GB on a 24 GB machine, so 1280
#: blows the budget on attention alone -- before any of the ~8 GB of SDXL weights.
#: It does not fail cleanly either: macOS swaps instead, and the run sits at step
#: 0/30 with the GPU busy and 12 GB of swap in use.
#:
#: Group offloading cannot help here. It pages *parameters*, and this is
#: activations -- LayerDiff 3D generates the layers as frames in one batch, so
#: they scale together.
MPS_MAX_RESOLUTION = 768


def cap_resolution(resolution: int, device: "object") -> tuple[int, str | None]:
    """Clamp ``resolution`` to what ``device`` can actually hold.

    Returns the resolution to use and, when it was lowered, a line explaining it.
    Only MPS is capped; CUDA has the memory-efficient kernels this relies on.
    """
    if getattr(device, "type", str(device)) != "mps":
        return resolution, None
    if resolution <= MPS_MAX_RESOLUTION:
        return resolution, None
    return MPS_MAX_RESOLUTION, (
        f"resolution {resolution} -> {MPS_MAX_RESOLUTION}: MPS materialises "
        f"attention scores, and {resolution} needs more than the "
        f"{MPS_MAX_RESOLUTION} cap allows on this device "
        "(see ocs.torch_device.MPS_MAX_RESOLUTION)"
    )


def _redirect(value: Any, target: Any) -> Any:
    """Map a CUDA device reference to ``target``; pass everything else through."""
    import torch

    if isinstance(value, str) and (value == "cuda" or value.startswith("cuda:")):
        return target
    if isinstance(value, torch.device) and value.type == "cuda":
        return target
    return value


def redirect_cuda_to(device: "object") -> None:
    """Send every explicit ``cuda`` placement to ``device`` instead.

    Idempotent, and a no-op when the target really is CUDA.
    """
    global _PATCHED
    import torch

    device = torch.device(device) if not isinstance(device, torch.device) else device
    if _PATCHED or device.type == "cuda":
        return

    def wrap_to(original):
        def patched(self, *args, **kwargs):
            args = tuple(_redirect(a, device) for a in args)
            if "device" in kwargs:
                kwargs["device"] = _redirect(kwargs["device"], device)
            return original(self, *args, **kwargs)

        return patched

    # nn.Module.to covers the five layerdiff sub-module placements; DiffusionPipeline
    # overrides .to itself, which is what marigold_pipeline.to() hits.
    torch.nn.Module.to = wrap_to(torch.nn.Module.to)
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(device)
    torch.Tensor.cuda = lambda self, *a, **k: self.to(device)

    from diffusers import DiffusionPipeline, ModelMixin

    DiffusionPipeline.to = wrap_to(DiffusionPipeline.to)

    def wrap_group_offload(original):
        """Point the onload device at the real accelerator, keeping streams off.

        ``enable_group_offload``'s first positional argument is the onload device,
        so see-through's ``enable_group_offload('cuda', num_blocks_per_group=1)``
        needs the same rewrite ``.to()`` gets. ``use_stream`` is pinned to False
        because the streamed path calls ``Tensor.pin_memory()``, which fails on
        MPS -- see :func:`should_group_offload`.
        """

        def patched(self, *args, **kwargs):
            # Only the onload device moves. offload_device is where parameters get
            # parked and must stay CPU -- pointing it at the accelerator too would
            # make onload == offload and quietly disable the offloading entirely,
            # which is the failure this whole path exists to avoid.
            if args:
                args = (device,) + args[1:]
            else:
                kwargs["onload_device"] = device
            kwargs["use_stream"] = False
            kwargs["record_stream"] = False
            return original(self, *args, **kwargs)

        return patched

    ModelMixin.enable_group_offload = wrap_group_offload(
        ModelMixin.enable_group_offload
    )
    DiffusionPipeline.enable_group_offload = wrap_group_offload(
        DiffusionPipeline.enable_group_offload
    )

    _PATCHED = True


def enable_mps_fallback() -> None:
    """Let unimplemented MPS ops fall back to CPU rather than raising.

    Set before torch reads it. Slow where it triggers, but the alternative is a
    hard stop partway through a 20-minute run over one missing kernel.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
