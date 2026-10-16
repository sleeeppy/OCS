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
- ``torch.cuda.empty_cache()`` is repointed at ``torch.mps.empty_cache()``. It is
  a harmless no-op without CUDA, but harmless is not the same as working -- and
  see-through calls it to hand back memory it has just freed. See
  :func:`redirect_cuda_to`.
- ``torch.cuda.manual_seed_all()`` is genuinely fine to leave: it registers a lazy
  call that never fires, and ``torch.manual_seed`` already seeds MPS.
- Group offloading is **required**, not optional. Without it a 1280 run peaked at
  a 35 GB physical footprint and 12 GB of swap on a 24 GB machine. See
  :func:`should_group_offload`.
- The MPS allocator's default ceiling is 1.7x the recommended working set, which
  on a 24 GB machine is 30 GB -- past physical RAM, into swap. See
  :func:`configure_mps_env`.
- Sort-like reductions over 5-D tensors abort the process outright. See
  :func:`guard_mps_sort_ndim`.
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
#: cost grows with the square of the token count. Measured directly, bf16, one
#: sequence, ten heads, ``driver_allocated_memory`` after a single
#: ``scaled_dot_product_attention``:
#:
#:     resolution   latent tokens   allocated
#:            768           9,216      4.2 GB
#:           1024          16,384     12.6 GB
#:           1280          25,600     25.4 GB
#:
#: That measurement is what set this to 768 at first, and it was too generous,
#: because it only sized *one* sequence. ``apply_layerdiff`` runs the schedule
#: twice for tag v3, and the second pass -- a crop around the head, generating
#: eleven tags -- attends with an effective batch of about ten. A real run failed
#: an hour in with
#:
#:     RuntimeError: Invalid buffer size: 15.82 GiB
#:
#: and 15.82 GiB is exactly ``100 x 9216^2 x 2`` bytes: ten sequences x ten heads
#: at 768's token count. So the per-call cost is roughly
#:
#:     bytes ~= batch * heads * (res/8)^4 * 2
#:
#: and against the ~10.8 GB ceiling ``configure_mps_env`` sets, that gives
#:
#:     resolution   tokens   one attention call
#:            768    9,216            15.8 GiB   <- fails
#:            640    6,400             7.6 GiB
#:            512    4,096             3.1 GiB
#:
#: Hence 640: the largest that leaves the head pass inside the budget. The failure
#: mode is worth noting -- it is a clean exception, but it arrives *after* the body
#: pass has finished, so the cost of getting this wrong is the whole run. What
#: makes that survivable is ``psd_io.read_layer_dir``: every layer both passes
#: produced is already on disk, so the hour is recoverable.
#:
#: Group offloading cannot help here. It pages *parameters*, and this is
#: activations -- LayerDiff 3D generates the layers as frames in one batch, so
#: they scale together.
MPS_MAX_RESOLUTION = 640


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
        f"{resolution} -> {MPS_MAX_RESOLUTION}: MPS materialises attention "
        f"scores, so {resolution} needs more memory than this device has "
        "(see ocs.torch_device.MPS_MAX_RESOLUTION)"
    )


#: MPS's sort kernel (``MPSNDArraySort``) only handles axes 0-3, so any sort-like
#: reduction over a tensor with more dimensions than this has to go via CPU.
_MPS_SORT_MAX_NDIM = 4

#: Reductions that MPS implements on top of that sort kernel.
_SORT_LIKE = ("median", "sort", "msort", "quantile", "nanmedian", "nanquantile",
              "kthvalue", "mode")


def guard_mps_sort_ndim() -> None:
    """Route sort-like reductions over >4-D tensors through CPU on MPS.

    LayerDiff 3D carries ``(augment, batch, channel, h, w)`` tensors, and both
    ``layerdiffuse/vae.py`` (``torch.median(result, dim=0)`` over the augmentation
    stack) and ``marigold/util/ensemble.py`` (four ``torch.median(..., dim=0)``
    calls over the ensemble) reduce across the leading axis of one. MPS lowers that
    to ``MPSNDArraySort``, which supports axes 0-3 only:

        MPSNDArraySort.mm:252: failed assertion
        `Axis = 4. This class only supports axis = 0, 1, 2, 3'

    That is a Metal assertion, not a Python exception: it calls ``abort()``, the
    process dies with SIGABRT, and no ``try``/``except`` or
    ``PYTORCH_ENABLE_MPS_FALLBACK`` can intercept it -- the op *is* implemented,
    just not at this rank. So the check has to be made before the call, on ndim.

    It is also why this cannot be left to fail loudly and be fixed later: it fires
    after the 15-minute denoise loop has already finished, taking the whole run
    with it.
    """
    import torch

    def wrap(name: str):
        original = getattr(torch, name)

        def patched(input, *args, **kwargs):
            if (
                isinstance(input, torch.Tensor)
                and input.device.type == "mps"
                and input.ndim > _MPS_SORT_MAX_NDIM
            ):
                result = original(input.cpu(), *args, **kwargs)
                return _to_device(result, input.device)
            return original(input, *args, **kwargs)

        patched.__name__ = name
        patched.__doc__ = original.__doc__
        return patched

    for name in _SORT_LIKE:
        if hasattr(torch, name):
            setattr(torch, name, wrap(name))


def _to_device(result: Any, device: Any) -> Any:
    """Move a tensor, or the tensors inside a return_types structseq, to ``device``.

    ``torch.median(x, dim=0)`` hands back a ``torch.return_types.median``, and
    callers reach for ``.values``, so the type has to survive the round trip --
    returning a plain tuple would break ``.values`` at the call site.
    """
    import torch

    if isinstance(result, torch.Tensor):
        return result.to(device)
    if isinstance(result, tuple):
        moved = [v.to(device) if isinstance(v, torch.Tensor) else v for v in result]
        try:
            return type(result)(moved)
        except TypeError:
            return tuple(moved)
    return result


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

    # torch.cuda.empty_cache() is a safe no-op without CUDA, which is why it is
    # tempting to leave alone -- but safe is not the same as working. see-through
    # calls it at three points to hand memory back after freeing something big,
    # the important one being right after it unloads both text encoders (~1.6 GB)
    # and just before the denoise loop. Measured: after dropping 3 GB of tensors,
    # torch.cuda.empty_cache() leaves driver_allocated_memory at 3.00 GB and
    # torch.mps.empty_cache() takes it to 0.00. On a device whose budget is
    # 17.8 GB, leaving that in the allocator is the difference between fitting and
    # swapping.
    if device.type == "mps":
        torch.cuda.empty_cache = torch.mps.empty_cache
        torch.cuda.synchronize = lambda *a, **k: torch.mps.synchronize()
        guard_mps_sort_ndim()

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


#: Fraction of physical RAM the MPS allocator may hold.
#:
#: Deliberately under half, and the reason is the part that is easy to get wrong:
#: on unified memory the MPS allocator and "CPU" memory are the *same* 24 GB, so
#: group offloading does not reduce total demand -- it moves the SDXL weights
#: (~8 GB) out of the MPS allocator and into CPU-resident copies that still occupy
#: physical RAM. The budget therefore has to cover only what stays on-device
#: (activations, and whichever block is paged in), not the whole model.
#:
#: Sizing it as if the allocator were the only consumer is what produced a 14.9 GB
#: ceiling that still collapsed: 14.9 + ~8 GB offloaded + the OS overruns 24 GB,
#: which showed up as steps 1-20 running at 28-38 s/it and step 21 jumping to
#: 295 s/it as the system finally started paging.
_MPS_RAM_BUDGET = 0.45


def configure_mps_env() -> dict[str, str]:
    """Environment torch must see *before* it initialises MPS.

    Two settings, and the second one is the difference between a run finishing and
    a run that looks like it hung.

    ``PYTORCH_ENABLE_MPS_FALLBACK`` lets an unimplemented MPS op drop to CPU
    rather than raising. Slow where it triggers, better than a hard stop partway
    through a long run over one missing kernel.

    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` caps the allocator, and its default is
    the trap. The ratio is relative to ``recommended_max_memory()`` -- 17.8 GB
    here -- and it defaults to **1.7**, so the allocator is free to grow to
    30.2 GB on a machine with 24 GB of RAM. It does exactly that: measured over
    one 768 run, the GPU driver reached 30.9 GB of allocated system memory (1.7 x
    17.76 = 30.19, so this is the ratio and nothing else), which put the system
    25.7 GB into swap. The process footprint stayed a healthy 13 GB the whole
    time, which is why this hides from the obvious diagnostic -- the growth is in
    the driver's cache, not the process heap.

    What it looks like: step time climbing without bound while the GPU stays busy.

        step  1-2    22 s/it
        step 20     153 s/it
        step 21     356 s/it
        step 23     427 s/it

    Capping the allocator inside physical RAM is what stops that. The low
    watermark is set below the high one so cached blocks get purged before the
    allocator reaches for more.

    Returns the variables it set, for logging.
    """
    import torch

    env = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}

    recommended = torch.mps.recommended_max_memory()
    if recommended > 0:
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        budget = physical * _MPS_RAM_BUDGET
        # Never raise the ceiling above what Metal recommends for one working set;
        # this is only ever meant to lower it.
        high = min(budget / recommended, 1.0)
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{high:.2f}"
        env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{high * 0.7:.2f}"

    for key, value in env.items():
        os.environ.setdefault(key, value)
    return {k: os.environ[k] for k in env}
