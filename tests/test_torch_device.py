"""The cuda→device redirection has to be narrow, not a blanket rewrite.

The risk is not that `device='cuda'` fails to move -- that is the easy half and
obvious when it breaks. It is that the patch is too eager and quietly drags
something else onto the accelerator: `.to('cpu')` is used all over see-through's
post-processing to pull tensors back for numpy, and a `.to(dtype)` with no device
must not relocate anything.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the GPU half is an optional install")

from ocs import torch_device


@pytest.fixture
def target(monkeypatch):
    """Redirect to CPU, on a fresh copy of the methods the patch replaces.

    CPU is the one target every machine has, and it is still a *different* device
    from the 'cuda' being asked for, so the rewrite is genuinely exercised.
    """
    monkeypatch.setattr(torch.nn.Module, "to", torch.nn.Module.to)
    monkeypatch.setattr(torch.nn.Module, "cuda", torch.nn.Module.cuda)
    monkeypatch.setattr(torch.Tensor, "cuda", torch.Tensor.cuda)
    monkeypatch.setattr(torch_device, "_PATCHED", False)
    device = torch.device("cpu")
    torch_device.redirect_cuda_to(device)
    monkeypatch.setattr(torch_device, "_PATCHED", False)
    return device


def test_selects_override_first(monkeypatch):
    monkeypatch.setenv("OCS_TORCH_DEVICE", "cpu")
    assert torch_device.select_device().type == "cpu"


def test_explicit_prefer_beats_env(monkeypatch):
    monkeypatch.setenv("OCS_TORCH_DEVICE", "cuda")
    assert torch_device.select_device("cpu").type == "cpu"


def test_group_offload_on_accelerators_but_not_cpu():
    # MPS needs it more than CUDA does: without it a 1280 run swaps on 24 GB.
    assert torch_device.should_group_offload(torch.device("cuda"), requested=True)
    assert torch_device.should_group_offload(torch.device("mps"), requested=True)
    # Nothing to offload to on CPU.
    assert not torch_device.should_group_offload(torch.device("cpu"), requested=True)
    # And an unrequested offload stays off even where it would work.
    assert not torch_device.should_group_offload(torch.device("cuda"), requested=False)


def test_settings_pass_group_offload_to_accelerators():
    from ocs.config import SeeThroughSettings

    s = SeeThroughSettings(group_offload=True)
    assert "--group_offload" in s.to_args(device_type="cuda")
    assert "--group_offload" in s.to_args(device_type="mps")
    assert "--group_offload" not in s.to_args(device_type="cpu")
    # tblr_split is unconditional -- it is what produces the -l/-r tags.
    assert "--tblr_split" in s.to_args(device_type="mps")


def test_group_offload_onload_device_is_rewritten(target, monkeypatch):
    """see-through passes 'cuda' positionally as onload_device."""
    from diffusers import ModelMixin

    seen = {}

    def fake(self, *args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs

    monkeypatch.setattr(ModelMixin, "enable_group_offload", fake)
    monkeypatch.setattr(torch_device, "_PATCHED", False)
    torch_device.redirect_cuda_to(target)

    class M(ModelMixin):
        pass

    M().enable_group_offload("cuda", num_blocks_per_group=1)
    assert seen["args"][0] == target
    assert seen["kwargs"]["num_blocks_per_group"] == 1
    # pin_memory() fails on MPS, and that is the only path that calls it.
    assert seen["kwargs"]["use_stream"] is False
    monkeypatch.setattr(torch_device, "_PATCHED", False)


@pytest.mark.parametrize("call", [
    lambda m, d: m.to(device="cuda"),                  # the five layerdiff sites
    lambda m, d: m.to("cuda"),                         # positional string
    lambda m, d: m.to(torch.device("cuda")),           # device object
    lambda m, d: m.to(torch.device("cuda:0")),         # indexed
    lambda m, d: m.to(device="cuda", dtype=torch.float16),
])
def test_cuda_requests_are_redirected(target, call):
    m = torch.nn.Linear(4, 4)
    call(m, target)
    assert m.weight.device.type == target.type


def test_dtype_is_preserved_through_the_rewrite(target):
    # inference_utils asks for bfloat16 in the same call as the device; losing
    # the dtype would silently run the whole pipeline in fp32.
    m = torch.nn.Linear(4, 4)
    m.to(dtype=torch.bfloat16, device="cuda")
    assert m.weight.dtype == torch.bfloat16
    assert m.weight.device.type == target.type


def test_cpu_placement_is_not_hijacked(target):
    m = torch.nn.Linear(4, 4).to("cuda")
    m.to("cpu")
    assert m.weight.device.type == "cpu"
    t = torch.zeros(2).to("cpu")
    assert t.device.type == "cpu"


def test_dtype_only_call_does_not_move_anything(target):
    m = torch.nn.Linear(4, 4)
    before = m.weight.device
    m.to(torch.float64)
    assert m.weight.dtype == torch.float64
    assert m.weight.device == before


def test_dot_cuda_methods_are_redirected(target):
    assert torch.nn.Linear(4, 4).cuda().weight.device.type == target.type
    assert torch.zeros(2).cuda().device.type == target.type


def test_redirect_is_idempotent(monkeypatch):
    """Double-patching would wrap the wrapper and recurse one layer per call."""
    monkeypatch.setattr(torch.nn.Module, "to", torch.nn.Module.to)
    monkeypatch.setattr(torch_device, "_PATCHED", False)
    torch_device.redirect_cuda_to(torch.device("cpu"))
    once = torch.nn.Module.to
    torch_device.redirect_cuda_to(torch.device("cpu"))
    assert torch.nn.Module.to is once
    monkeypatch.setattr(torch_device, "_PATCHED", False)


def test_redirect_to_cuda_is_a_noop(monkeypatch):
    """On a real CUDA box nothing should be wrapped at all."""
    monkeypatch.setattr(torch.nn.Module, "to", torch.nn.Module.to)
    monkeypatch.setattr(torch_device, "_PATCHED", False)
    original = torch.nn.Module.to
    torch_device.redirect_cuda_to(torch.device("cuda"))
    assert torch.nn.Module.to is original


def test_describe_reports_the_device_it_was_given():
    info = torch_device.describe(torch.device("cpu"))
    assert info["device"] == "cpu"
    assert info["torch"] == torch.__version__
    assert info["group_offload_supported"] is False


def test_resolution_cap_only_applies_to_mps():
    # 1280 needs 25.4 GB for one attention call on MPS; CUDA has the kernels
    # that make it cheap, so it must not be touched there.
    assert torch_device.cap_resolution(1280, torch.device("cuda")) == (1280, None)
    assert torch_device.cap_resolution(1280, torch.device("cpu")) == (1280, None)

    capped, note = torch_device.cap_resolution(1280, torch.device("mps"))
    assert capped == torch_device.MPS_MAX_RESOLUTION
    assert note and "1280" in note


def test_resolution_under_the_cap_is_left_alone():
    for res in (512, 640, torch_device.MPS_MAX_RESOLUTION):
        assert torch_device.cap_resolution(res, torch.device("mps")) == (res, None)


def test_cap_rewrites_argv_and_says_so(capsys):
    from scripts.run_seethrough import _cap_resolution_args

    argv = ["x.py", "--srcp", "a.png", "--resolution", "1280",
            "--resolution_depth", "768", "--seed", "42"]
    _cap_resolution_args(argv, torch.device("mps"))
    assert argv[argv.index("--resolution") + 1] == str(torch_device.MPS_MAX_RESOLUTION)
    # 768 is already at the cap, so it stays and draws no note.
    assert argv[argv.index("--resolution_depth") + 1] == "768"
    assert "--seed" in argv and argv[argv.index("--seed") + 1] == "42"
    assert "[ocs] --resolution" in capsys.readouterr().out


def test_cap_tolerates_a_missing_or_unparseable_value(capsys):
    from scripts.run_seethrough import _cap_resolution_args

    # Trailing flag with no value, and a non-integer -- neither should raise.
    for argv in (["x.py", "--resolution"], ["x.py", "--resolution", "auto"]):
        before = list(argv)
        _cap_resolution_args(argv, torch.device("mps"))
        assert argv == before


def test_offload_device_stays_on_cpu(target, monkeypatch):
    """onload moves to the accelerator; offload must not follow it.

    If both ended up on the same device the offloading would be a no-op, which is
    precisely the 35 GB-footprint failure this path exists to prevent.
    """
    from diffusers import ModelMixin

    seen = {}
    monkeypatch.setattr(
        ModelMixin, "enable_group_offload",
        lambda self, *a, **k: seen.update(args=a, kwargs=k),
    )
    monkeypatch.setattr(torch_device, "_PATCHED", False)
    torch_device.redirect_cuda_to(target)

    class M(ModelMixin):
        pass

    M().enable_group_offload(onload_device=torch.device("cuda"),
                             offload_device=torch.device("cpu"))
    assert seen["kwargs"]["onload_device"] == target
    assert seen["kwargs"]["offload_device"] == torch.device("cpu")
    monkeypatch.setattr(torch_device, "_PATCHED", False)


def test_redirect_accepts_a_plain_string_device(monkeypatch):
    monkeypatch.setattr(torch.nn.Module, "to", torch.nn.Module.to)
    monkeypatch.setattr(torch_device, "_PATCHED", False)
    torch_device.redirect_cuda_to("cpu")          # not a torch.device
    assert torch.nn.Linear(2, 2).to("cuda").weight.device.type == "cpu"
    monkeypatch.setattr(torch_device, "_PATCHED", False)


def test_empty_cache_is_repointed_on_mps(monkeypatch):
    """A no-op empty_cache leaves freed memory in the allocator.

    see-through calls torch.cuda.empty_cache() right after unloading both text
    encoders. Without this the ~1.6 GB stays cached for the rest of the run.
    """
    monkeypatch.setattr(torch.nn.Module, "to", torch.nn.Module.to)
    monkeypatch.setattr(torch.cuda, "empty_cache", torch.cuda.empty_cache)
    monkeypatch.setattr(torch_device, "_PATCHED", False)

    torch_device.redirect_cuda_to(torch.device("mps"))
    assert torch.cuda.empty_cache is torch.mps.empty_cache
    monkeypatch.setattr(torch_device, "_PATCHED", False)


def test_empty_cache_is_left_alone_off_mps(monkeypatch):
    monkeypatch.setattr(torch.nn.Module, "to", torch.nn.Module.to)
    monkeypatch.setattr(torch.cuda, "empty_cache", torch.cuda.empty_cache)
    monkeypatch.setattr(torch_device, "_PATCHED", False)
    original = torch.cuda.empty_cache

    torch_device.redirect_cuda_to(torch.device("cpu"))
    assert torch.cuda.empty_cache is original
    monkeypatch.setattr(torch_device, "_PATCHED", False)


def test_mps_env_caps_the_allocator_inside_physical_ram(monkeypatch):
    """The default high watermark is 1.7x recommended, which overshoots RAM.

    On a 24 GB machine that is a 30 GB ceiling, and the allocator uses it -- the
    driver reached 30.9 GB and drove 25 GB of swap. The cap has to land below
    physical RAM or the run degrades without bound.
    """
    import os

    for key in ("PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO"):
        monkeypatch.delenv(key, raising=False)

    env = torch_device.configure_mps_env()
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

    if not torch.backends.mps.is_available():
        pytest.skip("watermark ratios are only set where MPS exists")

    high = float(env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
    low = float(env["PYTORCH_MPS_LOW_WATERMARK_RATIO"])
    recommended = torch.mps.recommended_max_memory()
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    assert recommended * high < physical, "ceiling must sit inside physical RAM"
    assert high < 1.7, "must be below torch's default, which is the bug"
    assert 0 < low < high, "purge threshold has to trip before the ceiling"


def test_mps_env_does_not_override_a_deliberate_setting(monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    env = torch_device.configure_mps_env()
    assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.5"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_five_dim_sortlike_does_not_abort_the_process():
    """MPSNDArraySort supports axes 0-3; a 5-D reduction calls abort().

    Not raises -- aborts. SIGABRT, uncatchable, and it fires after the denoise
    loop has already spent 15 minutes. LayerDiff 3D reduces over the leading axis
    of (augment, batch, channel, h, w) in vae.py and in marigold's ensemble.
    """
    torch_device.guard_mps_sort_ndim()
    x = torch.randn(3, 2, 4, 8, 8, device="mps")

    r = torch.median(x, dim=0)
    # The structseq type has to survive the CPU round trip: callers use .values.
    assert type(r).__name__ == "median"
    assert r.values.shape == (2, 4, 8, 8)
    assert r.values.device.type == "mps"
    # Right answer, not merely a surviving one.
    assert torch.allclose(r.values.cpu(), torch.median(x.cpu(), dim=0).values)

    assert torch.sort(x, dim=0).values.device.type == "mps"
    assert torch.quantile(x.float(), 0.5, dim=0).device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_four_dim_stays_on_mps():
    """The guard must not add a CPU round trip to shapes MPS handles fine."""
    torch_device.guard_mps_sort_ndim()
    x = torch.randn(2, 4, 8, 8, device="mps")
    assert torch.median(x, dim=0).values.device.type == "mps"
    assert x.ndim == torch_device._MPS_SORT_MAX_NDIM


def test_sort_guard_leaves_cpu_tensors_alone():
    torch_device.guard_mps_sort_ndim()
    x = torch.randn(3, 2, 2, 2, 2)
    r = torch.median(x, dim=0)
    assert r.values.device.type == "cpu"
    assert torch.allclose(r.values, torch.median(x, dim=0).values)


def test_signal_exits_are_named_not_shown_as_negative_codes():
    from ocs.seethrough import _describe_exit

    # "exited -15" sends the reader hunting for a see-through bug that isn't one.
    assert "SIGTERM" in _describe_exit(-15)
    assert "terminated externally" in _describe_exit(-15)
    assert "out-of-memory" in _describe_exit(-9)
    assert _describe_exit(1) == "inference_psd.py exited 1"


def test_error_tail_strips_progress_noise(tmp_path):
    """Dropping --disable_progressbar made the raw tail a wall of tqdm redraws."""
    from ocs.seethrough import _error_tail

    log = tmp_path / "seethrough.log"
    log.write_text(
        "\n".join(
            ["Loading weights: 100%|####| 517/517 [00:00<00:00, 7060.70it/s]"] * 50
            + ["  4%|#   | 1/30 [01:08<32:59, 68.24s/it]"] * 50
            + ["Traceback (most recent call last):", "RuntimeError: the real cause"]
        ),
        encoding="utf-8",
    )
    tail = _error_tail(log)
    assert "RuntimeError: the real cause" in tail
    assert "Traceback" in tail
    assert "it/s" not in tail and "s/it" not in tail


def test_error_tail_says_so_when_there_is_nothing_useful(tmp_path):
    from ocs.seethrough import _error_tail

    log = tmp_path / "seethrough.log"
    log.write_text("Loading weights: 100%|####| 1/1 [00:00<00:00, 10it/s]\n")
    assert "no diagnostic output" in _error_tail(log)
    assert _error_tail(None) == ""
