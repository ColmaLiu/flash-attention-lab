"""Directly compile the submodule FA4 SM100 device implementation.

This module is host integration. The complete kernel, pipeline, MMA and softmax
sources live in third_party/flash-attention/flash_attn/cute, with their license.
The source patch exposes its rescale threshold for controlled experiments.
"""
from pathlib import Path
import hashlib
import sys
import types
from flash_attention_lab.registry import ROOT
from flash_attention_lab.errors import EnvironmentUnsupported
from flash_attention_lab.cpp_dependencies import require_source
from flash_attention_lab.source_cache import prepare_flash_attention_source

_COMPILED = {}
_SOURCE_DIGEST = None


def _check_source_digest():
    global _SOURCE_DIGEST
    digest = hashlib.sha256()
    paths = sorted((require_source("flash-attention") / "flash_attn/cute").rglob("*.py"))
    paths.extend([Path(__file__), ROOT / "patches/flash-attention/rescale-threshold.patch", ROOT / "flash_attention_lab/source_cache.py"])
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    current = digest.hexdigest()
    if _SOURCE_DIGEST is not None and current != _SOURCE_DIGEST:
        raise EnvironmentUnsupported("FA4 reference source changed in this process; start a fresh lab process before validation or timing")
    _SOURCE_DIGEST = current


def _source_namespace(source_root):
    """Load the pinned source, never an unrelated installed flash_attn package."""
    base = source_root / "flash_attn"
    for name, path in (("flash_attn", base), ("flash_attn.cute", base / "cute")):
        if name in sys.modules:
            paths = [Path(p).resolve() for p in getattr(sys.modules[name], "__path__", ())]
            if path.resolve() not in paths:
                raise EnvironmentUnsupported("another flash_attn package is loaded; run this reference in a fresh lab process")
        else:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__package__ = name
            sys.modules[name] = module


def prepare_fa4(inputs, outputs, config, stream, *, q_stage, emulated_exp, rescale_threshold, persistent):
    _check_source_digest()
    _source_namespace(prepare_flash_attention_source())
    try:
        import cutlass
        import cutlass.cute as cute
        import cuda.bindings.driver as cuda
        from flash_attn.cute.cute_dsl_utils import to_cute_tensor
        from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
    except ImportError as exc:
        raise EnvironmentUnsupported("FA4 reference dependencies missing: run bash scripts/bootstrap.sh") from exc
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    tensors = tuple(to_cute_tensor(t, enable_tvm_ffi=False) for t in (q, k, v, outputs["o"], outputs["lse"]))
    key = (tuple(q.shape), q.dtype, config["causal"], q_stage, emulated_exp, rescale_threshold, persistent)
    if key not in _COMPILED:
        kernel = FlashAttentionForwardSm100(
            q.shape[-1], is_causal=config["causal"], m_block_size=128,
            n_block_size=128, q_stage=q_stage, is_static_persistent=persistent,
            use_2cta_instrs=False, rescale_threshold=rescale_threshold,
        )
        kernel.enable_ex2_emu = emulated_exp
        _COMPILED[key] = cute.compile(kernel, *tensors, cutlass.Float32(config["scale"]), stream=cuda.CUstream(stream))
    compiled = _COMPILED[key]
    scale = cutlass.Float32(config["scale"])
    cu_stream = cuda.CUstream(stream)
    return lambda: compiled(*tensors, scale, stream=cu_stream)
