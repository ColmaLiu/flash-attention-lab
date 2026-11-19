"""DLPack ownership and explicit build configuration at the FFI boundary."""

import gc
import os

import pytest
import torch
import tvm_ffi

from flash_attention_lab import build
from flash_attention_lab.ffi import launch_arguments, tensor_arguments


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.uint8,
        torch.int64,
    ],
)
def test_tensor_arguments_keep_offset_storage_alive_without_copying(dtype):
    storage = torch.arange(32, dtype=dtype)
    view = storage[8:24].view(4, 4)
    address = view.data_ptr()
    args = tensor_arguments({"x": view}, 0)
    assert isinstance(args, tvm_ffi.Map)
    del view, storage
    gc.collect()
    result = torch.from_dlpack(args["x"])
    assert result.data_ptr() == address
    torch.testing.assert_close(result, torch.arange(8, 24, dtype=dtype).view(4, 4))
    result.fill_(3)
    assert torch.from_dlpack(args["x"]).eq(3).all()


def test_noncontiguous_tensor_arguments_are_rejected():
    with pytest.raises((ValueError, RuntimeError), match="contiguous"):
        tensor_arguments({"x": torch.ones(3, 4).T}, 0)


def test_launch_arguments_convert_options_once_and_keep_empty_maps():
    args = launch_arguments({}, {}, {}, {"causal": True, "scale": 0.125}, 0)
    assert all(isinstance(value, tvm_ffi.Map) for value in args[:4])
    assert all(len(value) == 0 for value in args[:3])
    assert dict(args[3]) == {"causal": 1.0, "scale": 0.125}
    assert args[4] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_ffi_build_preserves_toolchain_target_and_restores_environment(
    tmp_path, monkeypatch, fail
):
    from tvm_ffi import cpp

    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setenv("CUDA_HOME", "/inherited/cuda")
    monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "9.0a")
    monkeypatch.setenv("CC", "/toolchain/gcc")
    captured = {}

    def load(**kwargs):
        captured.update(kwargs)
        assert os.environ["CUDA_HOME"] == "/selected/cuda"
        assert os.environ["TVM_FFI_CUDA_ARCH_LIST"] == "10.0a"
        if fail:
            raise RuntimeError("fixture build failed")
        return "module"

    monkeypatch.setattr(cpp, "load", load)
    flags = ("-O3", "-std=c++17", "-gencode=arch=compute_100a,code=sm_100a")
    arguments = (
        "fixture",
        "kernel.cu",
        "binding.cpp",
        "/selected/cuda",
        "/selected/lib",
        ("/selected/include",),
        flags,
        "testhash",
    )
    if fail:
        with pytest.raises(RuntimeError, match="fixture build failed"):
            build._build.__wrapped__(*arguments)
    else:
        assert build._build.__wrapped__(*arguments) == "module"
    assert os.environ["CUDA_HOME"] == "/inherited/cuda"
    assert os.environ["TVM_FFI_CUDA_ARCH_LIST"] == "9.0a"
    assert captured["backend"] == "cuda"
    assert captured["keep_module_alive"] is True
    assert captured["sources"] == ["binding.cpp", "kernel.cu"]
    assert captured["extra_cuda_cflags"] == [
        "-O3",
        "-std=c++17",
        "-ccbin",
        "/toolchain/gcc",
    ]
    assert "-std=c++20" in captured["extra_cflags"]
