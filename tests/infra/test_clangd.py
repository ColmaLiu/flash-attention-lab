"""Editor configuration must reflect the host/device ABI without using a GPU."""

import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from scripts import configure_clangd


def touch(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def toolkit_at(root):
    touch(root / "include/cuda.h")
    touch(root / "include/cuda_runtime.h")
    touch(root / "nvvm/libdevice/libdevice.10.bc")
    touch(root / "lib/libcudart.so")
    return root


@pytest.mark.parametrize("layout", ["", "targets/x86_64-linux"])
def test_clang_cuda_root_finds_system_and_conda_layouts(tmp_path, layout):
    toolkit = tmp_path / "cuda"
    expected = toolkit_at(toolkit / layout)
    assert configure_clangd.clang_cuda_root(toolkit) == expected


@pytest.mark.parametrize(
    "missing", ["include/cuda.h", "nvvm/libdevice/libdevice.10.bc"]
)
def test_incomplete_toolkit_is_rejected(tmp_path, missing):
    toolkit = toolkit_at(tmp_path / "cuda")
    (toolkit / missing).unlink()
    with pytest.raises(RuntimeError, match="headers and libdevice"):
        configure_clangd.clang_cuda_root(toolkit)


def test_curand_prefers_toolkit_headers(tmp_path, monkeypatch):
    include = tmp_path / "cuda/include"
    touch(include / "curand_mtgp32_kernel.h")

    def no_wheel(name):
        pytest.fail("a complete Toolkit does not need a cuRAND wheel")

    monkeypatch.setattr(configure_clangd, "distribution", no_wheel)
    assert configure_clangd.clang_cuda_includes([include]) == [include]


def test_curand_falls_back_to_wheel_metadata(tmp_path, monkeypatch):
    include = tmp_path / "cuda/include"
    wheel = tmp_path / "site-packages"
    curand = wheel / "nvidia/curand/include"
    touch(curand / "curand_mtgp32_kernel.h")

    class Wheel:
        def locate_file(self, name):
            return wheel / name

    def find_wheel(name):
        assert name == "nvidia-curand-cu12"
        return Wheel()

    monkeypatch.setattr(configure_clangd, "distribution", find_wheel)
    assert configure_clangd.clang_cuda_includes([include]) == [include, curand]


@pytest.mark.parametrize("installed", [False, True])
def test_missing_curand_header_is_rejected(tmp_path, monkeypatch, installed):
    class Wheel:
        def locate_file(self, name):
            return tmp_path / name

    def find_wheel(name):
        if not installed:
            raise PackageNotFoundError(name)
        return Wheel()

    monkeypatch.setattr(configure_clangd, "distribution", find_wheel)
    with pytest.raises(RuntimeError, match="curand_mtgp32_kernel.h"):
        configure_clangd.clang_cuda_includes([tmp_path / "cuda/include"])


def test_commands_cover_custom_student_files_and_use_matching_language(tmp_path):
    root = tmp_path / "checkout with spaces"
    sources = [
        "flash_attention_lab/ffi_api.h",
        "flash_attention_lab/binding.cpp",
        "solutions/s43_schedule/kernel.cu",
        "solutions/s43_schedule/attention_dense.cuh",
        "students/cuda/s00_layout/binding.cpp",
        "students/cuda/s00_layout/helper.cuh",
    ]
    for name in sources + ["third_party/ignored.cu", "artifacts/ignored.cpp"]:
        touch(root / name)
    toolkit = toolkit_at(tmp_path / "cuda")
    ffi = tmp_path / "tvm_ffi/include"
    commands = configure_clangd.compilation_commands(
        root, toolkit, [toolkit / "include"], ffi
    )
    assert {entry["file"] for entry in commands} == {str(root / n) for n in sources}
    for entry in commands:
        args = entry["arguments"]
        assert entry["directory"] == str(root)
        assert args[-2:] == ["-c", entry["file"]]
        assert str(ffi) in args
        assert "-I" + str(root / "flash_attention_lab") in args
        assert not any("TORCH" in arg or "__CUDA_NO_" in arg for arg in args)
        if Path(entry["file"]).suffix in (".cu", ".cuh"):
            assert "-std=c++17" in args
            assert args[args.index("-x") + 1] == "cuda"
            assert "--cuda-gpu-arch=sm_100a" in args
            assert "--cuda-path=" + str(toolkit) in args
        else:
            assert "-std=c++20" in args
            assert args[args.index("-x") + 1] == "c++"
            assert '-DLAB_FFI_MODULE="falab_editor"' in args
            assert not any(arg.startswith("--cuda-") for arg in args)


def test_standalone_header_inherits_its_owners_selector(tmp_path):
    root = tmp_path / "repo"
    lesson = root / "solutions/s31_tcgen"
    touch(lesson / "kernel_impl.cuh")
    owner = touch(
        lesson / "kernel.cu",
        '#define LAB_MULTISTAGE 0\n#include "kernel_impl.cuh"\n#define LAB_LATE 1\n',
    )
    toolkit = toolkit_at(tmp_path / "cuda")
    for selector in (0, 1):
        owner.write_text(
            owner.read_text().replace("MULTISTAGE 0", f"MULTISTAGE {selector}")
        )
        commands = configure_clangd.compilation_commands(root, toolkit, [], tmp_path)
        header = next(c for c in commands if c["file"].endswith(".cuh"))
        assert f"-DLAB_MULTISTAGE={selector}" in header["arguments"]
        assert "-DLAB_LATE=1" not in header["arguments"]
        source = next(c for c in commands if c["file"].endswith(".cu"))
        assert not any(arg.startswith("-DLAB_") for arg in source["arguments"])


def test_generation_needs_no_visible_gpu_and_keeps_sources(tmp_path, monkeypatch):
    import torch
    from flash_attention_lab import environment, runtime

    def no_gpu(*args, **kwargs):
        pytest.fail("editor configuration must not inspect or initialize CUDA")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(environment, "require_environment", no_gpu)
    monkeypatch.setattr(torch.cuda, "is_available", no_gpu)
    monkeypatch.setattr(torch.cuda, "init", no_gpu)
    toolkit = toolkit_at(tmp_path / "cuda")
    monkeypatch.setattr(runtime, "cuda_home", lambda: toolkit)
    root = tmp_path / "repo"
    source = touch(root / "students/cuda/s00_layout/kernel.cu", "// Student work\n")
    monkeypatch.setattr(configure_clangd, "ROOT", root)
    configure_clangd.main()
    entries = json.loads((root / "compile_commands.json").read_text())
    assert [entry["file"] for entry in entries] == [str(source)]
    assert source.read_text() == "// Student work\n"
