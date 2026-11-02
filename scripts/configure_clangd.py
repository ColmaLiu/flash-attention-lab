"""Generate local clangd commands; does not build or implement student kernels.

Run from the repository root after sourcing scripts/activate.sh:
    python scripts/configure_clangd.py
"""
import json
from pathlib import Path
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from flash_attention_lab.runtime import configure_runtime, cuda_home, cuda_development_paths
    configure_runtime()
    from flash_attention_lab.environment import require_environment
    # Inspect the selected GPU before selecting the same target as build.py.
    require_environment("cuda")
    from torch.utils.cpp_extension import include_paths

    toolkit = cuda_home()
    cuda_includes, _ = cuda_development_paths(toolkit)
    common = ["clang++", "-std=c++17", "-fPIC", "-DTORCH_API_INCLUDE_EXTENSION_H",
              "-DTORCH_EXTENSION_NAME=falab_editor", "-I" + str(ROOT / "flash_attention_lab")]
    for path in dict.fromkeys([*map(str, cuda_includes), *include_paths(), sysconfig.get_path("include")]):
        common += ["-isystem", path]
    cuda_flags = ["-x", "cuda", "--cuda-path=" + str(toolkit), "--cuda-gpu-arch=sm_100a",
                  "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                  "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]
    sources = sorted((ROOT / "students" / "cuda").rglob("*.cu"))
    sources += sorted(p for p in (ROOT / "solutions").rglob("*") if p.suffix in (".cu", ".cuh", ".cpp"))
    sources += sorted((ROOT / "flash_attention_lab").glob("*.cu"))
    # An explicit header entry avoids C-language fallback when opened first.
    sources += sorted((ROOT / "flash_attention_lab").glob("*.h"))
    sources += sorted((ROOT / "flash_attention_lab").glob("*.cpp"))
    entries = []
    for source in sources:
        entries.append({"directory": str(ROOT), "file": str(source),
                        "arguments": common
                        + (cuda_flags if source.suffix in (".cu", ".h", ".cuh") else ["-x", "c++"])
                        + ["-c", str(source)]})
    output = ROOT / "compile_commands.json"
    output.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {len(entries)} editor commands to {output}")
    print(f"CUDA Toolkit: {toolkit}; editor target: sm_100a")
    print("Restart the clangd language server. This is not student correctness validation.")


if __name__ == "__main__":
    main()
