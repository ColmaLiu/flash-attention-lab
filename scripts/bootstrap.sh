#!/usr/bin/env bash
set -eo pipefail
# Conda activation hooks are not guaranteed to be nounset-safe.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
lab_with_cuda=false
case "${1:-}" in
    '') ;;
    --with-cuda) lab_with_cuda=true ;;
    --help|-h)
        echo 'Usage: bash scripts/bootstrap.sh [--with-cuda]'
        echo 'Creates LAB_CONDA_ENV (default flash-attention-lab); --with-cuda adds CUDA 12.8 and GCC 13.'
        exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 6 ;;
esac
if (( $# > 1 )); then echo 'Expected at most one option' >&2; exit 6; fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'ENV_UNSUPPORTED: this installation recipe targets Linux x86_64' >&2; exit 4
fi
lab_conda_exe="${LAB_CONDA_EXE:-${CONDA_EXE:-}}"
if [[ -z "$lab_conda_exe" ]]; then lab_conda_exe="$(type -P conda || true)"; fi
if [[ -z "$lab_conda_exe" || ! -x "$lab_conda_exe" ]]; then
    echo 'ENV_UNSUPPORTED: install Conda, or set LAB_CONDA_EXE to its executable' >&2; exit 4
fi
lab_env="${LAB_CONDA_ENV:-flash-attention-lab}"
lab_base="$("$lab_conda_exe" info --base)"
if [[ "$lab_env" == base || "${lab_env%/}" == "${lab_base%/}" ]]; then
    echo 'ENV_UNSUPPORTED: choose a dedicated LAB_CONDA_ENV, not base' >&2; exit 4
fi
lab_target=(--name "$lab_env")
if [[ "$lab_env" == /* ]]; then lab_target=(--prefix "$lab_env"); fi
if [[ "$lab_env" == */* && "$lab_env" != /* ]]; then
    echo 'ENV_UNSUPPORTED: LAB_CONDA_ENV must be a name or an absolute prefix' >&2; exit 4
fi
lab_hook="$("$lab_conda_exe" shell.bash hook)"
eval "$lab_hook"
if ! conda activate "$lab_env" 2>/dev/null; then
    "$lab_conda_exe" env create --yes "${lab_target[@]}" --file environment.yml
    conda activate "$lab_env"
fi
# Check the resolved prefix too, including a symlink alias of base.
if [[ "$CONDA_PREFIX" -ef "$lab_base" ]]; then
    echo 'ENV_UNSUPPORTED: refusing to install in base' >&2; exit 4
fi
if ! "$CONDA_PREFIX/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo 'ENV_UNSUPPORTED: existing environment needs Python 3.12; choose a new LAB_CONDA_ENV' >&2; exit 4
fi
if $lab_with_cuda; then
    "$lab_conda_exe" install --yes "${lab_target[@]}" --override-channels \
        -c nvidia/label/cuda-12.8.1 -c conda-forge 'cuda-toolkit=12.8' 'gxx_linux-64=13'
    conda deactivate
    conda activate "$lab_env"
fi
# Both source versions are owned by the parent gitlinks. Do not recursively
# initialize upstream reference repositories' unrelated nested dependencies.
if ! command -v git >/dev/null || ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo 'ENV_UNSUPPORTED: use a Git clone with its pinned source submodules' >&2; exit 4
fi
git submodule update --init -- third_party/cutlass third_party/flash-attention
"$CONDA_PREFIX/bin/python" -m pip install -r requirements.txt
"$CONDA_PREFIX/bin/python" -m pip check
echo 'Installed. Run ./lab doctor, ./lab smoke, then ./lab task s00_layout.'
