#!/usr/bin/env bash
# Source interactively or from ./lab; preserve the caller's shell options and cwd.
_flash_lab_activate() {
    local lab_exe="${LAB_CONDA_EXE:-${CONDA_EXE:-}}"
    local lab_target="${LAB_CONDA_ENV:-flash-attention-lab}"
    local lab_hook lab_restore_nounset=false lab_code=0
    if [[ -z "$lab_exe" ]]; then lab_exe="$(type -P conda || true)"; fi
    if [[ -z "$lab_exe" || ! -x "$lab_exe" ]]; then
        echo 'ENV_UNSUPPORTED: conda not found; set LAB_CONDA_EXE to its executable' >&2
        return 4
    fi
    if [[ "$lab_target" == base ]]; then
        echo 'ENV_UNSUPPORTED: choose a dedicated LAB_CONDA_ENV, not base' >&2
        return 4
    fi
    lab_hook="$("$lab_exe" shell.bash hook)" || return 4
    [[ $- == *u* ]] && lab_restore_nounset=true
    # Conda package activation hooks may read unset variables (e.g. NVCC flags).
    set +u
    if ! eval "$lab_hook" || ! conda activate "$lab_target"; then
        echo 'ENV_UNSUPPORTED: run bash scripts/bootstrap.sh to create the project Conda environment' >&2
        lab_code=4
    else
        export PATH="$CONDA_PREFIX/bin:$PATH"
    fi
    if $lab_restore_nounset; then set -u; fi
    return "$lab_code"
}
if _flash_lab_activate; then
    unset -f _flash_lab_activate
else
    unset -f _flash_lab_activate
    return 4
fi
