"""Native CUDA/PTX SM100 GEMM; tensor-map preparation is outside timing.

Zero-valued tile_m/tile_n/tile_k/stages select the host's shape-based defaults.
Long-K defaults use four stages: M64/N32/K256 for N<=128, otherwise
M128/N64/K128. Short-K QK uses M128/N128/K128 with one stage. split_k defaults
to one complete output per CTA; an explicit larger value times both partial
GEMM and its reduction. single_issuer=1 is an explicit pipeline ablation.
The host validates supported tile combinations; see README.md for layouts.
"""

from flash_attention_lab.build import build_cuda

IMPLEMENTED = True
BACKEND = "python"
CONFIG = {}
RUNTIME_REQUIRES = []


def workspace_spec(inputs, config):
    return {}


def prepare(inputs, outputs, workspace, config, stream):
    lib = build_cuda("s32_specialize", solution=True)
    a, b, out = inputs["a"], inputs["b"], outputs["c"]
    state = lib.prepare(
        a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        a.shape[0],
        b.shape[1],
        a.shape[1],
        stream,
        int(config.get("tile_m", 0)),
        int(config.get("tile_n", 0)),
        int(config.get("tile_k", 0)),
        int(config.get("split_k", 0)),
        int(config.get("stages", 0)),
        int(config.get("single_issuer", 0)),
    )

    def launch():
        lib.run(state)

    launch.extra_workspace_bytes = lib.workspace_bytes(state)
    return launch
