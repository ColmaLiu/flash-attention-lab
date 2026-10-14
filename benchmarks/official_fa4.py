"""Optional official comparison in a separately provisioned environment.

python -m benchmarks.official_fa4 --help
No installation side effects. Canonical BHND -> official BNHD conversion reported.
"""
import argparse
import importlib.metadata
import json
import time
from flash_attention_lab.runtime import configure_runtime
configure_runtime()
import torch
from flash_attention_lab.accuracy import compare
from flash_attention_lab.cases import make_inputs
from flash_attention_lab.environment import snapshot
from flash_attention_lab.reference import mathematical_reference
from flash_attention_lab.registry import EXERCISES
from flash_attention_lab.runner import save_artifact
from flash_attention_lab.timing import measure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exercise", choices=[k for k,v in EXERCISES.items() if v["kind"] == "attention"], default="s40_fa4_base")
    p.add_argument("--case", type=int, default=0)
    p.add_argument("--source-commit", required=True, help="exact official source commit used to install")
    args = p.parse_args()
    if not 0 <= args.case < len(EXERCISES[args.exercise]["cases"]):
        p.error("case index is outside the selected exercise contract")
    try:
        from flash_attn.cute import flash_attn_func
    except ImportError as exc:
        print("ENV_UNSUPPORTED: optional official FA4 is not installed; use a separate environment:", exc)
        return 4
    e = EXERCISES[args.exercise]; case = e["cases"][args.case]
    if case["dtype"] == "fp32":
        print("ENV_UNSUPPORTED: official comparison requires a low-precision contract")
        return 4
    inputs,cfg = make_inputs(e,case,"cuda")
    q,k,v = [inputs[n].transpose(1,2).contiguous() for n in ("q","k","v")]
    torch.cuda.synchronize()
    out = flash_attn_func(q,k,v,causal=cfg["causal"],softmax_scale=cfg["scale"])
    ref = mathematical_reference("attention",inputs,cfg)
    # Official default returns O only, so its LSE is not claimed validated here.
    errors = compare(e,{"o":out.transpose(1,2).contiguous()},{"o":ref["o"]},inputs)
    def call(): return flash_attn_func(q,k,v,causal=cfg["causal"],softmax_scale=cfg["scale"])
    def with_conversions():
        iq,ik,iv=[inputs[n].transpose(1,2).contiguous() for n in ("q","k","v")]
        return flash_attn_func(iq,ik,iv,causal=cfg["causal"],softmax_scale=cfg["scale"]).transpose(1,2).contiguous()
    result={"backend":"optional_official_fa4", "source_commit":args.source_commit,
            "package_version":importlib.metadata.version("flash-attn-4"), "case":case,"errors":errors,
            "environment":snapshot(), "official_api_gpu_span":measure(call),
            "api_plus_layout_gpu_span":measure(with_conversions),
            "scope":"official API can allocate output internally; API timing is not a pure preallocated kernel measurement. Second span includes BHND↔BNHD conversions and their allocations; neither includes input generation or cold JIT."}
    print(json.dumps(result,indent=2)); print(save_artifact("official_fa4",result))
    return 0


if __name__ == "__main__": raise SystemExit(main())
