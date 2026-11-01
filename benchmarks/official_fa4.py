"""Optional official FA4 API comparison; no installation side effects.

Inputs are canonical contiguous BNHD, so no input layout conversion is needed.
This optional public-API measurement validates O only and is explicitly distinct
from the main cuDNN O+LSE benchmark. Source provenance is supplied by the caller.
"""
import argparse
import importlib.metadata
import json

from flash_attention_lab.runtime import configure_runtime
configure_runtime()
import torch
from flash_attention_lab.accuracy import compare
from flash_attention_lab.cases import make_inputs
from flash_attention_lab.environment import snapshot
from flash_attention_lab.reference import mathematical_reference
from flash_attention_lab.registry import EXERCISES
from flash_attention_lab.reporting import save_artifact
from flash_attention_lab.timing import measure, measure_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", choices=[k for k, v in EXERCISES.items() if v["kind"] == "attention"], default="s40_attention")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--source-commit", required=True, help="exact official source commit used to install (caller-supplied provenance)")
    args = parser.parse_args()
    e = EXERCISES[args.exercise]
    if not 0 <= args.case < len(e["cases"]):
        parser.error("case index is outside the selected exercise contract")
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        parser.error("--source-commit must be a full lowercase 40-character Git commit")
    try:
        from flash_attn.cute import flash_attn_func
    except ImportError as exc:
        print("ENV_UNSUPPORTED: optional official FA4 is not installed; use a separately provisioned environment:", exc)
        return 4
    case = e["cases"][args.case]
    if case["dtype"] == "fp32":
        print("ENV_UNSUPPORTED: official comparison requires a low-precision contract")
        return 4
    inputs, cfg = make_inputs(e, case, "cuda")
    saved = {name: tensor.clone() for name, tensor in inputs.items()}
    oracle = mathematical_reference("attention", inputs, cfg)
    output = {}

    def call():
        result = flash_attn_func(inputs["q"], inputs["k"], inputs["v"], causal=cfg["causal"], softmax_scale=cfg["scale"])
        output["o"] = result[0] if isinstance(result, tuple) else result

    def validate():
        errors = compare(e, output, {"o": oracle["o"]}, inputs)
        for name, tensor in inputs.items():
            if not torch.equal(tensor, saved[name]):
                raise RuntimeError("official API modified input " + name)
        return errors

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        stream.synchronize()
        errors = validate()
        eager = measure(call)
        validate()
        graph = measure_graph(call)
        validate()
    torch.cuda.current_stream().wait_stream(stream)
    try:
        version = importlib.metadata.version("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown distribution; use source_commit"
    result = {
        "backend": "optional_official_fa4", "source_commit": args.source_commit,
        "provenance": "caller supplied; not independently verified against installed source",
        "package_version": version, "case": case, "errors": errors,
        "environment": snapshot(), "official_api_eager_span": eager,
        "official_api_graph_span": graph,
        "scope": "canonical BNHD API; no input layout conversion; O validated, LSE not requested/validated; "
                 "eager includes internal allocation, graph replay excludes allocation; "
                 "O-only timing is not the full O+LSE comparison provided by lab bench",
    }
    print(json.dumps(result, indent=2))
    print(save_artifact("official_fa4", result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
