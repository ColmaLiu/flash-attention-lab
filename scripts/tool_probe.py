"""Only scalar infrastructure sentinels; never a student's kernel or acceptance."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flash_attention_lab.runtime import configure_runtime
configure_runtime()

import torch
from flash_attention_lab.build import build_cuda, descriptors


def main():
    from flash_attention_lab.environment import require_environment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    require_environment("cuda")
    output = torch.zeros(1, device="cuda", dtype=torch.int32)
    stream = torch.cuda.current_stream().cuda_stream
    cpp = build_cuda("infra_smoke")
    cuda_launch = lambda: cpp.launch({}, descriptors({"sentinel": output}), {}, {}, stream)
    require_environment("cute")
    from flash_attention_lab.cute_smoke import prepare_sentinel
    launches = [("cuda", cuda_launch), ("cute", prepare_sentinel(output, stream))]
    for name, launch in launches:
        output.zero_()
        torch.cuda.nvtx.range_push("infra_tool_probe")
        launch()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        assert output.item() == 7
        print(name, "infrastructure sentinel OK", flush=True)


if __name__ == "__main__": main()
