"""Full local reports, separate from terminal summaries."""

import json
import time
from .registry import ROOT


def save_artifact(prefix, result):
    directory = ROOT / "artifacts" / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (prefix + "_" + str(time.time_ns()) + ".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return path
