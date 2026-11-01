"""Vendored FA4 answer; explicit configuration, no student fallback."""
import importlib.util
from flash_attention_lab.registry import ROOT
IMPLEMENTED = True
BACKEND = "python"
CONFIG = {'q_stage': 2, 'emulated_exp': True, 'rescale_threshold': 8.0, 'persistent': False}
RUNTIME_REQUIRES = []
def workspace_spec(inputs, config): return {}
def prepare(inputs, outputs, workspace, config, stream):
    import sys
    name = "fa_lab_full_fa4_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "solutions/s40_attention/fa4_reference.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].prepare_fa4(inputs, outputs, config, stream,
        q_stage=int(config["q_stage"]), emulated_exp=bool(config["emulated_exp"]),
        rescale_threshold=float(config["rescale_threshold"]), persistent=bool(config["persistent"]))
