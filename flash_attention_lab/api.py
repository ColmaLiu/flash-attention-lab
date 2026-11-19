"""Independent student and solution paths; libraries never replace student kernels."""

from dataclasses import dataclass
import torch
from .cases import allocate_outputs, validate_inputs
from .errors import DependencyMissing, StudentNotImplemented
from .registry import EXERCISES, load_student


def require_implemented(e, stack=()):
    if e["id"] in stack:
        raise DependencyMissing(
            "student dependency cycle: " + " -> ".join((*stack, e["id"]))
        )
    module = load_student(e)
    if not module.IMPLEMENTED:
        raise StudentNotImplemented(
            f"{e['id']}: STUDENT_TODO in students/{e['backend']}/{e['id']}; set IMPLEMENTED=True only after implementing"
        )
    # Curriculum prerequisites are separately tracked; only declared code reuse is a runtime dependency.
    for dep in module.RUNTIME_REQUIRES:
        if dep not in EXERCISES:
            raise DependencyMissing(f"{e['id']}: unknown module {dep}")
        try:
            require_implemented(EXERCISES[dep], (*stack, e["id"]))
        except StudentNotImplemented as exc:
            raise DependencyMissing(
                f"{e['id']}: required student module {dep}: {exc}"
            ) from exc
    return module


@dataclass
class Prepared:
    launch: object
    outputs: dict
    workspace: dict
    inputs: dict
    extra_workspace_bytes: int = 0


def prepare(exercise_id, inputs, config=None, *, solution=False):
    e = EXERCISES[exercise_id]
    module = load_student(e, solution=True) if solution else require_implemented(e)
    cfg = {"causal": False, "scale": 1.0, **module.CONFIG, **(config or {})}
    if (
        e["kind"] == "attention"
        and "scale" not in module.CONFIG
        and (not config or "scale" not in config)
    ):
        cfg["scale"] = inputs["q"].shape[-1] ** -0.5
    validate_inputs(e, inputs, cfg)
    t = next(iter(inputs.values()))
    with torch.cuda.device(t.device):
        outputs = allocate_outputs(e, inputs)
        spec = module.workspace_spec(inputs, cfg)
        workspace = {
            n: torch.empty(shape, dtype=dtype, device=t.device)
            for n, (shape, dtype) in spec.items()
        }
        stream = torch.cuda.current_stream(t.device).cuda_stream
        if e["backend"] == "cuda" and getattr(module, "BACKEND", None) != "python":
            from .build import build_cuda
            from .ffi import launch_arguments

            lib = build_cuda(exercise_id, solution=solution)
            args = launch_arguments(inputs, outputs, workspace, cfg, stream)

            def launch():
                return lib.launch(*args)
        else:
            launch = module.prepare(inputs, outputs, workspace, cfg, stream)
        if not callable(launch):
            raise TypeError(
                "student prepare must return a zero-argument launch callable"
            )
    return Prepared(
        launch, outputs, workspace, inputs, getattr(launch, "extra_workspace_bytes", 0)
    )


def run(exercise_id, inputs, config=None):
    """Convenience full invocation includes allocation/setup; benchmark uses prepare separately."""
    plan = prepare(exercise_id, inputs, config)
    plan.launch()
    return plan.outputs
