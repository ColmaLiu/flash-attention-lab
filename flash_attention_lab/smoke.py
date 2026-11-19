import torch
from .build import build_cuda
from .ffi import launch_arguments
from .cases import make_inputs, output_spec, validate_inputs
from .environment import require_environment
from .reference import mathematical_reference
from .registry import CATALOG, load_student
from .timing import measure


def infrastructure_smoke(stage=None):
    exercises = [e for e in CATALOG if stage is None or e["stage"] == stage]
    backends = {e["backend"] for e in exercises}
    for backend in backends:
        require_environment(backend)
    torch.set_num_threads(1)
    case_count = 0
    for e in exercises:
        module = load_student(e)
        assert isinstance(module.IMPLEMENTED, bool)
        assert callable(module.workspace_spec)
        # Bounded infrastructure probe; full curriculum correctness is a separate command.
        for case in e["cases"][:1]:
            inputs, config = make_inputs(e, case, "cuda")
            validate_inputs(e, inputs, config)
            ref = mathematical_reference(e["kind"], inputs, config)
            expected = output_spec(e, inputs)
            assert set(ref) == set(expected)
            for name, tensor in ref.items():
                assert tuple(tensor.shape) == tuple(expected[name][0])
                assert torch.isfinite(tensor).all(), (e["id"], case, name)
            case_count += 1
    result = {
        "state": "INFRASTRUCTURE_OK",
        "modules_imported": len(exercises),
        "input_reference_cases": case_count,
        "student_correctness": "NOT_TESTED",
    }
    out = torch.empty(1, dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    if "cuda" in backends:
        lib = build_cuda("infra_smoke")
        args = launch_arguments({}, {"sentinel": out}, {}, {}, stream)

        def launch():
            return lib.launch(*args)

        out.zero_()
        launch()
        torch.cuda.synchronize()
        assert out.item() == 7
        result["cuda_sentinel"] = measure(launch, 2, 3, 5)
        # One empty binding per stage verifies the host ABI without a long compile sweep.
        bindings, compiled_stages = [], set()
        for e in exercises:
            if (
                e["backend"] != "cuda"
                or load_student(e).IMPLEMENTED
                or e["stage"] in compiled_stages
            ):
                continue
            compiled_stages.add(e["stage"])
            module = build_cuda(e["id"])
            try:
                module.launch({}, {}, {}, {}, stream)
            except RuntimeError as exc:
                assert "STUDENT_TODO" in str(exc), str(exc)
            else:
                raise AssertionError(f"{e['id']} stub returned without error")
            bindings.append(e["id"])
        result["cuda_stub_bindings_compiled"] = bindings
    return result
