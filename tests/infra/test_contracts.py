import json
import subprocess
import sys
import torch
import pytest
from flash_attention_lab.registry import CATALOG, EXERCISES, ROOT, load_student
from flash_attention_lab.cases import make_inputs, output_spec, validate_inputs
from flash_attention_lab.reference import mathematical_reference
from flash_attention_lab.api import prepare, require_implemented
from flash_attention_lab.accuracy import compare
from flash_attention_lab.errors import CorrectnessFailure, DependencyMissing, StudentNotImplemented


@pytest.mark.parametrize("e", CATALOG, ids=lambda e:e["id"])
def test_exercise_contract_and_references(e):
    assert (ROOT / "docs/tasks" / (e["id"] + ".md")).is_file()
    assert (ROOT / "experiments" / (e["id"] + ".md")).is_file()
    assert all(dep in EXERCISES for dep in e["prerequisites"])
    student = load_student(e)
    assert callable(student.workspace_spec)
    if e["backend"] == "cute": assert callable(student.prepare)
    for case in e["cases"]:
        inputs,cfg = make_inputs(e,case)
        ref = mathematical_reference(e["kind"],inputs,cfg)
        spec = output_spec(e,inputs)
        assert ref.keys() == spec.keys()
        for n,t in ref.items():
            assert tuple(t.shape) == tuple(spec[n][0])
            assert t.dtype == torch.float64 and torch.isfinite(t).all()


def test_reference_has_analytic_attention_invariants():
    e=EXERCISES["s11_fa1_edges"]
    q=torch.zeros(1,1,3,2)
    v=torch.tensor([[[[1.,2.],[3.,4.],[5.,6.]]]])
    tensors={"q":q,"k":q.clone(),"v":v}
    full=mathematical_reference("attention",tensors,{"scale":0.7,"causal":False})
    causal=mathematical_reference("attention",tensors,{"scale":0.7,"causal":True})
    assert torch.equal(full["o"],torch.tensor([[[[3.,4.],[3.,4.],[3.,4.]]]],dtype=torch.float64))
    assert torch.equal(causal["o"],torch.tensor([[[[1.,2.],[2.,3.],[3.,4.]]]],dtype=torch.float64))
    torch.testing.assert_close(causal["lse"],torch.tensor([[[0.,2.,3.]]],dtype=torch.float64).clamp_min(1).log())


def test_error_budget_rejects_near_zero_corruption_and_nan():
    e=EXERCISES["s00_softmax"]
    inputs={"x":torch.zeros(4,128)}
    ref=mathematical_reference("softmax",inputs,{})
    for value in (0.1,float("nan")):
        out={"y":ref["y"].float().clone()};out["y"][0,0]=value
        with pytest.raises(CorrectnessFailure): compare(e,out,ref,inputs)


def test_missing_dependency_is_named(monkeypatch):
    from types import SimpleNamespace
    import flash_attention_lab.api as api
    def fake(e):
        return SimpleNamespace(IMPLEMENTED=e['id']=="s31_port",RUNTIME_REQUIRES=["s00_softmax"] if e['id']=="s31_port" else [])
    monkeypatch.setattr(api,"load_student",fake)
    with pytest.raises(DependencyMissing,match="s00_softmax"):
        require_implemented(EXERCISES["s31_port"])


def test_unimplemented_never_reaches_allocation_or_reference(monkeypatch):
    import flash_attention_lab.api as api
    from types import SimpleNamespace
    monkeypatch.setattr(api,"load_student",lambda e:SimpleNamespace(IMPLEMENTED=False))
    with pytest.raises(StudentNotImplemented,match="s00_softmax"):
        prepare("s00_softmax",{})


def test_contract_rejects_cpu_input_before_launch():
    e=EXERCISES["s00_softmax"]
    inputs,cfg=make_inputs(e,e['cases'][0])
    with pytest.raises(ValueError,match="contiguous CUDA"):
        validate_inputs(e,inputs,cfg)


def test_cli_rejects_invalid_case_and_bench_counts():
    from flash_attention_lab.cli import main, parser
    assert main(["check","s00_softmax","--case","9999"]) == 6
    args=parser().parse_args(["bench","s40_fa4_base","--case","0","--warmup","2","--repeats","3","--iterations","4"])
    assert (args.warmup,args.repeats,args.iterations)==(2,3,4)
    with pytest.raises(ValueError):
        parser().parse_args(["bench","s40_fa4_base","--warmup","0"])


def test_timeout_only_terminates_its_child():
    from flash_attention_lab.cli import bounded
    assert bounded([sys.executable,"-c","import time; time.sleep(2)"],0.05) == 5


def test_positive_error_check_does_not_claim_a_kernel_pass():
    e=EXERCISES["s00_softmax"]
    inputs,cfg=make_inputs(e,e["cases"][0])
    expected=mathematical_reference(e["kind"],inputs,cfg)
    rounded={name:t.float() for name,t in expected.items()}
    result=compare(e,rounded,expected,inputs)
    assert result["y"]["bad_elements"] == 0


def test_curriculum_is_acyclic():
    seen=set()
    for e in CATALOG:
        assert set(e["prerequisites"]) <= seen
        seen.add(e["id"])
    assert {e["stage"] for e in CATALOG} == set(range(5))
