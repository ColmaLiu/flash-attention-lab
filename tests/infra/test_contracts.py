import json
import math
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
def test_exercise_metadata_and_small_references(e):
    assert (ROOT / "docs/tasks" / (e["id"] + ".md")).is_file()
    assert (ROOT / "experiments/README.md").is_file()
    assert all(dep in EXERCISES for dep in e["prerequisites"])
    student = load_student(e)
    assert callable(student.workspace_spec)
    if e["backend"] == "cute": assert callable(student.prepare)
    for case in e["cases"]:
        shape = case["shape"]
        assert all(isinstance(n, int) and n > 0 for n in shape)
        assert case["dtype"] in ("fp32", "fp16", "bf16")
        assert isinstance(case["causal"], bool)
        assert not case["causal"] or e["kind"] == "attention"
        # CPU unit tests cover oracle behavior on small tensors. GPU smoke generates
        # inputs and FP64 references for EVERY declared case, including large shapes.
        work = math.prod(shape)
        if e["kind"] == "attention": work = shape[0] * shape[2] * shape[1] ** 2
        elif e["kind"] == "gemm": work = math.prod(shape)
        limit = 16 * 1024**2 if e["kind"] == "gemm" else 4 * 1024**2 if e["kind"] == "attention" else 1024**2
        if work > limit: continue
        inputs,cfg = make_inputs(e,case)
        ref = mathematical_reference(e["kind"],inputs,cfg)
        spec = output_spec(e,inputs)
        assert ref.keys() == spec.keys()
        for n,t in ref.items():
            assert tuple(t.shape) == tuple(spec[n][0])
            assert t.dtype == torch.float64 and torch.isfinite(t).all()


def test_reference_has_analytic_attention_invariants():
    e=EXERCISES["s11_materialized"]
    q=torch.zeros(1,3,1,2)
    v=torch.tensor([[[[1.,2.]],[[3.,4.]],[[5.,6.]]]])
    tensors={"q":q,"k":q.clone(),"v":v}
    full=mathematical_reference("attention",tensors,{"scale":0.7,"causal":False})
    causal=mathematical_reference("attention",tensors,{"scale":0.7,"causal":True})
    assert torch.equal(full["o"],torch.tensor([[[[3.,4.]],[[3.,4.]],[[3.,4.]]]],dtype=torch.float64))
    assert torch.equal(causal["o"],torch.tensor([[[[1.,2.]],[[2.,3.]],[[3.,4.]]]],dtype=torch.float64))
    torch.testing.assert_close(causal["lse"],torch.tensor([[[0.,2.,3.]]],dtype=torch.float64).clamp_min(1).log())


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bounded_reference_matches_dense_definition_and_respects_budget(monkeypatch, causal, dtype):
    import flash_attention_lab.reference as reference
    generator = torch.Generator().manual_seed(123)
    inputs = {name: torch.randn(2, 17, 3, 8, generator=generator).to(dtype) for name in ("q", "k", "v")}
    q, k, v = (inputs[name].permute(0,2,1,3).double() for name in ("q", "k", "v"))
    scores = (q @ k.transpose(-1, -2)) * 0.37
    if causal: scores.masked_fill_(torch.ones(17, 17, dtype=torch.bool).triu(1), -torch.inf)
    expected = {"o": (scores.softmax(-1) @ v).permute(0,2,1,3), "lse": scores.logsumexp(-1)}
    calls = []
    original = reference._attention_rows
    def capture(q, k, *args):
        calls.append((q.shape[-2], k.shape[-2]))
        return original(q, k, *args)
    monkeypatch.setattr(reference, "ATTENTION_SCORE_ELEMENTS", 4 * 17)
    monkeypatch.setattr(reference, "_attention_rows", capture)
    actual = reference.mathematical_reference("attention", inputs, {"scale": 0.37, "causal": causal})
    for key in expected: torch.testing.assert_close(actual[key], expected[key], atol=1e-12, rtol=1e-12)
    assert sum(rows for rows, _ in calls) == 2 * 3 * 17
    assert max(rows * keys for rows, keys in calls) <= 4 * 17


def test_large_cases_are_in_default_acceptance(monkeypatch):
    from flash_attention_lab import runner
    e = EXERCISES["s20_mma"]
    monkeypatch.setattr(runner, "require_implemented", lambda e: None)
    monkeypatch.setattr(runner, "require_environment", lambda backend: None)
    seen = []
    def validate(e, case, *, nondefault):
        seen.append((tuple(case["shape"]), nondefault))
        return {}
    monkeypatch.setattr(runner, "validate_case", validate)
    result = runner.check(e)
    assert result["full_contract"] and result["case_count"] == len(e["cases"])
    assert len(seen) == 2 * len(e["cases"])
    assert ((2048, 128, 2048), True) in seen


def test_error_budget_rejects_near_zero_corruption_and_nan():
    e=EXERCISES["s01_softmax"]
    inputs={"x":torch.zeros(4,128)}
    ref=mathematical_reference("softmax",inputs,{})
    for value in (0.1,float("nan")):
        out={"y":ref["y"].float().clone()};out["y"][0,0]=value
        with pytest.raises(CorrectnessFailure): compare(e,out,ref,inputs)


def test_missing_dependency_is_named(monkeypatch):
    from types import SimpleNamespace
    import flash_attention_lab.api as api
    def fake(e):
        return SimpleNamespace(IMPLEMENTED=e['id']=="s31_tcgen",RUNTIME_REQUIRES=["s01_softmax"] if e['id']=="s31_tcgen" else [])
    monkeypatch.setattr(api,"load_student",fake)
    with pytest.raises(DependencyMissing,match="s01_softmax"):
        require_implemented(EXERCISES["s31_tcgen"])


def test_unimplemented_never_reaches_allocation_or_reference(monkeypatch):
    import flash_attention_lab.api as api
    from types import SimpleNamespace
    monkeypatch.setattr(api,"load_student",lambda e:SimpleNamespace(IMPLEMENTED=False))
    with pytest.raises(StudentNotImplemented,match="s01_softmax"):
        prepare("s01_softmax",{})


def test_contract_rejects_cpu_input_before_launch():
    e=EXERCISES["s01_softmax"]
    inputs,cfg=make_inputs(e,e['cases'][0])
    with pytest.raises(ValueError,match="contiguous CUDA"):
        validate_inputs(e,inputs,cfg)


def test_cli_rejects_invalid_case_and_bench_counts():
    from flash_attention_lab.cli import main, parser
    assert main(["check","s01_softmax","--case","9999"]) == 6
    args=parser().parse_args(["bench","s40_attention","--case","0","--warmup","2","--repeats","3","--iterations","4"])
    assert (args.warmup,args.repeats,args.iterations)==(2,3,4)
    with pytest.raises(ValueError):
        parser().parse_args(["bench","s40_attention","--warmup","0"])


def test_timeout_only_terminates_its_child():
    from flash_attention_lab.cli import bounded
    assert bounded([sys.executable,"-c","import time; time.sleep(2)"],0.05) == 5


def test_positive_error_check_does_not_claim_a_kernel_pass():
    e=EXERCISES["s01_softmax"]
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


def test_bnHD_uniform_causal_has_nonzero_prefix_means():
    e = EXERCISES['s12_online']
    case = next(c for c in e['cases'] if c['id'] == 'causal_uniform')
    inputs, config = make_inputs(e, case)
    ref = mathematical_reference('attention', inputs, config)
    n = inputs['v'].shape[1]
    prefix = inputs['v'].double().cumsum(1) / torch.arange(1, n+1).reshape(1,n,1,1)
    torch.testing.assert_close(ref['o'], prefix)
    assert torch.count_nonzero(ref['o']) > 0
    torch.testing.assert_close(ref['lse'], torch.arange(1, n+1).double().log().expand_as(ref['lse']))


def test_domain_accepts_unlisted_shapes_and_rejects_alignment_dtype_and_mask():
    from types import SimpleNamespace
    e = EXERCISES['s31_tcgen']
    def inputs(m,n,k,dtype=torch.bfloat16):
        def tensor(shape):
            return SimpleNamespace(shape=shape, ndim=2, dtype=dtype, device='cuda:0',
                                   is_cuda=True, is_contiguous=lambda:True, requires_grad=False, data_ptr=lambda:256)
        return {'a':tensor((m,k)), 'b':tensor((k,n))}
    validate_inputs(e, inputs(256,24,48), {'scale':1.,'causal':False})
    for scale in (1e39, 1e-50, float('inf'), float('nan'), 0., -1., True, '1'):
        with pytest.raises(ValueError, match='scale'):
            validate_inputs(e,inputs(256,24,48),{'scale':scale,'causal':False})
    misaligned = inputs(256,24,48)
    misaligned['a'].data_ptr = lambda:258
    with pytest.raises(ValueError, match='alignment'):
        validate_inputs(e,misaligned,{'scale':1.,'causal':False})
    for tensors, causal in [(inputs(65,24,48),False),(inputs(256,24,48,torch.float32),False),
                            (inputs(256,24,48),True)]:
        with pytest.raises(ValueError, match='domain'):
            validate_inputs(e,tensors,{'scale':1.,'causal':causal})


def test_public_attention_api_keeps_student_path_and_official_layout(monkeypatch):
    import flash_attention_lab as lab
    from flash_attention_lab import api
    observed = []
    def run(exercise, inputs, config):
        observed.append((exercise,inputs,config))
        return {'o':'output','lse':'statistics'}
    monkeypatch.setattr(api,'run',run)
    assert lab.flash_attn_func('q','k','v', exercise='s22_fa2', causal=True, return_lse=True) == ('output','statistics')
    assert observed == [('s22_fa2',{'q':'q','k':'k','v':'v'},{'causal':True})]


def test_cases_have_fault_rationale_and_representative_performance_work():
    for e in CATALOG:
        assert len({c['id'] for c in e['cases']}) == len(e['cases'])
        assert all(c['purpose'] and c['suite'] in ('debug','edge','stress','perf') for c in e['cases'])
        assert sum(c['suite']=='perf' for c in e['cases']) >= 2
        assert e['cases'][0]['suite'] == 'debug'
