# s42_numerics: FA4 source experiment

Two query stages, emulated exponential, conditional rescaling threshold 8, nonpersistent scheduling.

`entry.py` selects the experiment configuration and calls the compile helper in `s40_attention/fa4_reference.py`. The complete CuTe DSL production kernel is [flash_fwd_sm100.py](../../third_party/flash-attention/flash_attn/cute/flash_fwd_sm100.py), with softmax, MMA, copy, pipeline and scheduling helpers alongside it. The FlashAttention submodule is pinned by the parent repository's gitlink.

[The source map](../../docs/reference-implementations.md) connects each experiment to its modules. The [rescale-threshold patch](../../patches/flash-attention/rescale-threshold.patch) is applied to a generated compilation copy under `.cache/sources/`; the upstream submodule remains clean. The student's CUDA C++/PTX implementation has its own entry.

Prepare wraps preallocated BNHD input/output and BHN LSE tensors and compiles before measurement. Launches use the supplied stream. Query stages, exponent strategy, rescale threshold and scheduling are recorded in CONFIG. Change the two numerical controls separately when attributing their effects. A single query stage still uses asynchronous engines internally.

```bash
bash scripts/bootstrap.sh
./lab check s42_numerics --solution --timeout 600
./lab solution s42_numerics --case 0 --timeout 600
```

Validation and measurements are tied to source fingerprints in ignored artifacts.
