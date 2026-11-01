# s40_attention: complete FA4 source experiment

One query stage, hardware exponential, exact rescaling, nonpersistent scheduling.

`entry.py` calls the direct compile helper in `s40_attention/fa4_reference.py`. The **complete kernel** is [flash_fwd_sm100.py](../../references/flash-attention/flash_attn/cute/flash_fwd_sm100.py); its full softmax, MMA, copy, pipeline and scheduling helpers are included beside it. This is a pinned production answer with explicit experimental configurations, not a separately authored CUDA FA4 implementation. The student's CUDA/PTX entry remains independent.

Read the [source map and documented rescale patch](../../references/README.md). Prepare wraps preallocated BNHD input/output and BHN LSE tensors and compiles before measurement; launches use the supplied stream. The two numerical controls in s42 can be changed separately in an independent experiment; do not attribute their combined effect to just one mechanism. q_stage=1 still uses asynchronous engines internally.

```bash
./lab python -m pip install -e '.[fa4]'
./lab check s40_attention --solution --timeout 600
./lab solution s40_attention --case 0 --timeout 600
```

Correctness and measured performance belong to current source fingerprints in ignored artifacts, not to this public documentation.
