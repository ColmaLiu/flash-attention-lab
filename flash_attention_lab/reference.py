"""Independent PyTorch mathematical oracle. Never used by student API.

Uses FP64 on the already-quantized inputs. Attention evaluates each row against
all keys; independent rows are grouped to bound temporary memory. No online
softmax, key-tile state, CUDA or DSL exercise implementations are provided.
"""
import torch

# At most 32 MiB per FP64 score tensor. Other temporaries and Q/K/V/O are additional.
ATTENTION_SCORE_ELEMENTS = 4 * 1024 * 1024


def _attention_rows(q, k, v, scale, causal, query_start=0):
    scores = (q @ k.transpose(-1, -2)) * scale
    if causal:
        query = torch.arange(query_start, query_start + q.shape[-2], device=q.device)
        key = torch.arange(k.shape[-2], device=q.device)
        scores.masked_fill_(key[None, :] > query[:, None], -torch.inf)
    maximum = scores.amax(-1, keepdim=True)
    weights = (scores - maximum).exp()
    denominator = weights.sum(-1, keepdim=True)
    return (weights / denominator) @ v, (maximum + denominator.log()).squeeze(-1)


def _attention_reference(inputs, config):
    q, k, v = (inputs[n].double() for n in ("q", "k", "v"))
    b, h, n, _ = q.shape
    if b * h * n * n <= ATTENTION_SCORE_ELEMENTS:
        out, lse = _attention_rows(q, k, v, config["scale"], config["causal"])
    else:
        out = torch.empty_like(q)
        lse = torch.empty((b, h, n), dtype=torch.float64, device=q.device)
        rows = max(1, ATTENTION_SCORE_ELEMENTS // n)
        for batch in range(b):
            for head in range(h):
                for start in range(0, n, rows):
                    stop = min(start + rows, n)
                    out[batch, head, start:stop], lse[batch, head, start:stop] = _attention_rows(
                        q[batch, head, start:stop], k[batch, head], v[batch, head],
                        config["scale"], config["causal"], start)
    return {"o": out, "lse": lse}


def mathematical_reference(kind, inputs, config):
    x = inputs.get("x")
    if kind in ("softmax", "online"):
        z = x.double()
        m = z.amax(-1)
        w = (z - m[:, None]).exp()
        result = {"y": w / w.sum(-1, keepdim=True)}
        if kind == "online":
            result.update(m=m, l=w.sum(-1))
        return result
    if kind == "reduce":
        return {"maximum": x.double().amax(-1), "sum": x.double().sum(-1)}
    if kind == "attention":
        return _attention_reference(inputs, config)
    if kind == "gemm":
        return {"c": inputs["a"].double() @ inputs["b"].double()}
    if kind == "transpose":
        return {"y": x.double().T.contiguous()}
    if kind == "copy":
        return {"y": x.double().clone()}
    if kind == "affine":
        return {"y": x.double() * 2 + 1}
    if kind == "exp2":
        return {"y": torch.exp2(x.double())}
    raise ValueError(kind)
