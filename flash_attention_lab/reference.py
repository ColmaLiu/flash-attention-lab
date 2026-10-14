"""Independent full-matrix PyTorch mathematical oracle. Never used by student API.

Uses FP64 on the already-quantized inputs. No online/tiled/device algorithm answers.
"""
import torch


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
        q, k, v = (inputs[n].double() for n in ("q", "k", "v"))
        s = (q @ k.transpose(-1, -2)) * config["scale"]
        if config["causal"]:
            n = q.shape[-2]
            mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
            s = s.masked_fill(mask, -torch.inf)
        m = s.amax(-1, keepdim=True)
        w = (s - m).exp()
        den = w.sum(-1, keepdim=True)
        return {"o": (w / den) @ v, "lse": (m + den.log()).squeeze(-1)}
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
