"""Student-owned host/JIT configuration. See docs/tasks/s44_fa4_rescale.md."""
IMPLEMENTED = False
# STUDENT_TODO[configuration]: choose your numeric launch/tile/stage options.
CONFIG = {}
# Learning prerequisites are in curriculum.json. List actual code reuse here;
# unimplemented reused modules produce a named dependency error.
RUNTIME_REQUIRES = []


def workspace_spec(inputs, config):
    # STUDENT_TODO[workspace-design]: declare {name: (shape, torch.dtype)}
    # only if your design needs global scratch. No launch/layout is preselected.
    return {}


# ENTRY CONTRACT (case-sensitive keys; contiguous tensors on one GPU):
# inputs["q"]: [B,H,N,D], torch.bfloat16; input Q; same shape/dtype/device, read-only.
# inputs["k"]: [B,H,N,D], torch.bfloat16; input K; same shape/dtype/device, read-only.
# inputs["v"]: [B,H,N,D], torch.bfloat16; input V; same shape/dtype/device, read-only.
# outputs["o"]: [B,H,N,D], torch.bfloat16; attention output O; input dtype.
# outputs["lse"]: [B,H,N], torch.float32; natural-log logsumexp of scaled, masked scores.
# config["scale"]: positive scalar (default 1/sqrt(D)); config["causal"]: bool.
# workspace: no fixed keys; defined by workspace_spec; allocated but uninitialized.
# Outputs are preallocated, not initialized; input tensors must not be modified.
# Full contract and supported shapes: docs/tasks/s44_fa4_rescale.md.
def prepare(inputs, outputs, workspace, config, stream):
    # STUDENT_TODO[compile-and-launch]: implement your CuTe kernel and its
    # layout/mapping/synchronization. Compile here; return a zero-argument
    # launcher. No allocation or compilation is allowed inside that launcher.
    from flash_attention_lab.errors import StudentNotImplemented
    raise StudentNotImplemented("s44_fa4_rescale: CuTe kernel and launch are not implemented")
