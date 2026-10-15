"""Student-owned host/JIT configuration. See docs/tasks/s31_port.md."""
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
# inputs["x"]: [4,128], torch.float32; input X; read-only.
# outputs["y"]: [4,128], torch.float32; row-wise softmax probabilities.
# config contains scale=1.0 and causal=False; neither affects this exercise.
# workspace: no fixed keys; defined by workspace_spec; allocated but uninitialized.
# Outputs are preallocated, not initialized; input tensors must not be modified.
# Full contract and supported shapes: docs/tasks/s31_port.md.
def prepare(inputs, outputs, workspace, config, stream):
    # STUDENT_TODO[compile-and-launch]: implement your CuTe kernel and its
    # layout/mapping/synchronization. Compile here; return a zero-argument
    # launcher. No allocation or compilation is allowed inside that launcher.
    from flash_attention_lab.errors import StudentNotImplemented
    raise StudentNotImplemented("s31_port: CuTe kernel and launch are not implemented")
