"""Student-owned host/JIT configuration. See docs/tasks/s34_warp_specialize.md."""
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


def prepare(inputs, outputs, workspace, config, stream):
    # STUDENT_TODO[compile-and-launch]: implement your CuTe kernel and its
    # layout/mapping/synchronization. Compile here; return a zero-argument
    # launcher. No allocation or compilation is allowed inside that launcher.
    from flash_attention_lab.errors import StudentNotImplemented
    raise StudentNotImplemented("s34_warp_specialize: CuTe kernel and launch are not implemented")
