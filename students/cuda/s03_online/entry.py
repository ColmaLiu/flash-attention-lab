"""Student-owned host/JIT configuration. See docs/tasks/s03_online.md."""
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
