class StudentNotImplemented(RuntimeError):
    pass


class EnvironmentUnsupported(RuntimeError):
    pass


class DependencyMissing(StudentNotImplemented):
    pass


class CorrectnessFailure(AssertionError):
    pass
