"""A scalar sentinel proves JIT and stream plumbing, contains no exercise algorithm."""
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda


@cute.kernel
def infrastructure_sentinel(out: cute.Tensor):
    out[0] = 7


@cute.jit
def invoke_sentinel(out: cute.Tensor, stream: cuda.CUstream):
    infrastructure_sentinel(out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)


def prepare_sentinel(out, stream):
    tensor = from_dlpack(out)
    cu_stream = cuda.CUstream(stream)
    compiled = cute.compile(invoke_sentinel, tensor, cu_stream)
    return lambda: compiled(tensor, cu_stream)
