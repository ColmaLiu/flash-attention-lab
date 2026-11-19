"""Owning, zero-copy tensor arguments prepared on the launcher's explicit stream."""

import tvm_ffi


def tensor_arguments(tensors, stream):
    converted = {}
    for name, tensor in tensors.items():
        device = tvm_ffi.device(str(tensor.device))
        with tvm_ffi.use_raw_stream(device, stream):
            converted[name] = tvm_ffi.from_dlpack(tensor, require_contiguous=True)
    return tvm_ffi.convert(converted)


def launch_arguments(inputs, outputs, workspace, config, stream):
    """Keep conversion and DLPack ownership outside repeated launch/capture."""
    return (
        tensor_arguments(inputs, stream),
        tensor_arguments(outputs, stream),
        tensor_arguments(workspace, stream),
        tvm_ffi.convert({name: float(value) for name, value in config.items()}),
        stream,
    )
