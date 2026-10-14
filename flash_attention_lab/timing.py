import statistics
import time
import torch


def measure(launch, warmup=5, repeats=7, iterations=10):
    if min(warmup, repeats, iterations) < 1: raise ValueError("timing counts must be positive")
    for _ in range(warmup): launch()
    torch.cuda.synchronize()
    samples, walls = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(iterations): launch()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - wall) * 1e3 / iterations)
        samples.append(start.elapsed_time(end) / iterations)
    median = statistics.median(samples)
    return {"gpu_stream_ms_median": median, "gpu_stream_ms_min": min(samples),
            "gpu_stream_ms_max": max(samples), "gpu_stream_ms_stdev": statistics.pstdev(samples),
            "gpu_stream_ms_samples": samples, "host_batch_ms_median": statistics.median(walls),
            "warmup": warmup, "repeats": repeats, "iterations": iterations,
            "scope": "preallocated launches on current stream; excludes JIT, inputs, allocation, layout conversion; includes device stream gaps caused by host dispatch; host_batch includes event recording and final wait; not end-to-end latency"}
