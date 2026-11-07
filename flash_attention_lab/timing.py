import statistics
import time
import torch


def measure(launch, warmup=5, repeats=7, iterations=10):
    if min(warmup, repeats, iterations) < 1:
        raise ValueError("timing counts must be positive")
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples, walls = [], []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        wall = time.perf_counter()
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - wall) * 1e3 / iterations)
        samples.append(start.elapsed_time(end) / iterations)
    median = statistics.median(samples)
    return {
        "gpu_stream_ms_median": median,
        "gpu_stream_ms_min": min(samples),
        "gpu_stream_ms_max": max(samples),
        "gpu_stream_ms_stdev": statistics.pstdev(samples),
        "gpu_stream_ms_samples": samples,
        "host_batch_ms_median": statistics.median(walls),
        "warmup": warmup,
        "repeats": repeats,
        "iterations": iterations,
        "scope": "eager current-stream span; excludes preparation/JIT/input generation; includes work and any allocation inside launch, plus device stream gaps caused by host dispatch; host_batch includes event recording and final wait; not end-to-end latency",
    }


def measure_graph(launch, warmup=5, repeats=7, iterations=10):
    """Capture a batch on the *current nondefault stream*, then time replay.

    The launcher must be prepared on that stream: CUDA bindings may retain its
    raw handle. Both student and reference use this exact measurement path.
    Allocations performed during capture are retained by the graph's pool and
    are absent from replay. There is no eager fallback when capture fails.
    """
    if min(warmup, repeats, iterations) < 1:
        raise ValueError("timing counts must be positive")
    stream = torch.cuda.current_stream()
    if stream == torch.cuda.default_stream():
        raise ValueError(
            "graph timing requires preparation and execution on a nondefault stream"
        )
    for _ in range(warmup):
        launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            launch()
    timing = measure(graph.replay, warmup=1, repeats=repeats, iterations=1)
    for key in (
        "gpu_stream_ms_median",
        "gpu_stream_ms_min",
        "gpu_stream_ms_max",
        "gpu_stream_ms_stdev",
        "host_batch_ms_median",
    ):
        timing[key] /= iterations
    timing["gpu_stream_ms_samples"] = [
        value / iterations for value in timing["gpu_stream_ms_samples"]
    ]
    timing.update(
        warmup=warmup,
        iterations=iterations,
        mode="cuda_graph",
        scope="per-operation CUDA graph replay span; identical batch size for student and reference; "
        "excludes compilation, preparation, capture, input generation and allocation; "
        "includes all operations inside launch including output copies; "
        "host_batch is replay/record/wait cost amortized over the captured batch, not eager API latency",
    )
    return timing


def measure_graph_group(launches, warmup=5, repeats=7, iterations=10):
    """Capture equal batches, then rotate every candidate through sampling positions."""
    launches = tuple(launches)
    if not launches:
        raise ValueError("graph timing requires at least one candidate")
    if min(warmup, repeats, iterations) < 1:
        raise ValueError("timing counts must be positive")
    stream = torch.cuda.current_stream()
    if stream == torch.cuda.default_stream():
        raise ValueError("graph timing requires a nondefault stream")
    graphs = []
    for launch in launches:
        for _ in range(warmup):
            launch()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iterations):
                launch()
        graph.replay()
        stream.synchronize()
        graphs.append(graph)
    samples = [[] for _ in launches]
    walls = [[] for _ in launches]
    for repeat in range(repeats):
        for offset in range(len(launches)):
            index = (repeat + offset) % len(launches)
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            wall = time.perf_counter()
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            walls[index].append((time.perf_counter() - wall) * 1e3 / iterations)
            samples[index].append(start.elapsed_time(end) / iterations)
    results = []
    for gpu, host in zip(samples, walls):
        results.append(
            {
                "gpu_stream_ms_median": statistics.median(gpu),
                "gpu_stream_ms_min": min(gpu),
                "gpu_stream_ms_max": max(gpu),
                "gpu_stream_ms_stdev": statistics.pstdev(gpu),
                "gpu_stream_ms_samples": gpu,
                "host_batch_ms_median": statistics.median(host),
                "warmup": warmup,
                "repeats": repeats,
                "iterations": iterations,
                "mode": "cuda_graph",
                "sampling_order": "cyclic rotation of candidate indices",
                "candidate_count": len(launches),
                "scope": "per-operation CUDA graph replay; same captured batch size; cyclic sampling order; "
                "excludes preparation, compilation, allocation and capture; includes output copies; "
                "host_batch is amortized replay/record/wait overhead, not eager API latency",
            }
        )
    return tuple(results)


def measure_graph_pair(first, second, warmup=5, repeats=7, iterations=10):
    """Capture a pair and alternate AB/BA order with the group measurement path."""
    return measure_graph_group((first, second), warmup, repeats, iterations)
