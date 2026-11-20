"""GPU timing.

graph (default): each candidate's launch is captured `iterations` times into
  one CUDA Graph; replays are timed with CUDA events, candidates interleaved
  in rotating order. Launch overhead is mostly removed.
cold: eager launches, L2 is flushed before every timed launch, one launch per
  event pair. Measures performance with inputs coming from HBM.
Both exclude compilation, make_plan, input generation and allocation.
Never compare a graph number with a cold number.
"""

import statistics

import torch


def _summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def graph(launches, warmup=3, repeats=15, iterations=20):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = []
    with torch.cuda.stream(stream):
        for launch in launches:
            for _ in range(warmup):
                launch()
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                for _ in range(iterations):
                    launch()
            g.replay()
            stream.synchronize()
            graphs.append(g)
        samples = [[] for _ in launches]
        for r in range(repeats):
            for offset in range(len(launches)):
                i = (r + offset) % len(launches)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                graphs[i].replay()
                end.record(stream)
                end.synchronize()
                samples[i].append(start.elapsed_time(end) / iterations)
    return [dict(_summary(s), mode="graph", iterations=iterations) for s in samples]


def cold(launches, warmup=3, repeats=30):
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(2 * l2 // 4, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    samples = [[] for _ in launches]
    with torch.cuda.stream(stream):
        for launch in launches:
            for _ in range(warmup):
                launch()
        stream.synchronize()
        for r in range(repeats):
            for offset in range(len(launches)):
                i = (r + offset) % len(launches)
                flush.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                launches[i]()
                end.record(stream)
                end.synchronize()
                samples[i].append(start.elapsed_time(end))
    return [dict(_summary(s), mode="cold") for s in samples]
