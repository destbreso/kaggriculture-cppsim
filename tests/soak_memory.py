"""Memory soak: RSS trend under the three Python-facing usage patterns.

    python tests/soak_memory.py

A: run_many batches (the search-driver pattern), B: Stream object churn,
C: run_episode single-call loop. A leak shows as monotone RSS growth
proportional to work done; allocator high-water shows as a plateau.
Reference run on macOS arm64 (Python 3.11, 60k episodes + 20k Stream
constructions): A -1.05 MB, B plateau at the high-water mark, C +0.00 MB.
Uses the bundled traces, so it runs from a bare clone. macOS/Linux only
(reads RSS via ps).
"""
import gc
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_golden import parse_trace  # noqa: E402

import kagsim  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PID = os.getpid()


def rss_mb():
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(PID)])
    return int(out.split()[0]) / 1024.0


def halves_delta(samples):
    half = len(samples) // 2
    return sum(samples[half:]) / half - sum(samples[:half]) / half


trace = sorted((ROOT / "traces").glob("*.txt"))[0]
_seed, acts, _finals = parse_trace(trace)
sa = kagsim.Stream(acts[0])
sb = kagsim.Stream(acts[1])
idle = kagsim.Stream([])
print(f"trace {trace.name}, baseline rss {rss_mb():.1f} MB", flush=True)

print("\nA: run_many, 40 batches x 1,000 episodes")
samples = []
for b in range(40):
    jobs = [(sa, sb, b * 1000 + i) for i in range(500)]
    jobs += [(sb, idle, b * 1000 + i) for i in range(500)]
    res = kagsim.run_many(jobs)
    assert len(res) == 1000
    del res, jobs
    gc.collect()
    samples.append(rss_mb())
a = halves_delta(samples)
print(f"A: {a:+.2f} MB across 20,000 episodes (half-vs-half)")

print("\nB: Stream churn, 100 rounds x 200 constructions")
samples = []
for _ in range(100):
    tmp = [kagsim.Stream(acts[0]) for _ in range(200)]
    del tmp
    gc.collect()
    samples.append(rss_mb())
b = halves_delta(samples)
q = len(samples) // 4
tail = sum(samples[3 * q:]) / q - sum(samples[2 * q:3 * q]) / q
print(f"B: {b:+.2f} MB across 20,000 Streams; last-quarter slope "
      f"{tail:+.2f} MB (a plateau here is allocator high-water, not a leak)")

print("\nC: run_episode, 20,000 single calls")
samples = []
for k in range(20):
    for i in range(1000):
        kagsim.run_episode(sa, sb, k * 1000 + i)
    gc.collect()
    samples.append(rss_mb())
c = halves_delta(samples)
print(f"C: {c:+.2f} MB across 20,000 episodes")

leak = a > 5.0 or c > 5.0 or tail > 5.0
print(f"\nSUMMARY  A {a:+.2f}  B {b:+.2f} (tail {tail:+.2f})  C {c:+.2f} MB")
print("VERDICT:", "SUSPECT" if leak else "CLEAN")
sys.exit(1 if leak else 0)
