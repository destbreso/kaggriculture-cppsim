# kaggriculture-cppsim

A bit-exact C++ engine for Kaggle's
[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
competition at **1.32.7**, with a batch Python API (`kagsim`). Episodes
run in well under a millisecond: cheap enough to ask statistical
questions properly instead of guessing.

## Credit first

The engine core is
**[nikital7's bit-exact C++ port](https://www.kaggle.com/code/nikital7/4000x-environment-speedup-kaggriculture)**,
which solved the three genuinely hard problems: CPython's Mersenne
Twister exactly (init_by_array seeding, 53-bit doubles, `_randbelow`
rejection sampling), Python dict insertion-order semantics where the
100-item shed cap decides which goods die, and slot-index market
resolution. This repository is that work, carried forward:

* **patched to engine 1.32.7**: the upstream port targets 1.32.6, which
  predates the rebalance that gave carrot, tomato and egg a convex
  `hinge` scarcity branch (and carrot a 5x amplitude). The patch is the
  `F_HINGE` shape plus three market rows in `sim/sim.hpp`;
* **re-validated the adversarial way**: the pristine 1.32.6 build FAILS
  every 1.32.7 trace in `traces/` (first divergence: a $4 carrot sale
  at step 97), the patched build reproduces money and market inventory
  for both players at every one of 719 steps, on all of them: including
  heavy episodes banking 85-114k with shed-cap pressure;
* **wrapped for Python** so any tool can plug it in.

## Install and use

```bash
pip install .          # builds the extension (needs a C++17 compiler)
```

```python
import kagsim

# actions: the same per-turn dicts the real environment consumes
stream = kagsim.Stream(actions)            # convert once
idle = kagsim.Stream([])

bank_a, bank_b = kagsim.run_episode(stream, idle, seed=11)

# batch mode, GIL released: characterise a strategy on 2,000 seeds in ~1 s
jobs = [(stream, idle, seed) for seed in range(2000)]
results = kagsim.run_many(jobs)            # ~2,000+ episodes/sec/core
```

Measured on an Apple-silicon laptop, single core: **~2,100 episodes/sec**
through the Python API (0.18-0.30 ms/episode for the bare C++ core).
The real environment runs ~1.2 episodes/sec.

## What this is for

* **Fixed-vs-fixed evaluation at scale**: seed panels of thousands
  instead of three, which is what kills panel overfitting (a search
  that spends generations on a 3-seed screen learns the seeds).
* **Search inner loops**: genetic/annealing searches whose candidates
  are action streams need no Python callback at all.
* **RL**: a step-level API is the natural next layer over `Sim::step`
  (see the architecture note below).

## Verifying it yourself

Never trust a simulator you have not validated:

```bash
# C++ core against the bundled 1.32.7 traces
g++ -O3 -std=c++17 -o validate tools/validate.cpp && ./validate traces/*.txt

# Python bindings against the same traces
python tests/test_golden.py

# export fresh traces from YOUR environment and check those instead
python tools/export_trace.py --agents your_agent.py,other.py 11 23 47
```

The trace format records the actions both agents actually produced plus
money and market inventory at every step; `validate` replays the actions
and demands equality. If you find a divergence, assume this port is
wrong and please open an issue with the trace.

## Architecture

```
sim/        sim.hpp, pyrandom.hpp: the engine core (header-only)
tools/      validate.cpp, bench.cpp, export_trace.py: the fidelity harness
traces/     the bundled 1.32.7 validation battery
python/     kagsim.cpp: pybind11 bindings (L0 batch API)
tests/      golden test through the bindings
```

Layers, by design:

* **L0 (this release)**: pre-converted `Stream`s, `run_episode` /
  `run_many`, GIL released. For fixed-vs-fixed workloads.
* **L1 (planned)**: step-mode `Env` with dict observations mirroring the
  real environment: for adaptive agents and RL, at ~100x the real env.
* **L2 (planned)**: vectorized batched envs for serious RL.

## Version discipline

`kagsim.ENGINE_VERSION` says which engine this build reproduces
(currently `1.32.7`). A measurement carries the engine it was taken on:
when the competition rebalances, the traces regenerate first, the sim
gets patched second, and nothing is trusted until `validate` passes
again.

Apache 2.0, preserving the upstream license. Maintained by
[destbreso](https://www.kaggle.com/destbreso); engine core by
[nikital7](https://www.kaggle.com/nikital7).
