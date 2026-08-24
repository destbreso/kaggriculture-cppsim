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

## Install

The repo IS the distribution channel, deliberately. No PyPI package,
and none planned: this is a competition-scoped tool tied to a moving
engine, the competition ends soon, and a registry package would be
maintenance with no upside: worse, a stale `pip install kagsim` after a
rebalance would hand people a confidently wrong simulator. Installing
from the repo means you always build today's code against today's
traces, and you read this file on the way in.

```bash
# as a dependency, from anywhere
pip install git+https://github.com/destbreso/kaggriculture-cppsim

# or pinned in a requirements.txt / pyproject dependency list
# kagsim @ git+https://github.com/destbreso/kaggriculture-cppsim@main

# or from a local clone, editable while you hack on it
git clone https://github.com/destbreso/kaggriculture-cppsim
pip install -e kaggriculture-cppsim
```

Needs a C++17 compiler (clang or gcc; on Kaggle's Linux images gcc is
present). Kaggle NOTEBOOK kernels run without internet, so a git
install will not work there: clone locally, or attach the repo as a
Kaggle dataset and `pip install /kaggle/input/<dataset>/`.

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

## How to use it well

**Rule one: validate before you trust.** Run the golden test after every
install and after every engine update, and wire your tools with a
FALLBACK to the real environment so a divergence degrades to slow
instead of to wrong:

```python
def make_bank_fn():
    """Prefer kagsim; fall back to kaggle_environments if it is absent
    or fails its self-check. Same signature either way."""
    try:
        import kagsim
        # self-check: one known episode must reproduce exactly
        s = kagsim.Stream([])                    # idle vs idle
        b0, b1 = kagsim.run_episode(s, s, seed=11)
        assert (b0, b1) == (3000.0, 3000.0), "kagsim self-check failed"

        def bank(stream_a, stream_b, seed):
            return kagsim.run_episode(stream_a, stream_b, seed)
        bank.backend = "kagsim"
        return bank
    except Exception:
        from kaggle_environments import make

        def bank(actions_a, actions_b, seed):
            env = make("kaggriculture",
                       configuration={"episodeSteps": 720,
                                      "seed": int(seed)})
            env.reset(2)
            t = 0
            while not env.done:
                a = actions_a[t] if t < len(actions_a) else None
                b = actions_b[t] if t < len(actions_b) else None
                env.step([a or {"farmer": ["PASS"], "hands": [],
                                "market": []},
                          b or {"farmer": ["PASS"], "hands": [],
                                "market": []}])
                t += 1
            return (float(env.state[0].reward or 0),
                    float(env.state[1].reward or 0))
        bank.backend = "kaggle_environments"
        return bank
```

Stronger than the smoke self-check: keep one of the bundled traces next
to your pipeline and assert `tests/test_golden.py` passes in CI or at
tool start-up. It costs half a second.

**Where kagsim applies, and where it does not:**

* USE IT for anything fixed-vs-fixed: replayed opponents, recorded
  streams, search populations, arena round-robins, big seed panels.
* DO NOT use it (yet) for opponents that are live Python callables: the
  L0 API never calls back into Python. Those games stay on the real
  environment until the L1 step-mode layer exists: a hybrid evaluator
  (kagsim for the fixed games, real env for the adaptive ones) is the
  practical pattern and works well.
* Re-derive any number you plan to PUBLISH on the real environment at
  least once. Bit-exact is bit-exact, but the claim should not depend
  on my build chain.

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
