# Roadmap

Where the layers stand, what each one is for, and why the next one deliberately waits.

## L0: batch streams. Shipped.

Pre-converted `Stream`s, `run_episode` / `run_many` with the GIL released, ~2,000 episodes/sec/core. The tool for fixed-vs-fixed workloads: seed panels, distribution reads, arena deepening. If both sides of your game are recorded action streams, this layer already saturates the problem and nothing below will make it faster.

## L1: step-mode. Shipped in 0.2.0.

`Game(seed)` / `observe(player)` / `step(a, b)`: the simulator hands a live Python agent the same observation dict the real interpreter builds, turn by turn. Proven two ways in `tests/test_l1.py`: a lockstep observation diff against the real environment (6/6 bundled traces exact, ~10k compared field-blocks each), and a heavyweight public adaptive agent reproducing its real-environment final banks to the dollar. The wall-clock cost of L1 is your agent's own Python, not the engine: a no-op agent runs hundreds of episodes per second, a heavy one ~15x faster than on the real environment. This is the tool for adaptive opponents, decision layers, and per-turn experimentation.

## L2: vectorized. Designed, deliberately not built yet.

The shape: a `VecGame` holding N simulations, `step_batch` advancing them in lockstep with the GIL released across a thread pool, `observe_batch` returning observations as numpy tensors (per-farm tile planes, market vectors, scalar banks) instead of dicts. The entire point is that a neural policy evaluates once per BATCH rather than once per game, which is what reinforcement learning at scale needs.

Why it waits, stated honestly:

1. It accelerates nothing that exists today. Fixed-vs-fixed saturates in L0. Live Python agents bottleneck on their own Python, which batching the engine cannot fix; L1 is already the optimal tool there.
2. The hard part of L2 is not code, it is the TENSOR ENCODING, and an encoding chosen without the network that will consume it is a representation chosen blind. The correct schema derives from the policy you intend to train, not the other way around.

So L2 gets built together with its first consumer: a self-play RL loop whose network dictates the observation schema. If you are that consumer, open an issue: the engine side is a weekend of work once the schema is fixed, and the state struct in `sim/sim.hpp` already holds everything the planes need.

## Settle telemetry. Shipped.

Per-farm counters at the exact settle sites of what the engine does
silently (refused purchases per op with the full remaining quantity
counted at order death, dead SELL units, shed-cap destruction, real hire
cost), exposed as `Game.telemetry(player)`. Instrumentation only:
behaviour and parity unaffected, pinned by `tests/test_telemetry.py`
known-truth scenarios. Built because outside-in reconstruction of these
events from state deltas is both slower (it needs observations) and
fragile (same-turn effects confound attribution); engine-side truth costs
one integer increment at code paths that already exist. Candidate
extensions if a consumer asks: per-item breakdowns, per-turn event logs
behind a flag, and mirroring the counters into `run_episode`/`run_many`
results for L0 batch workloads.
