"""L1 validation: the step-mode Game must hand each seat the SAME
observation the real interpreter builds, at every turn.

    python tests/test_l1.py            # needs kaggle-environments==1.32.7

Method: drive the real environment and kagsim.Game in lockstep with the
recorded actions of a bundled trace (both seats fixed), and deep-compare
the observation dicts turn by turn: farms (tiles included), market,
town, day, hour and each seat's private block. Any state divergence
anywhere in the engine shows up here as a dict diff at the turn it
first happens. Final rewards must match too.

This is the gate that makes L1 trustworthy for adaptive agents: an
agent is a function of its observation, so observation equality at
every turn implies behavioural equality inside the simulator.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_golden import parse_trace  # noqa: E402

import kagsim  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMPARE = ("player", "day", "hour", "farms", "market", "town", "private")


def norm(x):
    return json.loads(json.dumps(x))


def diff_path(a, b, path=""):
    """First differing path between two normalised structures, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k} missing in real"
            if k not in b:
                return f"{path}.{k} missing in kagsim"
            d = diff_path(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff_path(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return None


def run_trace(path):
    from kaggle_environments import make

    seed, acts, finals = parse_trace(path)
    n = len(acts[0])
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed})
    env.reset(2)
    game = kagsim.Game(seed)

    checked = 0
    for t in range(n):
        for p in (0, 1):
            real = norm(env.state[p].observation)
            ours = norm(game.observe(p))
            for key in COMPARE:
                if key not in real:
                    continue
                d = diff_path(real[key], ours[key], key)
                if d:
                    return (f"turn {t} seat {p}: {d}", checked)
                checked += 1
        step_real = norm(env.state[0].observation).get("step")
        if step_real is not None and step_real != game.step_count:
            return (f"turn {t}: step {step_real} vs {game.step_count}",
                    checked)
        env.step([acts[0][t], acts[1][t]])
        game.step(acts[0][t], acts[1][t])
    r_real = [float(s.reward or 0) for s in env.state]
    r_ours = [game.reward(0), game.reward(1)]
    if r_real != r_ours:
        return (f"final rewards {r_real} vs {r_ours}", checked)
    if (r_ours[0], r_ours[1]) != finals:
        return (f"rewards vs trace TRUTH {finals}: {r_ours}", checked)
    return (None, checked)


def main():
    traces = sorted((ROOT / "traces").glob("*.txt"))
    fails = 0
    for tr in traces:
        err, checked = run_trace(tr)
        tag = "OK  " if err is None else "FAIL"
        print(f"{tag} {tr.name}: {checked:,} field-blocks compared"
              + (f"  -> {err}" if err else ""))
        fails += err is not None
    print(f"\n{len(traces) - fails}/{len(traces)} traces observation-exact")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
