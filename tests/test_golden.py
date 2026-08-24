"""Golden test: the kagsim bindings must reproduce every bundled trace.

    python tests/test_golden.py

Each trace records the encoded actions both agents produced in the REAL
environment plus both players' money at every step (the TRUTH block).
Here we rebuild per-turn action dicts from the encoding and replay them
through the bindings, demanding the final banks match. The rebuild only
has to round-trip to the same (op, arg, n) triples the sim consumes,
so every unit is emitted as [OP, ITEM, n] even when the original had no
item: the triple is identical either way.

(`tools/validate.cpp` checks the C++ core step-by-step; this checks the
Python-facing path, dict conversion included, end to end.)
"""
import sys
from pathlib import Path

import kagsim

ROOT = Path(__file__).resolve().parent.parent
OPS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE",
       "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
       "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
         "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]
MOPS = ["NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL",
        "SELL"]


def unit_of(op, arg, n):
    name = OPS[op] if op < len(OPS) else "UNKNOWN_OP"
    item = ITEMS[arg] if arg < len(ITEMS) else "UNKNOWN_ITEM"
    return [name, item, n]


def order_of(op, item, n):
    if op <= 0 or op >= len(MOPS):
        return None
    if op in (1, 2):                   # HIRE / BUY_LAND
        return [MOPS[op]]
    return [MOPS[op], ITEMS[item] if item < len(ITEMS) else "UNKNOWN_ITEM", n]


def parse_trace(path):
    lines = path.read_text().splitlines()
    seed, n = (int(x) for x in lines[0].split())
    i = 1
    if lines[i].startswith("CONFIG"):
        i += 1
    acts = [[], []]
    for _t in range(n):
        for seat in (0, 1):
            toks = lines[i].split()
            i += 1
            nu, no = int(toks[0]), int(toks[1])
            p = 2
            units = []
            for _ in range(nu):
                units.append(unit_of(int(toks[p]), int(toks[p + 1]),
                                     int(toks[p + 2])))
                p += 3
            orders = []
            for _ in range(no):
                o = order_of(int(toks[p]), int(toks[p + 1]),
                             int(toks[p + 2]))
                p += 3
                if o:
                    orders.append(o)
            acts[seat].append({"farmer": units[0] if units else ["PASS"],
                               "hands": units[1:], "market": orders})
    assert lines[i] == "TRUTH", f"expected TRUTH at line {i}"
    final = lines[-1].split()
    return seed, acts, (float(final[0]), float(final[1]))


def main():
    traces = sorted((ROOT / "traces").glob("replay_*.txt"))
    if not traces:
        print("no traces bundled?")
        return 1
    fails = 0
    for p in traces:
        seed, acts, finals = parse_trace(p)
        sa, sb = kagsim.Stream(acts[0]), kagsim.Stream(acts[1])
        b0, b1 = kagsim.run_episode(sa, sb, seed)
        ok = abs(b0 - finals[0]) < 0.5 and abs(b1 - finals[1]) < 0.5
        print(f"{'PASS' if ok else 'FAIL'} {p.name:28s} kagsim=({b0:,.0f}, "
              f"{b1:,.0f})  real=({finals[0]:,.0f}, {finals[1]:,.0f})")
        fails += 0 if ok else 1
    print("GOLDEN:", "PASS" if fails == 0 else f"{fails} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
