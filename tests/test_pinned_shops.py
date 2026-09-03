"""The shop sequence as an input, and the reason it has to be one.

    python tests/test_pinned_shops.py

The environment draws each shop from a generator that `spawn_weeds` has
already advanced once per EMPTY tile on each farm, so the eight shops a
season unlocks are a function of the board. Two agents that plant
differently never play the same world at the same seed, and the gap
between them is then dominated by which shops each was dealt: on one
measured tape, suppressing a single planting on day 0 moved the final bank
with a standard deviation of 25,842 over 40 seeds while the change itself
is worth about -1,300.

`shops=` pins the sequence so a comparison can be made inside one world.
Three checks, in the order that makes them believable:

  NULL      unpinned play is byte-identical to play before the feature
  IDENTITY  pinning the sequence a seed would have drawn changes nothing
  POWER     with the pin, an agent-side change is measured, not the draw
"""
import sys

import kagsim

SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
PASS_ACT = {"farmer": ["PASS"], "hands": [], "market": []}


def _drawn(seed, shops=None):
    """Play an idle episode and report the sequence the town unlocked."""
    g = kagsim.Game(seed, 720, shops) if shops is not None else kagsim.Game(seed)
    seq, last = [], []
    while not g.done:
        cur = list((g.observe(0).get("town") or {}).get("unlocked_shops") or [])
        if cur != last:
            seq += cur[len(last):]
            last = cur
        g.step(dict(PASS_ACT), dict(PASS_ACT))
    return seq


def _drawn_after_planting(seed, n_plant, shops=None):
    """The sequence a season unlocks when the farmer plants n tiles first.
    Planting is what the weed roll counts, so this is the smallest probe of
    the board-dependence the pin exists to remove."""
    g = kagsim.Game(seed, 720, shops) if shops is not None else kagsim.Game(seed)
    seq, last, t = [], [], 0
    while not g.done:
        cur = list((g.observe(0).get("town") or {}).get("unlocked_shops") or [])
        if cur != last:
            seq += cur[len(last):]
            last = cur
        if t < n_plant:
            a = {"farmer": ["PASS"], "hands": [],
                 "market": [["BUY_SEED", "WHEAT", 1]]}
        elif t < 2 * n_plant:
            a = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}
        else:
            a = dict(PASS_ACT)
        g.step(a, dict(PASS_ACT))
        t += 1
    return seq


def main():
    fails = []

    # NULL: the feature must be invisible when it is not used.
    for seed in (11, 23, 47, 101, 202):
        a = _drawn(seed)
        b = _drawn(seed, None)
        if a != b:
            fails.append(f"NULL seed {seed}: {a} != {b}")
    print(f"NULL      unpinned play unchanged: "
          f"{'ok' if not fails else 'FAILED'}")

    # IDENTITY: pinning what a seed would have drawn changes nothing.
    for seed in (11, 23, 47):
        drawn = _drawn(seed)
        again = _drawn(seed, drawn)
        if again != drawn:
            fails.append(f"IDENTITY seed {seed}: {again} != {drawn}")
    print(f"IDENTITY  pinning the drawn sequence is a no-op: "
          f"{'ok' if len(fails) == 0 else 'FAILED'}")

    # POWER: without the pin the board re-rolls the season; with it, it does
    # not. Read the SEQUENCE, not the bank: an idle probe never sells, so its
    # bank cannot show a shop effect even when the shops plainly change.
    free_changed = pinned_changed = 0
    N = 24
    for seed in range(1, N + 1):
        if _drawn_after_planting(seed, 0) != _drawn_after_planting(seed, 3):
            free_changed += 1
        if (_drawn_after_planting(seed, 0, SHOPS)
                != _drawn_after_planting(seed, 3, SHOPS)):
            pinned_changed += 1
    print(f"POWER     three planted tiles re-roll the season in "
          f"{free_changed}/{N} seeds unpinned, {pinned_changed}/{N} pinned")
    if free_changed == 0:
        fails.append("POWER: planting changed no sequence at all; the probe is "
                     "not exercising the board-dependence")
    if pinned_changed != 0:
        fails.append(f"POWER: the pin leaked in {pinned_changed} of {N} seeds")

    # And a pinned sequence must be exactly what was asked for.
    got = _drawn(7, SHOPS)
    if got != SHOPS:
        fails.append(f"the pin did not hold: {got}")
    print(f"HELD      the pinned sequence is what came out: "
          f"{'ok' if got == SHOPS else 'FAILED'}")

    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  ", f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
