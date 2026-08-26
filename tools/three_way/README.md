# The three-engine comparison

The harness behind the published before/after figures, kept here because it
produced a headline number while living in a scratch directory with no version
control and no record of the machine conditions it ran under.

It builds three engines into one process and times them **interleaved**, so
drift hits all three equally:

* **A** nikital7's pristine port, 1.32.6, unmodified. The baseline this work
  stands on, and the author deserves the credit for it.
* **B** ours before this iteration: the same port patched to 1.32.7, with no
  parity shortcut and no RNG changes.
* **C** ours after.

## Running it

The three headers are not in this repo, because two of them are other builds:
`orig_sim.hpp` and `orig_pyrandom.hpp` are the pristine port, `sim_noparity.hpp`
is this repo at commit `914b7e1`, and `sim.hpp` plus `pyrandom.hpp` are HEAD.
Assemble them in one directory, then:

    g++ -O3 -std=c++17 -I. -o three three_way.cpp
    ./three 1500 9 1.5

The last argument is the load-average threshold and it **refuses** rather than
warns. Pass `--force` as a fifth argument only if you are prepared to label the
result contaminated; the JSON records `load_average` and `forced` either way, so
the artifact says whether to doubt it. That stamp exists because an earlier
version did not have it, and a number taken at load 3.3 sat on a published page
for a day with nothing in the file to show it.
