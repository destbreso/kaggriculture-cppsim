// What a Mersenne Twister costs to SEED against what it costs to COPY.
//
//   g++ -O3 -std=c++17 -I../sim -o seedcost seedcost.cpp && ./seedcost
//
// This is the measurement behind two claims the speedup notebook makes: that
// seeding is where end_of_day's time goes, and that a cache of already-seeded
// states is worth a further multiple. Both were typed into the page before this
// existed. Run it on the pre-fix engine and on the current one to get the pair.
//
// Method notes, because a micro-benchmark this small is easy to get wrong:
//   * the compiler will delete work whose result is unused, so every timed
//     loop feeds a volatile sink;
//   * a copy of a seeded generator is what a cache hands out, so that is what
//     is timed, not a reference;
//   * medians of repetitions, never a single pass, and the repetition spread
//     is printed so a reader can see whether the number is stable.
#include "pyrandom.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace kag;
using clk = std::chrono::steady_clock;

static volatile double sink = 0.0;

int main() {
    const int REPS = 21, N = 20000;

    // Seeding: construct a fresh generator, as end_of_day does once a day.
    std::vector<double> seed_us;
    for (int r = 0; r < REPS; ++r) {
        auto t0 = clk::now();
        for (int i = 0; i < N; ++i) {
            PyRandom g(static_cast<uint64_t>(i));
            sink = sink + g.random();          // keep the construction alive
        }
        auto t1 = clk::now();
        seed_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
    }

    // Copying: what a per-thread cache of seeded states would hand out.
    PyRandom seeded(20260826ull);
    std::vector<double> copy_us;
    for (int r = 0; r < REPS; ++r) {
        auto t0 = clk::now();
        for (int i = 0; i < N; ++i) {
            PyRandom g = seeded;               // the cache hit
            sink = sink + g.random();
        }
        auto t1 = clk::now();
        copy_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
    }

    // One draw, so the reader can see that use is not where the time is.
    std::vector<double> draw_us;
    for (int r = 0; r < REPS; ++r) {
        PyRandom g(7ull);
        auto t0 = clk::now();
        for (int i = 0; i < N; ++i) sink = sink + g.random();
        auto t1 = clk::now();
        draw_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
    }

    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto spread = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return 100.0 * (v.back() - v.front()) / v.front();
    };

    printf("{\n \"reps\": %d,\n \"iters_per_rep\": %d,\n", REPS, N);
    printf(" \"seed_us\": %.4f,\n \"seed_spread_pct\": %.1f,\n",
           med(seed_us), spread(seed_us));
    printf(" \"copy_us\": %.4f,\n \"copy_spread_pct\": %.1f,\n",
           med(copy_us), spread(copy_us));
    printf(" \"draw_us\": %.5f,\n \"draw_spread_pct\": %.1f,\n",
           med(draw_us), spread(draw_us));
    printf(" \"seed_over_copy\": %.1f,\n", med(seed_us) / med(copy_us));
    printf(" \"seed_over_draw\": %.1f\n}\n", med(seed_us) / med(draw_us));
    return 0;
}
