// Profiling by ablation, because a sampling profiler cannot see through -O3.
//
// The engine is header-only and inlined, so a sampler attributes nearly
// everything to step(). Instead, stub one phase at a time and measure what the
// episode costs without it. The difference is that phase's cost.
//
//   g++ -O3 -std=c++17 -I../sim -o ablate ablate.cpp && ./ablate > ablation.json
//
// What this DOES NOT claim: that the five deltas add up to the episode. They do
// not, and the tool prints the residual rather than hiding it. Removing a phase
// changes what the phases after it see, and the loop itself costs something no
// stub removes. The share column below is therefore a share OF THE MEASURED
// DELTAS, and the residual line says how much of the episode that leaves out.
//
// Written 2026-08-26 after an audit found the published ablation table had been
// typed rather than computed: its shares summed to 101.9 % and did not match its
// own microseconds.
#define KAG_ABLATE 1
#include "sim.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace kag;
using clk = std::chrono::steady_clock;

// Which phase is stubbed for this run. 0 = nothing, the honest baseline.
// It lives in namespace kag because the seam in sim.hpp declares it from
// inside Sim, so the name it resolves is kag::g_skip.
namespace kag { int g_skip = 0; }

static double one_episode_us(int seed, int reps) {
    std::vector<double> samples;
    samples.reserve(reps);
    Action idle;
    idle.clear();
    for (int r = 0; r < reps; ++r) {
        Config cfg;
        cfg.seed = seed + r;              // vary the seed so weeds really re-roll
        Sim sim(cfg);                     // construction is NOT timed: it is a
                                          // one-off board reset, and leaving it
                                          // inside would land in the residual
                                          // and be read as loop overhead
        auto t0 = clk::now();
        while (!sim.st.done) sim.step(idle, idle);
        auto t1 = clk::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];          // median, never the mean
}

int main(int argc, char** argv) {
    const int REPS = 400, SEED = 11;
    // OUTER PASSES. A single pass is not a measurement: taken in two different
    // idle windows the same binary gave 228 and 208 microseconds an episode,
    // a 9 % swing, where two passes back to back agreed to 2.7 %. So the tool
    // repeats the whole ablation and reports the median WITH its range, which
    // is the standard this work claims everywhere else and did not meet here.
    const int PASSES = argc > 1 ? atoi(argv[1]) : 7;
    struct Row { const char* name; int id; std::vector<double> obs; };
    std::vector<Row> rows = {
        {"apply_unit_actions", 1, {}}, {"process_market", 2, {}},
        {"town_consume", 3, {}}, {"decay_plants", 4, {}}, {"end_of_day", 5, {}},
    };
    std::vector<double> fulls;

    for (int pass = 0; pass < PASSES; ++pass) {
        // Interleave the baseline with each ablation rather than batching them,
        // so a machine that drifts during the run is not read as a phase cost.
        g_skip = 0;
        fulls.push_back(one_episode_us(SEED, REPS));
        for (auto& r : rows) {
            g_skip = 0;
            double a = one_episode_us(SEED, REPS);
            g_skip = r.id;
            double stubbed = one_episode_us(SEED, REPS);
            g_skip = 0;
            double b = one_episode_us(SEED, REPS);
            r.obs.push_back((a + b) / 2.0 - stubbed);   // baseline straddles it
        }
    }
    g_skip = 0;

    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto rng_pct = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double m = v[v.size() / 2];
        return m != 0 ? 100.0 * (v.back() - v.front()) / m : 0.0;
    };
    double full = med(fulls);
    double total_delta = 0;
    for (auto& r : rows) if (med(r.obs) > 0) total_delta += med(r.obs);

    printf("{\n \"episode_us\": %.2f,\n \"episode_range_pct\": %.1f,\n"
           " \"passes\": %d,\n \"reps\": %d,\n \"seed\": %d,\n",
           full, rng_pct(fulls), PASSES, REPS, SEED);
    printf(" \"note\": \"share is of the summed deltas, not of the episode; "
           "stubs interact and the loop itself is not stubbable\",\n");
    printf(" \"phases\": [\n");
    for (size_t i = 0; i < rows.size(); ++i)
        printf("  {\"phase\": \"%s\", \"us\": %.2f, \"range_pct\": %.1f, "
               "\"share_pct\": %.2f}%s\n",
               rows[i].name, med(rows[i].obs), rng_pct(rows[i].obs),
               total_delta > 0 ? 100.0 * med(rows[i].obs) / total_delta : 0.0,
               i + 1 < rows.size() ? "," : "");
    printf(" ],\n \"summed_deltas_us\": %.2f,\n", total_delta);
    printf(" \"residual_us\": %.2f,\n", full - total_delta);
    printf(" \"residual_pct_of_episode\": %.2f\n}\n",
           full > 0 ? 100.0 * (full - total_delta) / full : 0.0);
    return 0;
}
