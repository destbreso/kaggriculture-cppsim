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
#include <string>
#include <vector>

using namespace kag;
using clk = std::chrono::steady_clock;

// Which phase is stubbed for this run. 0 = nothing, the honest baseline.
int g_skip = 0;

static double one_episode_us(int seed, int reps) {
    std::vector<double> samples;
    samples.reserve(reps);
    Action idle;
    idle.clear();
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        Sim sim;
        sim.reset(seed + r);
        while (!sim.st.done) sim.step(idle, idle);
        auto t1 = clk::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];          // median, never the mean
}

int main() {
    const int REPS = 400, SEED = 11;
    struct Row { const char* name; int id; double us; };
    std::vector<Row> rows = {
        {"apply_unit_actions", 1, 0}, {"process_market", 2, 0},
        {"town_consume", 3, 0}, {"decay_plants", 4, 0}, {"end_of_day", 5, 0},
    };

    // Interleave the baseline with each ablation rather than batching them, so
    // a machine that drifts during the run cannot be mistaken for a phase cost.
    g_skip = 0;
    double full = one_episode_us(SEED, REPS);
    double total_delta = 0;
    for (auto& r : rows) {
        g_skip = 0;
        double a = one_episode_us(SEED, REPS);
        g_skip = r.id;
        double stubbed = one_episode_us(SEED, REPS);
        g_skip = 0;
        double b = one_episode_us(SEED, REPS);
        r.us = (a + b) / 2.0 - stubbed;          // baseline straddles the probe
        if (r.us > 0) total_delta += r.us;
    }
    g_skip = 0;

    printf("{\n \"episode_us\": %.2f,\n \"reps\": %d,\n \"seed\": %d,\n",
           full, REPS, SEED);
    printf(" \"note\": \"share is of the summed deltas, not of the episode; "
           "stubs interact and the loop itself is not stubbable\",\n");
    printf(" \"phases\": [\n");
    for (size_t i = 0; i < rows.size(); ++i)
        printf("  {\"phase\": \"%s\", \"us\": %.2f, \"share_pct\": %.2f}%s\n",
               rows[i].name, rows[i].us,
               total_delta > 0 ? 100.0 * rows[i].us / total_delta : 0.0,
               i + 1 < rows.size() ? "," : "");
    printf(" ],\n \"summed_deltas_us\": %.2f,\n", total_delta);
    printf(" \"residual_us\": %.2f,\n", full - total_delta);
    printf(" \"residual_pct_of_episode\": %.2f\n}\n",
           full > 0 ? 100.0 * (full - total_delta) / full : 0.0);
    return 0;
}
