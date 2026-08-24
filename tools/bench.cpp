// Throughput benchmark: replay a recorded episode as fast as possible.
#include "../sim/sim.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace kag;
struct Turn { Action a[2]; };

// Older trace files have no CONFIG line. Peek for the token and rewind if it is
// absent, so both formats load.
static void read_config(std::istream& in, Config& cfg) {
    std::streampos pos = in.tellg();
    std::string tok;
    if (!(in >> tok)) return;
    if (tok != "CONFIG") { in.clear(); in.seekg(pos); return; }
    in >> cfg.episode_steps >> cfg.board_size >> cfg.starting_money
       >> cfg.max_orders >> cfg.turns_per_day >> cfg.shed_capacity
       >> cfg.weed_chance >> cfg.shop_unlock_interval
       >> cfg.shop_sell_interval >> cfg.center_sell_interval >> cfg.hire_mult;
}


static void load(const std::string& path, uint64_t& seed, Config& cfg,
                 std::vector<Turn>& turns) {
    std::ifstream in(path);
    int n; in >> seed >> n;
    read_config(in, cfg);
    turns.resize(n);
    for (int t = 0; t < n; ++t)
        for (int p = 0; p < 2; ++p) {
            int nu, no; in >> nu >> no;
            Action& a = turns[t].a[p];
            a.n_units = std::min(nu, MAX_UNITS);
            for (int i = 0; i < nu; ++i) {
                int op, arg, cnt; in >> op >> arg >> cnt;
                if (i < MAX_UNITS) { a.units[i] = { (uint8_t)op, (uint8_t)arg, (int16_t)cnt }; }
            }
            a.n_orders = std::min(no, 16);
            for (int i = 0; i < no; ++i) {
                int op, item, cnt; in >> op >> item >> cnt;
                if (i < 16) a.orders[i] = { (uint8_t)op, (uint8_t)item, cnt };
            }
        }
}

int main(int argc, char** argv) {
    uint64_t seed; std::vector<Turn> turns;
    Config base;                      // overwritten by the file's CONFIG line
    load(argc > 1 ? argv[1] : "../data/replay_70117.txt", seed, base, turns);
    int reps = argc > 2 ? std::atoi(argv[2]) : 2000;

    double sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        Config cfg = base; cfg.seed = seed + r;   // vary seed so weeds actually re-roll
        Sim sim(cfg);
        for (auto& tn : turns) sim.step(tn.a[0], tn.a[1]);
        sink += sim.st.farms[0].money;
    }
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d episodes in %.3fs  =  %.0f eps/sec  (%.3f ms/episode)   [sink %.0f]\n",
                reps, dt, reps / dt, dt / reps * 1000, sink);
    return 0;
}