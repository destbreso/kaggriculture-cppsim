// Replay the exact actions a Python episode emitted and assert the C++ sim
// reproduces its money and market-inventory trajectory step for step.
#include "../sim/sim.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
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


static bool load(const std::string& path, uint64_t& seed, Config& cfg,
                 std::vector<Turn>& turns,
                 std::vector<std::array<double, 2>>& money,
                 std::vector<std::array<int, N_PRODUCTS>>& inv) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    int n;
    in >> seed >> n;
    read_config(in, cfg);
    turns.resize(n);
    for (int t = 0; t < n; ++t)
        for (int p = 0; p < 2; ++p) {
            int nu, no; in >> nu >> no;
            Action& a = turns[t].a[p];
            a.n_units = std::min(nu, MAX_UNITS);
            for (int i = 0; i < nu; ++i) {
                int op, arg, cnt; in >> op >> arg >> cnt;
                if (i < MAX_UNITS) {
                    a.units[i].op = static_cast<uint8_t>(op);
                    a.units[i].arg = static_cast<uint8_t>(arg);
                    a.units[i].n = static_cast<int16_t>(cnt);
                }
            }
            a.n_orders = std::min(no, 16);
            for (int i = 0; i < no; ++i) {
                int op, item, cnt; in >> op >> item >> cnt;
                if (i < 16) {
                    a.orders[i].op = static_cast<uint8_t>(op);
                    a.orders[i].item = static_cast<uint8_t>(item);
                    a.orders[i].n = cnt;
                }
            }
        }
    std::string tag; in >> tag;                 // "TRUTH"
    money.clear(); inv.clear();
    double m0, m1;
    while (in >> m0 >> m1) {
        money.push_back({m0, m1});
        std::array<int, N_PRODUCTS> row{};
        for (int i = 0; i < N_PRODUCTS; ++i) in >> row[i];
        inv.push_back(row);
    }
    return true;
}

int main(int argc, char** argv) {
    int failures = 0;
    for (int f = 1; f < argc; ++f) {
        uint64_t seed = 0;
        std::vector<Turn> turns;
        std::vector<std::array<double, 2>> money;
        std::vector<std::array<int, N_PRODUCTS>> inv;
        Config cfg;                       // overwritten by the file's CONFIG line
        if (!load(argv[f], seed, cfg, turns, money, inv)) { ++failures; continue; }
        cfg.seed = seed;
        Sim sim(cfg);

        int first_bad = -1;
        const char* what = "";
        for (size_t t = 0; t < turns.size(); ++t) {
            // Compare BEFORE stepping: state at index t must match truth[t].
            for (int p = 0; p < 2 && first_bad < 0; ++p)
                if (sim.st.farms[p].money != money[t][p]) { first_bad = (int)t; what = "money"; }
            for (int i = 0; i < N_PRODUCTS && first_bad < 0; ++i)
                if (sim.st.market.inventory[i] != inv[t][i]) { first_bad = (int)t; what = "inventory"; }
            if (first_bad >= 0) break;
            sim.step(turns[t].a[0], turns[t].a[1]);
        }
        size_t last = money.size() - 1;
        bool final_ok = first_bad < 0 &&
                        sim.st.farms[0].money == money[last][0] &&
                        sim.st.farms[1].money == money[last][1];

        if (first_bad >= 0) {
            std::printf("FAIL %s  seed=%llu  first %s mismatch at step %d\n",
                        argv[f], (unsigned long long)seed, what, first_bad);
            std::printf("     cpp money=(%.0f, %.0f)  py money=(%.0f, %.0f)\n",
                        sim.st.farms[0].money, sim.st.farms[1].money,
                        money[first_bad][0], money[first_bad][1]);
            std::printf("     cpp inv=");
            for (int i = 0; i < N_PRODUCTS; ++i) std::printf("%d ", sim.st.market.inventory[i]);
            std::printf("\n     py  inv=");
            for (int i = 0; i < N_PRODUCTS; ++i) std::printf("%d ", inv[first_bad][i]);
            std::printf("\n");
            ++failures;
        } else if (!final_ok) {
            std::printf("FAIL %s  final money cpp=(%.0f, %.0f) py=(%.0f, %.0f)\n", argv[f],
                        sim.st.farms[0].money, sim.st.farms[1].money, money[last][0], money[last][1]);
            ++failures;
        } else {
            std::printf("PASS %s  seed=%-8llu  %zu steps exact   final=(%.0f, %.0f)\n",
                        argv[f], (unsigned long long)seed, turns.size(),
                        sim.st.farms[0].money, sim.st.farms[1].money);
        }
    }
    return failures ? 1 : 0;
}