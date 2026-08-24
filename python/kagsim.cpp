// kagsim: Python bindings for the bit-exact Kaggriculture C++ engine.
//
// L0 batch API (this file): pre-convert action streams once, then play
// fixed-vs-fixed episodes at C++ speed (~1,000+ eps/sec/core):
//
//     import kagsim
//     s = kagsim.Stream(actions)              # list of {farmer,hands,market}
//     bank_a, bank_b = kagsim.run_episode(s, s2, seed)
//     results = kagsim.run_many([(s, s2, seed), ...])   # GIL released
//
// Engine core: sim.hpp, based on nikital7's bit-exact 1.32.6 port,
// patched to 1.32.7 (hinge scarcity branch) and re-validated trace-exact.
// Action encoding mirrors tools/export_trace.py exactly.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../sim/sim.hpp"

namespace py = pybind11;
using namespace kag;

static const std::unordered_map<std::string, uint8_t> OPI = {
    {"PASS", 0}, {"NORTH", 1}, {"SOUTH", 2}, {"EAST", 3}, {"WEST", 4},
    {"PICKUP", 5}, {"DROP", 6}, {"PLACE", 7}, {"PLANT", 8}, {"WATER", 9},
    {"HARVEST", 10}, {"FERTILIZE", 11}, {"DIG", 12}, {"BUILD_COOP", 13},
    {"BUILD_PASTURE", 14}, {"FEED", 15}, {"COLLECT_FERTILIZER", 16},
    {"CARE", 17}};
static const std::unordered_map<std::string, uint8_t> ITI = {
    {"WHEAT", 0}, {"CARROT", 1}, {"TOMATO", 2}, {"STRAWBERRY", 3},
    {"MELON", 4}, {"EGG", 5}, {"MILK", 6}, {"WOOL", 7}, {"FERTILIZER", 8},
    {"GOOSE", 9}, {"COW", 10}, {"SHEEP", 11}};
static const std::unordered_map<std::string, uint8_t> MOPI = {
    {"HIRE", 1}, {"BUY_LAND", 2}, {"BUY_SEED", 3}, {"BUY_PRODUCT", 4},
    {"BUY_ANIMAL", 5}, {"SELL", 6}};

static UnitAction conv_unit(const py::handle& h) {
    UnitAction u;                       // defaults to PASS
    if (!py::isinstance<py::list>(h)) return u;
    auto v = h.cast<py::list>();
    if (v.size() == 0) return u;
    auto it = OPI.find(py::str(v[0]).cast<std::string>());
    u.op = (it == OPI.end()) ? OP_INVALID : it->second;
    if (v.size() >= 2 && py::isinstance<py::str>(v[1])) {
        auto ii = ITI.find(v[1].cast<std::string>());
        u.arg = (ii == ITI.end()) ? 255 : ii->second;
    }
    if (v.size() >= 3) {
        try { u.n = static_cast<int16_t>(v[2].cast<long>()); }
        catch (...) { u.n = 1; }
    }
    return u;
}

static Order conv_order(const py::handle& h) {
    Order o;                            // defaults to M_NONE
    if (!py::isinstance<py::list>(h)) return o;
    auto v = h.cast<py::list>();
    if (v.size() == 0) return o;
    auto it = MOPI.find(py::str(v[0]).cast<std::string>());
    if (it == MOPI.end()) return o;
    if (it->second == 1 || it->second == 2) {    // HIRE / BUY_LAND
        o.op = it->second; o.item = 0; o.n = 1;
        return o;
    }
    if (v.size() < 3) return o;
    o.op = it->second;
    auto ii = ITI.find(py::str(v[1]).cast<std::string>());
    o.item = (ii == ITI.end()) ? 255 : ii->second;
    try { o.n = static_cast<int32_t>(v[2].cast<long>()); }
    catch (...) { o.n = 0; if (o.n == 0) o.op = 0; }
    return o;
}

struct Stream {
    std::vector<Action> turns;
    explicit Stream(const py::list& acts) {
        turns.reserve(acts.size());
        for (const auto& t : acts) {
            Action a;
            a.clear();
            if (py::isinstance<py::dict>(t)) {
                auto d = t.cast<py::dict>();
                if (d.contains("farmer"))
                    a.units[0] = conv_unit(d["farmer"]);
                int nu = 1;
                if (d.contains("hands")) {
                    for (const auto& hh : d["hands"].cast<py::list>()) {
                        if (nu >= MAX_UNITS) break;
                        a.units[nu++] = conv_unit(hh);
                    }
                }
                a.n_units = nu;
                if (d.contains("market")) {
                    for (const auto& oo : d["market"].cast<py::list>()) {
                        if (a.n_orders >= 16) break;
                        Order o = conv_order(oo);
                        if (o.op != 0) a.orders[a.n_orders++] = o;
                    }
                }
            }
            turns.push_back(a);
        }
    }
    size_t size() const { return turns.size(); }
};

static std::pair<double, double> run_episode_raw(const Stream& sa,
                                                 const Stream& sb,
                                                 uint64_t seed, int steps) {
    Config c;
    c.seed = seed;
    c.episode_steps = steps;
    Sim sim(c);
    Action empty;
    empty.clear();
    for (int t = 0; !sim.st.done; ++t) {
        const Action& a = (t < static_cast<int>(sa.turns.size()))
                              ? sa.turns[t] : empty;
        const Action& b = (t < static_cast<int>(sb.turns.size()))
                              ? sb.turns[t] : empty;
        sim.step(a, b);
    }
    return {sim.reward(0), sim.reward(1)};
}

PYBIND11_MODULE(kagsim, m) {
    m.doc() = "Bit-exact Kaggriculture engine (1.32.7), batch API. "
              "Based on nikital7's C++ port, hinge-patched and "
              "trace-validated.";
    py::class_<Stream>(m, "Stream")
        .def(py::init<const py::list&>())
        .def("__len__", &Stream::size);
    m.def("run_episode",
          [](const Stream& a, const Stream& b, uint64_t seed, int steps) {
              py::gil_scoped_release rel;
              return run_episode_raw(a, b, seed, steps);
          },
          py::arg("stream_a"), py::arg("stream_b"), py::arg("seed"),
          py::arg("steps") = 720);
    m.def("run_many",
          [](const std::vector<std::tuple<const Stream*, const Stream*,
                                          uint64_t>>& jobs, int steps) {
              std::vector<std::pair<double, double>> out;
              out.reserve(jobs.size());
              py::gil_scoped_release rel;
              for (const auto& [a, b, seed] : jobs)
                  out.push_back(run_episode_raw(*a, *b, seed, steps));
              return out;
          },
          py::arg("jobs"), py::arg("steps") = 720);
    m.attr("__version__") = "0.1.0";
    m.attr("ENGINE_VERSION") = "1.32.7";
}
