// Kaggriculture simulator — a faithful C++ port of kaggriculture.py.
//
// Design goals, in order: (1) bit-identical results to the Python interpreter,
// (2) zero heap allocation per step, (3) trivially copyable state so search can
// snapshot and fork episodes.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <array>
#include "pyrandom.hpp"

namespace kag {

// ---------------------------------------------------------------- item space
// PRODUCTS order matches kaggriculture.py exactly. The first five products are
// also the five crops, in CROPS order, so a crop id doubles as a product id.
enum Item : uint8_t {
    WHEAT = 0, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER,
    GOOSE, COW, SHEEP, N_ITEMS
};
constexpr int N_PRODUCTS = 9;
constexpr int N_CROPS = 5;
constexpr int N_ANIMALS = 3;
inline bool is_crop(uint8_t i) { return i < N_CROPS; }
inline bool is_product(uint8_t i) { return i < N_PRODUCTS; }
inline bool is_animal(uint8_t i) { return i >= GOOSE && i < N_ITEMS; }

// ---------------------------------------------------------------- unit ops
enum Op : uint8_t {
    OP_PASS = 0, OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST,
    OP_PICKUP, OP_DROP, OP_PLACE,
    OP_PLANT, OP_WATER, OP_HARVEST, OP_FERTILIZE, OP_DIG,
    OP_BUILD_COOP, OP_BUILD_PASTURE,
    OP_FEED, OP_COLLECT_FERTILIZER, OP_CARE, OP_INVALID
};

enum MOp : uint8_t {
    M_NONE = 0, M_HIRE, M_BUY_LAND, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_SELL
};

// ---------------------------------------------------------------- static data
struct CropDef { int seed, first_yield_day, max_yield_day, interval, max_yield; bool ongoing; };
inline constexpr CropDef CROPS[N_CROPS] = {
    {  10, 2,  4, 0, 6, false },  // WHEAT
    {  20, 2,  3, 0, 4, false },  // CARROT
    {  50, 8,  8, 1, 4, true  },  // TOMATO
    { 100, 10, 10, 2, 4, true },  // STRAWBERRY
    {  80, 10, 12, 0, 6, false }, // MELON
};

enum Structure : uint8_t { ST_COOP = 0, ST_PASTURE = 1 };
struct AnimalDef { int cost; Structure structure; int first_yield_day, interval, max_held; Item product; };
inline constexpr AnimalDef ANIMALS[N_ANIMALS] = {
    { 300, ST_COOP,    4, 1, 4, EGG  },  // GOOSE
    { 400, ST_PASTURE, 8, 2, 6, MILK },  // COW
    { 500, ST_PASTURE, 6, 3, 6, WOOL },  // SHEEP
};

enum Shape : uint8_t { F_LINEAR, F_SQ, F_SQRT, F_LOG, F_LOG10, F_HINGE };
struct MarketDef { double base; int I0; double T; Shape below_f; double below_t; Shape above_f; double above_t; };
inline constexpr MarketDef MARKET[N_PRODUCTS] = {
    {  25, 10000, 400, F_SQRT,   0.80, F_LOG,    0.20 },  // WHEAT
    {  35, 10000, 450, F_HINGE,  1.00, F_SQRT,   0.70 },  // CARROT  (1.32.7)
    {  60, 10000, 200, F_HINGE,  0.40, F_SQRT,   0.60 },  // TOMATO  (1.32.7)
    { 120, 10000, 100, F_SQRT,   0.70, F_LINEAR, 1.60 },  // STRAWBERRY
    { 250, 10000, 300, F_LOG,    0.20, F_SQ,     3.60 },  // MELON
    {  50, 10000, 332, F_HINGE,  0.40, F_LOG,    0.20 },  // EGG     (1.32.7)
    { 160, 10000, 122, F_SQRT,   0.60, F_LINEAR, 1.60 },  // MILK
    { 200, 10000, 105, F_LOG,    0.20, F_SQ,     3.20 },  // WOOL
    { 100, 10000, 200, F_LINEAR, 0.40, F_LINEAR, 0.40 },  // FERTILIZER
};

inline double shape(Shape f, double x, double T = 0.0) {
    if (x < 0.0) x = 0.0;
    switch (f) {
        case F_LINEAR: return x;
        case F_SQ:     return x * x;
        case F_SQRT:   return std::sqrt(x);
        case F_LOG:    return std::log(1.0 + x);
        case F_LOG10:  return std::log10(1.0 + x);
        case F_HINGE: {                 // 1.32.7: u + 8*max(0, u-1)^2
            if (T <= 0.0) return x;
            double u = x / T;
            double e = u - 1.0;
            if (e < 0.0) e = 0.0;
            return u + 8.0 * e * e;
        }
    }
    return x;
}

// Amplitudes are derived constants; precompute once.
struct MarketAmp { double below, above; };
inline const std::array<MarketAmp, N_PRODUCTS>& market_amps() {
    static const std::array<MarketAmp, N_PRODUCTS> a = [] {
        std::array<MarketAmp, N_PRODUCTS> r{};
        for (int i = 0; i < N_PRODUCTS; ++i) {
            r[i].below = MARKET[i].below_t * MARKET[i].base / shape(MARKET[i].below_f, MARKET[i].T, MARKET[i].T);
            r[i].above = MARKET[i].above_t * MARKET[i].base / shape(MARKET[i].above_f, MARKET[i].T, MARKET[i].T);
        }
        return r;
    }();
    return a;
}

// Python: max(PRICE_FLOOR, int(round(price))). Python's round() on a float is
// round-half-to-even, which is exactly nearbyint's default mode.
inline int market_price(int item, int inv) {
    const MarketDef& p = MARKET[item];
    const MarketAmp& a = market_amps()[item];
    double price;
    if (inv < p.I0) price = p.base + a.below * shape(p.below_f, static_cast<double>(p.I0 - inv), p.T);
    else            price = p.base - a.above * shape(p.above_f, static_cast<double>(inv - p.I0), p.T);
    int v = static_cast<int>(std::nearbyint(price));
    return v < 1 ? 1 : v;
}

// ---------------------------------------------------------------- shops / town
enum ShopId : uint8_t {
    SHOP_BAKERY = 0, SHOP_BRUNCH_SPOT, SHOP_FARMERS_MARKET, SHOP_ICE_CREAM_SHOP,
    SHOP_PET_CAFE, SHOP_PIZZA_SHOP, SHOP_SMOOTHIE_SHOP, SHOP_YARN_STORE, N_SHOPS
};
// NOTE: this array is in sorted(SHOPS) order, because the environment unlocks
// with `rng.choice(sorted(SHOPS))`.
inline constexpr uint16_t SHOP_MASK[N_SHOPS] = {
    (1u << EGG) | (1u << WHEAT),                                        // BAKERY
    (1u << EGG) | (1u << WHEAT) | (1u << STRAWBERRY),                   // BRUNCH_SPOT
    (1u << WHEAT) | (1u << CARROT) | (1u << TOMATO) | (1u << STRAWBERRY),// FARMERS_MARKET
    (1u << STRAWBERRY) | (1u << MILK) | (1u << WHEAT),                  // ICE_CREAM_SHOP
    (1u << CARROT),                                                     // PET_CAFE
    (1u << MILK) | (1u << TOMATO) | (1u << WHEAT),                      // PIZZA_SHOP
    (1u << STRAWBERRY) | (1u << MILK),                                  // SMOOTHIE_SHOP
    (1u << WOOL),                                                       // YARN_STORE
};
inline constexpr int SHOP_MULT[N_SHOPS] = { 1, 1, 1, 1, 2, 1, 1, 2 };  // single-product shops pull 2x
constexpr int MAX_SHOP_INSTANCES = 8;

// ---------------------------------------------------------------- config
struct Config {
    int episode_steps = 720;
    int board_size = 10;
    int starting_money = 3000;
    int max_orders = 10;
    int turns_per_day = 24;
    int shed_capacity = 100;
    double weed_chance = 0.005;
    int shop_unlock_interval = 3;
    int shop_sell_interval = 4;
    int center_sell_interval = 24;
    int hire_mult = 1;
    uint64_t seed = 0;

    // FORCED SHOP SEQUENCE, added 2026-09-03. The environment draws each shop
    // from a generator that spawn_weeds has already advanced once per EMPTY
    // tile on each farm, so the shops a season unlocks are a function of the
    // BOARD: suppressing a single planting on day 0 re-rolls all eight of them
    // and moves the final bank with a standard deviation of 25,842 over 40
    // seeds. Two agents that plant differently therefore never play the same
    // world at the same seed, and the difference between them is dominated by
    // which shops each was dealt. Setting `n_forced_shops` pins the sequence so
    // a comparison can be made inside ONE world. Everything else, weeds
    // included, still reads the board exactly as the environment does.
    uint8_t forced_shops[MAX_SHOP_INSTANCES] = {};
    int n_forced_shops = 0;
};

// ---------------------------------------------------------------- state
enum TileKind : uint8_t { T_EMPTY = 0, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT };

struct Tile {
    TileKind kind = T_EMPTY;
    uint8_t  what = 0;          // crop id (PLANT) or animal id (COOP/PASTURE with animal)
    bool     has_animal = false;
    bool     watered_today = false;
    bool     fed_today = false;
    bool     cared_today = false;
    bool     fertilizer_available = false;
    int8_t   consecutive_dry = 0;    // unwatered (plant) / unfed (animal)
    int8_t   yield_units = 0;
    int8_t   pending_care_bonus = 0;
    int16_t  planted_day = 0;        // or placed_day
    int32_t  max_lifespan_step = -1;
    int16_t  fertilized_until_day = -1;
};

constexpr int MAX_UNITS = 40;   // farmer + hands; hires are Fibonacci-priced so this is ample
constexpr int BOARD = 10;

struct Farm {
    double money = 0;
    Tile tiles[BOARD][BOARD];
    int8_t pos_x[MAX_UNITS], pos_y[MAX_UNITS];
    int n_units = 1;                 // index 0 is the main farmer
    int n_quadrants = 1;             // NW always unlocked
    int hires_today = 0;

    int16_t shed[N_ITEMS] = {0};
    int shed_total = 0;
    int16_t seeds[N_CROPS] = {0};
    int16_t inv[MAX_UNITS][N_ITEMS] = {{0}};
    // Per-unit inventories are Python dicts, and several code paths iterate
    // them in INSERTION order. That order decides which items win the last
    // slots when the 100-item shed cap binds, so it is load-bearing, not
    // cosmetic, and must be modelled explicitly.
    uint8_t inv_keys[MAX_UNITS][N_ITEMS] = {{0}};
    uint8_t inv_nkeys[MAX_UNITS] = {0};

    // Instrumentation only - never read by the interpreter, so behaviour and
    // parity are unaffected. `discarded` is the production the 100-item shed
    // cap silently destroyed, which is otherwise invisible to search.
    int32_t discarded[N_ITEMS] = {0};
    int32_t produced[N_ITEMS] = {0};
    int32_t sold_units[N_ITEMS] = {0};
    double  sell_revenue = 0;   // coins actually received from SELLs
    double  total_spend = 0;    // coins actually paid out
    // Settle telemetry (instrumentation only, same contract as `discarded`):
    // counters of what the settle does in SILENCE, so an audit reads engine
    // truth instead of reconstructing it from state deltas outside.
    int32_t tel_sell_dead = 0;       // SELL units refused: shed empty
    int32_t tel_refused_product = 0; // BUY_PRODUCT units refused (funds/cap)
    int32_t tel_refused_seed = 0;    // BUY_SEED units refused (funds)
    int32_t tel_refused_animal = 0;  // BUY_ANIMAL units refused (funds/cap)
    int32_t tel_refused_hire = 0;    // HIREs refused (funds or unit cap)
    int32_t tel_refused_land = 0;    // BUY_LANDs refused (funds/none left)
    int32_t tel_hire_paid = 0;       // coins actually paid for hires

    void inv_add(int u, int item, int n) {
        if (n <= 0) return;
        if (inv[u][item] == 0) inv_keys[u][inv_nkeys[u]++] = (uint8_t)item;
        inv[u][item] += (int16_t)n;
    }
    void inv_erase(int u, int item) {
        inv[u][item] = 0;
        int k = 0;
        while (k < inv_nkeys[u] && inv_keys[u][k] != item) ++k;
        if (k == inv_nkeys[u]) return;
        for (int j = k; j + 1 < inv_nkeys[u]; ++j) inv_keys[u][j] = inv_keys[u][j + 1];
        inv_nkeys[u]--;
    }
    bool inv_take(int u, int item, int n) {
        if (inv[u][item] < n) return false;
        inv[u][item] -= (int16_t)n;
        if (inv[u][item] == 0) inv_erase(u, item);
        return true;
    }
    void inv_clear(int u) { for (int i = 0; i < N_ITEMS; ++i) inv[u][i] = 0; inv_nkeys[u] = 0; }
};

struct Market {
    int32_t inventory[N_PRODUCTS];
    int32_t prices[N_PRODUCTS];
};

struct State {
    Farm farms[2];
    Market market;
    uint8_t shops[MAX_SHOP_INSTANCES];
    int n_shops = 0;
    int step = 0;
    int day = 0;
    int hour = 0;
    bool done = false;
};

// ---------------------------------------------------------------- actions
struct UnitAction { uint8_t op = OP_PASS; uint8_t arg = 0; int16_t n = 1; };
struct Order { uint8_t op = M_NONE; uint8_t item = 0; int32_t n = 0; };

struct Action {
    UnitAction units[MAX_UNITS];     // units[0] = farmer
    int n_units = 1;
    Order orders[16];
    int n_orders = 0;
    void clear() { n_units = 1; units[0] = UnitAction{}; n_orders = 0; }
};

// ---------------------------------------------------------------- helpers
inline int quadrant_of(int x, int y, int bs) {
    // 0=NW 1=NE 2=SW 3=SE
    int half = bs / 2;
    return (y < half ? 0 : 2) + (x < half ? 0 : 1);
}
// Land is unlocked in the order NE, SW, SE.
inline constexpr int LAND_ORDER[3] = { 1, 2, 3 };
inline constexpr int LAND_PRICES[3] = { 1000, 2000, 4000 };

inline void shed_access_tiles(int bs, int out[4][2]) {
    int h = bs / 2;
    out[0][0] = h - 1; out[0][1] = h - 1;
    out[1][0] = h;     out[1][1] = h - 1;
    out[2][0] = h - 1; out[2][1] = h;
    out[3][0] = h;     out[3][1] = h;
}
inline bool is_shed_adjacent(int x, int y, int bs) {
    int h = bs / 2;
    return (x == h - 1 || x == h) && (y == h - 1 || y == h);
}

inline int fib(int n) { int a = 1, b = 1; for (int i = 0; i < n; ++i) { int t = b; b = a + b; a = t; } return a; }

class Sim {
public:
    Config cfg;
    State st;

    explicit Sim(const Config& c = Config{}) : cfg(c) { reset(); }

    void reset() {
        st = State{};
        int h = cfg.board_size / 2;
        for (int p = 0; p < 2; ++p) {
            Farm& f = st.farms[p];
            f.money = cfg.starting_money;
            for (int y = 0; y < cfg.board_size; ++y)
                for (int x = 0; x < cfg.board_size; ++x)
                    f.tiles[y][x].kind = (quadrant_of(x, y, cfg.board_size) == 0) ? T_EMPTY : T_LOCKED;
            f.n_units = 1;
            f.pos_x[0] = static_cast<int8_t>(h - 1);
            f.pos_y[0] = static_cast<int8_t>(h - 1);
        }
        for (int i = 0; i < N_PRODUCTS; ++i) {
            st.market.inventory[i] = MARKET[i].I0;
            st.market.prices[i] = static_cast<int>(MARKET[i].base);
        }
    }

    double reward(int p) const { return st.farms[p].money; }

    // One environment step, given both players' actions.
    void step(const Action& a0, const Action& a1) {
        if (st.done) return;
        const Action* acts[2] = { &a0, &a1 };
        int step_i = st.step;
        int day = step_i / cfg.turns_per_day;

#ifdef KAG_ABLATE
        // Profiling seam, compiled out of every normal build. tools/ablate.cpp
        // stubs one phase at a time to price it, because -O3 inlining defeats a
        // sampling profiler here. Absent from any build that is not that tool.
        extern int g_skip;
        if (g_skip != 1) for (int p = 0; p < 2; ++p) apply_unit_actions(p, *acts[p], day);
        if (g_skip != 2) process_market(*acts[0], *acts[1]);
        if (g_skip != 3) town_consume(step_i);
        if (g_skip != 4) for (int p = 0; p < 2; ++p) decay_plants(st.farms[p], step_i);
        if (g_skip != 5 && (step_i + 1) % cfg.turns_per_day == 0) end_of_day(day);
#else
        for (int p = 0; p < 2; ++p) apply_unit_actions(p, *acts[p], day);
        process_market(*acts[0], *acts[1]);
        town_consume(step_i);
        for (int p = 0; p < 2; ++p) decay_plants(st.farms[p], step_i);
        if ((step_i + 1) % cfg.turns_per_day == 0) end_of_day(day);
#endif

        st.step = step_i + 1;
        st.day = st.step / cfg.turns_per_day;
        st.hour = st.step % cfg.turns_per_day;
        if (step_i >= cfg.episode_steps - 2) st.done = true;
    }

private:
    // ------------------------------------------------------------ unit actions
    void apply_unit_actions(int p, const Action& a, int day) {
        Farm& f = st.farms[p];
        // Atomic PLANT validation: if requests for a crop exceed seeds held,
        // every PLANT of that crop this turn is dropped.
        // Demand is counted over every submitted unit action, including ones
        // addressed to hands that do not exist (matching the Python, where the
        // blocked set is computed before the position lookup no-ops).
        int demand[N_CROPS] = {0};
        for (int i = 0; i < a.n_units; ++i)
            if (a.units[i].op == OP_PLANT && a.units[i].arg < N_CROPS) demand[a.units[i].arg]++;
        int n = std::min(a.n_units, f.n_units);
        bool blocked[N_CROPS];
        for (int c = 0; c < N_CROPS; ++c) blocked[c] = demand[c] > f.seeds[c];

        for (int i = 0; i < n; ++i) {
            UnitAction u = a.units[i];
            if (u.op == OP_PLANT && u.arg < N_CROPS && blocked[u.arg]) continue;
            apply_unit(f, i, u, day);
        }
    }

    void apply_unit(Farm& f, int idx, const UnitAction& u, int day) {
        const int bs = cfg.board_size;
        int fx = f.pos_x[idx], fy = f.pos_y[idx];
        int16_t* inv = f.inv[idx];

        switch (u.op) {
            case OP_PASS: return;
            case OP_NORTH: case OP_SOUTH: case OP_EAST: case OP_WEST: {
                int nx = fx + (u.op == OP_EAST) - (u.op == OP_WEST);
                int ny = fy + (u.op == OP_SOUTH) - (u.op == OP_NORTH);
                // Movement onto LOCKED tiles is legal (hands can spawn there).
                if (nx < 0 || nx >= bs || ny < 0 || ny >= bs) return;
                f.pos_x[idx] = static_cast<int8_t>(nx);
                f.pos_y[idx] = static_cast<int8_t>(ny);
                return;
            }
            default: break;
        }

        Tile& tile = f.tiles[fy][fx];

        // Shed ops resolve before the LOCKED guard: three of the four
        // shed-access tiles start locked, and the shed itself is always owned.
        if (u.op == OP_DROP) {
            if (!is_shed_adjacent(fx, fy, bs)) return;
            // Insertion order, matching `for item, n in list(inv.items())`.
            uint8_t keys[N_ITEMS];
            int nk = f.inv_nkeys[idx];
            for (int k = 0; k < nk; ++k) keys[k] = f.inv_keys[idx][k];
            for (int k = 0; k < nk; ++k) {
                int it = keys[k];
                if (inv[it] <= 0) { f.inv_erase(idx, it); continue; }
                int room = std::max(0, cfg.shed_capacity - f.shed_total);
                int take = std::min<int>(inv[it], room);
                if (take > 0) { f.shed[it] += take; f.shed_total += take; }
                f.discarded[it] += inv[it] - take;
                f.inv_erase(idx, it);
            }
            return;
        }
        if (u.op == OP_PICKUP) {
            if (!is_shed_adjacent(fx, fy, bs)) return;
            if (u.n <= 0 || u.arg >= N_ITEMS) return;
            int take = std::min<int>(u.n, f.shed[u.arg]);
            if (take <= 0) return;
            f.shed[u.arg] -= take; f.shed_total -= take;
            f.inv_add(idx, u.arg, take);
            return;
        }
        if (u.op == OP_PLACE) {
            if (u.arg >= N_ITEMS) return;
            if (is_animal(u.arg)) {
                const AnimalDef& ad = ANIMALS[u.arg - GOOSE];
                TileKind want = (ad.structure == ST_COOP) ? T_COOP : T_PASTURE;
                if (tile.kind == want && !tile.has_animal) {
                    if (f.inv_take(idx, u.arg, 1)) {
                        Tile t{};
                        t.kind = want; t.what = u.arg; t.has_animal = true;
                        t.planted_day = static_cast<int16_t>(day);
                        tile = t;
                    }
                    return;
                }
            }
            if (is_shed_adjacent(fx, fy, bs)) {
                int n = std::min<int>(u.n, inv[u.arg]);
                if (n <= 0) return;
                n = std::min(n, std::max(0, cfg.shed_capacity - f.shed_total));
                if (n <= 0) return;
                f.inv_take(idx, u.arg, n);
                f.shed[u.arg] += n; f.shed_total += n;
            }
            return;
        }

        if (tile.kind == T_LOCKED) return;

        switch (u.op) {
            case OP_PLANT: {
                if (u.arg >= N_CROPS || tile.kind != T_EMPTY || f.seeds[u.arg] <= 0) return;
                f.seeds[u.arg] -= 1;
                const CropDef& cd = CROPS[u.arg];
                Tile t{};
                t.kind = T_PLANT; t.what = u.arg;
                t.planted_day = static_cast<int16_t>(day);
                t.consecutive_dry = 1;                 // planting day counts as unwatered
                t.yield_units = cd.ongoing ? 0 : 1;
                t.max_lifespan_step = cd.ongoing ? -1
                    : (day + cd.max_yield_day + 1) * cfg.turns_per_day;
                tile = t;
                return;
            }
            case OP_WATER: {
                if (tile.kind != T_PLANT || tile.watered_today) return;
                tile.watered_today = true;
                const CropDef& cd = CROPS[tile.what];
                if (!cd.ongoing) {
                    int age = day - tile.planted_day;
                    int w0 = (cd.max_yield_day + 1) / 2;
                    if (age >= w0 && age <= cd.max_yield_day) {
                        int bonus = (tile.fertilized_until_day >= day) ? 2 : 1;
                        tile.yield_units = static_cast<int8_t>(std::min(cd.max_yield, tile.yield_units + bonus));
                    }
                }
                return;
            }
            case OP_HARVEST: {
                if (tile.kind == T_EMPTY || tile.kind == T_WEED) return;
                if (tile.yield_units <= 0) return;
                if (tile.kind == T_PLANT) {
                    const CropDef& cd = CROPS[tile.what];
                    if (day - tile.planted_day < cd.first_yield_day) return;
                    f.inv_add(idx, tile.what, tile.yield_units);
                    f.produced[tile.what] += tile.yield_units;
                    tile.yield_units = 0;
                    if (!cd.ongoing) { Tile e{}; e.kind = T_EMPTY; tile = e; }
                } else if (tile.has_animal) {
                    f.inv_add(idx, ANIMALS[tile.what - GOOSE].product, tile.yield_units);
                    f.produced[ANIMALS[tile.what - GOOSE].product] += tile.yield_units;
                    tile.yield_units = 0;
                }
                return;
            }
            case OP_FERTILIZE: {
                if (tile.kind != T_PLANT) return;
                if (!f.inv_take(idx, FERTILIZER, 1)) return;
                tile.fertilized_until_day = std::max<int16_t>(tile.fertilized_until_day,
                                                             static_cast<int16_t>(day + 2));
                return;
            }
            case OP_DIG: {
                if (tile.kind == T_EMPTY) return;
                if (tile.has_animal) return;           // animals are not removable
                Tile e{}; e.kind = T_EMPTY; tile = e;
                return;
            }
            case OP_BUILD_COOP:
                if (tile.kind != T_EMPTY) return;
                { Tile t{}; t.kind = T_COOP; tile = t; }
                return;
            case OP_BUILD_PASTURE:
                if (tile.kind != T_EMPTY) return;
                { Tile t{}; t.kind = T_PASTURE; tile = t; }
                return;
            case OP_FEED:
                if (!tile.has_animal || tile.fed_today) return;
                if (!f.inv_take(idx, WHEAT, 1)) return;
                tile.fed_today = true;
                return;
            case OP_COLLECT_FERTILIZER:
                if (!tile.has_animal || !tile.fertilizer_available) return;
                tile.fertilizer_available = false;
                f.inv_add(idx, FERTILIZER, 1);
                f.produced[FERTILIZER] += 1;
                return;
            case OP_CARE:
                if (!tile.has_animal || tile.cared_today) return;
                tile.cared_today = true;
                return;
            default: return;
        }
    }

    // ------------------------------------------------------------ market
    struct OState { uint8_t type; uint8_t item; int32_t remaining; bool live; };

    void process_market(const Action& a0, const Action& a1) {
        const Action* acts[2] = { &a0, &a1 };
        int nq[2];
        for (int p = 0; p < 2; ++p) nq[p] = std::min(acts[p]->n_orders, cfg.max_orders);
        int max_len = std::max(nq[0], nq[1]);

        for (int i = 0; i < max_len; ++i) {
            OState os[2];
            for (int p = 0; p < 2; ++p) {
                os[p].live = false;
                if (i < nq[p]) {
                    const Order& o = acts[p]->orders[i];
                    if (o.op == M_HIRE || o.op == M_BUY_LAND) {
                        os[p].type = o.op; os[p].live = true; os[p].remaining = 1;
                    } else if (o.op != M_NONE && o.n > 0) {
                        os[p].type = o.op; os[p].item = o.item; os[p].remaining = o.n; os[p].live = true;
                    }
                }
            }
            // Atomic orders resolve once, in player order.
            for (int p = 0; p < 2; ++p) {
                if (!os[p].live) continue;
                if (os[p].type == M_HIRE) { do_hire(st.farms[p]); os[p].live = false; }
                else if (os[p].type == M_BUY_LAND) { do_buy_land(st.farms[p]); os[p].live = false; }
            }

            // Per-unit lockstep: both players see the same pre-commit inventory.
            for (;;) {
                struct Q { bool ok = false; uint8_t type = 0; uint8_t item = 0; int price = 0; };
                Q q[2];
                for (int p = 0; p < 2; ++p) {
                    if (!os[p].live || os[p].remaining <= 0) continue;
                    uint8_t t = os[p].type, it = os[p].item;
                    if (t == M_SELL && is_product(it)) {
                        q[p] = { true, t, it, market_price(it, st.market.inventory[it]) };
                    } else if (t == M_BUY_PRODUCT && (it == WHEAT || it == FERTILIZER)) {
                        // Quoted at post-buy inventory so a round trip nets zero.
                        q[p] = { true, t, it, market_price(it, st.market.inventory[it] - 1) };
                    } else if (t == M_BUY_SEED && is_crop(it)) {
                        q[p] = { true, t, it, CROPS[it].seed };
                    } else if (t == M_BUY_ANIMAL && is_animal(it)) {
                        q[p] = { true, t, it, ANIMALS[it - GOOSE].cost };
                    } else {
                        os[p].live = false;
                    }
                }
                if (!q[0].ok && !q[1].ok) break;
                bool committed = false;
                for (int p = 0; p < 2; ++p) {
                    if (!q[p].ok) continue;
                    if (commit_unit(q[p].type, q[p].item, q[p].price, st.farms[p])) {
                        os[p].remaining -= 1; committed = true;
                    } else {
                        // the whole REMAINDER of the order dies here in
                        // silence; telemetry counts every undelivered unit
                        Farm& fp = st.farms[p];
                        switch (os[p].type) {
                            case M_SELL:        fp.tel_sell_dead += os[p].remaining; break;
                            case M_BUY_PRODUCT: fp.tel_refused_product += os[p].remaining; break;
                            case M_BUY_SEED:    fp.tel_refused_seed += os[p].remaining; break;
                            case M_BUY_ANIMAL:  fp.tel_refused_animal += os[p].remaining; break;
                        }
                        os[p].live = false;
                    }
                }
                if (!committed) break;
            }
            refresh_prices();
        }
    }

    bool commit_unit(uint8_t op, uint8_t item, int price, Farm& f) {
        switch (op) {
            case M_SELL:
                if (f.shed[item] <= 0) return false;
                f.shed[item] -= 1; f.shed_total -= 1;
                f.money += price;
                f.sold_units[item] += 1;
                f.sell_revenue += price;
                if (price > 1) st.market.inventory[item] += 1;   // $1 sales don't add supply
                return true;
            case M_BUY_PRODUCT:
                if (f.money < price) return false;
                if (f.shed_total >= cfg.shed_capacity) return false;
                f.money -= price; f.total_spend += price;
                f.shed[item] += 1; f.shed_total += 1;
                st.market.inventory[item] -= 1;
                return true;
            case M_BUY_SEED:
                if (f.money < price) return false;
                f.money -= price; f.total_spend += price; f.seeds[item] += 1;
                return true;
            case M_BUY_ANIMAL:
                if (f.money < price) return false;
                if (f.shed_total >= cfg.shed_capacity) return false;
                f.money -= price; f.total_spend += price;
                f.shed[item] += 1; f.shed_total += 1;
                return true;
        }
        return false;
    }

    void refresh_prices() {
        for (int i = 0; i < N_PRODUCTS; ++i)
            st.market.prices[i] = market_price(i, st.market.inventory[i]);
    }

    void do_hire(Farm& f) {
        int cost = cfg.hire_mult * fib(f.hires_today);
        if (f.money < cost || f.n_units >= MAX_UNITS) { f.tel_refused_hire += 1; return; }
        f.money -= cost; f.total_spend += cost;
        f.tel_hire_paid += cost;
        f.hires_today += 1;
        // Spawn on the first free shed-access tile (NWSE), ties by occupancy.
        int acc[4][2]; shed_access_tiles(cfg.board_size, acc);
        int occ[4] = {0,0,0,0};
        for (int u = 0; u < f.n_units; ++u)
            for (int k = 0; k < 4; ++k)
                if (f.pos_x[u] == acc[k][0] && f.pos_y[u] == acc[k][1]) occ[k]++;
        int best = 0;
        for (int k = 1; k < 4; ++k) if (occ[k] < occ[best]) best = k;
        int idx = f.n_units++;
        f.pos_x[idx] = static_cast<int8_t>(acc[best][0]);
        f.pos_y[idx] = static_cast<int8_t>(acc[best][1]);
        f.inv_clear(idx);
    }

    void do_buy_land(Farm& f) {
        int extra = f.n_quadrants - 1;
        if (extra >= 3) { f.tel_refused_land += 1; return; }
        int cost = LAND_PRICES[extra];
        if (f.money < cost) { f.tel_refused_land += 1; return; }
        f.money -= cost; f.total_spend += cost;
        int quad = LAND_ORDER[extra];
        f.n_quadrants += 1;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x)
                if (quadrant_of(x, y, cfg.board_size) == quad && f.tiles[y][x].kind == T_LOCKED)
                    f.tiles[y][x].kind = T_EMPTY;
    }

    // ------------------------------------------------------------ town / decay
    void town_consume(int step_i) {
        if (step_i % cfg.shop_sell_interval == 0) {
            for (int s = 0; s < st.n_shops; ++s) {
                uint16_t mask = SHOP_MASK[st.shops[s]];
                int mult = SHOP_MULT[st.shops[s]];
                for (int it = 0; it < N_PRODUCTS; ++it)
                    if (mask & (1u << it)) st.market.inventory[it] -= mult;
            }
        }
        if (step_i % cfg.center_sell_interval == 0) {
            for (int it = 0; it < N_PRODUCTS; ++it)
                if (it != FERTILIZER) st.market.inventory[it] -= 1;
        }
        refresh_prices();
    }

    void decay_plants(Farm& f, int step_i) {
        // PARITY SHORTCUT, and it is a proof rather than a heuristic.
        // Every assignment of max_lifespan_step is `k * turns_per_day`
        // (or -1, which the loop rejects), so with an even turns_per_day
        // the value is even and `(step_i - max_lifespan_step) % 2` equals
        // `step_i % 2`. On an odd step no tile can satisfy the decay
        // condition, so the whole scan is provably a no-op. Guarded on
        // the config so an odd turns_per_day falls back to scanning.
        if ((cfg.turns_per_day % 2) == 0 && (step_i % 2) != 0) return;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x) {
                Tile& t = f.tiles[y][x];
                if (t.kind != T_PLANT) continue;
                if (t.max_lifespan_step < 0 || step_i < t.max_lifespan_step) continue;
                if ((step_i - t.max_lifespan_step) % 2 != 0) continue;
                t.yield_units -= 1;
                if (t.yield_units <= 0) { Tile w{}; w.kind = T_WEED; t = w; }
            }
    }

    void end_of_day(int day) {
        PyRandom rng((cfg.seed * 1000003ull) ^ static_cast<uint64_t>(day));
        for (int p = 0; p < 2; ++p) {
            Farm& f = st.farms[p];
            daily_refresh_plants(f, day);
            daily_refresh_animals(f, day);
            spawn_weeds(f, rng);
            drop_inventories(f);
            int h = cfg.board_size / 2;
            f.n_units = 1;
            f.pos_x[0] = static_cast<int8_t>(h - 1);
            f.pos_y[0] = static_cast<int8_t>(h - 1);
            f.hires_today = 0;
            for (int u = 0; u < MAX_UNITS; ++u) f.inv_clear(u);
        }
        int next_day = day + 1;
        if (next_day > 0 && next_day % cfg.shop_unlock_interval == 0 && st.n_shops < MAX_SHOP_INSTANCES) {
            // The draw is consumed either way, so pinning the sequence changes
            // the shop and nothing else about the generator's history.
            int drawn = rng.choice_index(N_SHOPS);
            st.shops[st.n_shops] = (st.n_shops < cfg.n_forced_shops)
                                       ? cfg.forced_shops[st.n_shops]
                                       : static_cast<uint8_t>(drawn);
            ++st.n_shops;
        }
    }

    void daily_refresh_plants(Farm& f, int day) {
        int next_day = day + 1;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x) {
                Tile& t = f.tiles[y][x];
                if (t.kind != T_PLANT) continue;
                bool was_watered = t.watered_today;
                t.consecutive_dry = was_watered ? 0 : static_cast<int8_t>(t.consecutive_dry + 1);
                t.watered_today = false;
                if (t.consecutive_dry >= 2) { Tile w{}; w.kind = T_WEED; t = w; continue; }
                const CropDef& cd = CROPS[t.what];
                if (!cd.ongoing) continue;
                int since = next_day - t.planted_day - cd.first_yield_day;
                if (since < 0) continue;
                if (since % cd.interval != 0) continue;
                int count = since / cd.interval + 1;
                if (count > cd.max_yield) continue;
                bool fert = was_watered && t.fertilized_until_day >= day;
                t.yield_units = static_cast<int8_t>(std::min(cd.max_yield, t.yield_units + (fert ? 2 : 1)));
                if (count == cd.max_yield) t.max_lifespan_step = (next_day + 1) * cfg.turns_per_day;
            }
    }

    void daily_refresh_animals(Farm& f, int day) {
        int next_day = day + 1;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x) {
                Tile& t = f.tiles[y][x];
                if (!t.has_animal) continue;
                t.consecutive_dry = t.fed_today ? 0 : static_cast<int8_t>(t.consecutive_dry + 1);
                if (t.consecutive_dry >= 2) {           // escapes; structure remains
                    TileKind k = t.kind;
                    Tile s{}; s.kind = k; t = s;
                    continue;
                }
                const AnimalDef& ad = ANIMALS[t.what - GOOSE];
                int since = next_day - t.planted_day - ad.first_yield_day;
                if (since >= 0 && since % ad.interval == 0) {
                    int bonus = t.fed_today ? t.pending_care_bonus : 0;
                    t.yield_units = static_cast<int8_t>(std::min(ad.max_held, t.yield_units + 1 + bonus));
                    t.pending_care_bonus = 0;
                }
                if (t.cared_today && t.fed_today) t.pending_care_bonus += 1;
                t.fertilizer_available = true;
                t.fed_today = false;
                t.cared_today = false;
            }
    }

    void spawn_weeds(Farm& f, PyRandom& rng) {
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x)
                if (f.tiles[y][x].kind == T_EMPTY && rng.random() < cfg.weed_chance)
                    f.tiles[y][x].kind = T_WEED;
    }

    void drop_inventories(Farm& f) {
        // Insertion order again: with the shed near capacity this decides which
        // goods survive the day and which are discarded.
        for (int u = 0; u < f.n_units; ++u) {
            uint8_t keys[N_ITEMS];
            int nk = f.inv_nkeys[u];
            for (int k = 0; k < nk; ++k) keys[k] = f.inv_keys[u][k];
            for (int k = 0; k < nk; ++k) {
                int it = keys[k];
                if (f.inv[u][it] <= 0) { f.inv_erase(u, it); continue; }
                int room = std::max(0, cfg.shed_capacity - f.shed_total);
                int take = std::min<int>(f.inv[u][it], room);
                if (take > 0) { f.shed[it] += take; f.shed_total += take; }
                f.discarded[it] += f.inv[u][it] - take;
                f.inv_erase(u, it);
            }
        }
    }
};

}  // namespace kag