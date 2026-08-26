// Does precomputing the constant half of the seeding change anything under
// threads or across episodes? Three checks, because the optimisation adds
// the one thing that can break a deterministic engine: shared state.
//
//   g++ -O3 -std=c++17 -I../sim -o test_rng_shared test_rng_shared.cpp
//   ./test_rng_shared
//
// 1. IDENTITY. The precomputed table must equal init_genrand(19650218)
//    word for word. If it does not, everything downstream is wrong and
//    the golden tests would only tell us so indirectly.
// 2. IDEMPOTENCE. The same seed must produce the same draw sequence no
//    matter how many generators were built before it, in what order.
// 3. THREAD SAFETY, tested where it can actually break: many threads
//    racing to be the first caller, in a fresh process, so the
//    function-local static's initialisation guard is exercised rather
//    than assumed.
#include "pyrandom.hpp"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using namespace kag;

static std::vector<double> draws(uint64_t seed, int n) {
    PyRandom r(seed);
    std::vector<double> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back(r.random());
    return out;
}

int main() {
    int failures = 0;

    // ---- 3 first, on purpose: the static is still uninitialised here, so
    // this is the only moment the initialisation race can be observed.
    const int T = 16, DRAWS = 64;
    std::vector<std::vector<double>> got(T);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> pool;
    for (int t = 0; t < T; ++t)
        pool.emplace_back([&, t] {
            ready.fetch_add(1);
            while (!go.load()) { }              // all threads start together
            got[t] = draws(20260826ull, DRAWS);
        });
    while (ready.load() < T) { }
    go.store(true);
    for (auto& th : pool) th.join();
    for (int t = 1; t < T; ++t)
        if (got[t] != got[0]) { ++failures; break; }
    printf("%-52s %s\n", "3. 16 threads racing the first construction",
           failures ? "FAIL" : "identical sequences");

    // ---- 1. the table is the recurrence
    PyRandom probe(1);                          // forces the table to exist
    uint32_t ref[624];
    ref[0] = 19650218u;
    for (int i = 1; i < 624; ++i)
        ref[i] = 1812433253u * (ref[i - 1] ^ (ref[i - 1] >> 30))
                 + static_cast<uint32_t>(i);
    bool same = true;
    for (int i = 0; i < 624; ++i)
        if (PyRandom::genrand_base()[i] != ref[i]) { same = false; break; }
    if (!same) ++failures;
    printf("%-52s %s\n", "1. precomputed table vs init_genrand(19650218)",
           same ? "624/624 words identical" : "MISMATCH");

    // ---- 2. order and history do not matter
    auto first = draws(999ull, 32);
    for (int i = 0; i < 5000; ++i) (void)draws(i, 4);   // churn the process
    auto later = draws(999ull, 32);
    bool idem = first == later;
    if (!idem) ++failures;
    printf("%-52s %s\n", "2. same seed after 5,000 other generators",
           idem ? "identical draws" : "DIVERGED");

    // ---- and the episode-level version of 2: interleaved seeds
    std::vector<double> a1 = draws(7, 16), b1 = draws(8, 16);
    std::vector<double> b2 = draws(8, 16), a2 = draws(7, 16);
    bool cross = (a1 == a2) && (b1 == b2);
    if (!cross) ++failures;
    printf("%-52s %s\n", "   two seeds interleaved in both orders",
           cross ? "identical draws" : "DIVERGED");

    printf("\n%s\n", failures ? "FAILED" : "PASS: the optimisation is "
                                           "idempotent and thread-safe");
    return failures ? 1 : 0;
}
