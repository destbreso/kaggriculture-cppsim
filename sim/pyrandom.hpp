// CPython-compatible Mersenne Twister.
//
// The environment seeds `random.Random((seed * 1_000_003) ^ day)` and calls
// .random() (weed spawns) and .choice() (shop unlocks). To reproduce episodes
// bit-for-bit we must match CPython's _randommodule.c exactly: init_by_array
// seeding from the integer's 32-bit little-endian words, the 53-bit
// random() construction, and getrandbits/_randbelow rejection sampling.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace kag {

class PyRandom {
public:
    explicit PyRandom(uint64_t key) { seed(key); }

    void seed(uint64_t key) {
        // CPython: abs(n) -> array of 32-bit words, little endian. A zero key
        // still yields a one-word array [0].
        uint32_t words[3];
        int n = 0;
        if (key == 0) {
            words[0] = 0;
            n = 1;
        } else {
            while (key) {
                words[n++] = static_cast<uint32_t>(key & 0xffffffffu);
                key >>= 32;
            }
        }
        init_by_array(words, n);
    }

    uint32_t genrand_uint32() {
        uint32_t y;
        if (index_ >= N) {
            for (int k = 0; k < N - M; ++k) {
                y = (mt_[k] & UPPER) | (mt_[k + 1] & LOWER);
                mt_[k] = mt_[k + M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            for (int k = N - M; k < N - 1; ++k) {
                y = (mt_[k] & UPPER) | (mt_[k + 1] & LOWER);
                mt_[k] = mt_[k + (M - N)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            y = (mt_[N - 1] & UPPER) | (mt_[0] & LOWER);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            index_ = 0;
        }
        y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // random.random(): 53 bits of precision from two 32-bit draws.
    double random() {
        uint32_t a = genrand_uint32() >> 5;
        uint32_t b = genrand_uint32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // random.getrandbits(k) for 0 < k <= 32.
    uint32_t getrandbits(int k) { return genrand_uint32() >> (32 - k); }

    // Random._randbelow_with_getrandbits(n), n > 0.
    uint32_t randbelow(uint32_t n) {
        int k = 0;
        for (uint32_t v = n; v; v >>= 1) ++k;   // n.bit_length()
        uint32_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    // random.choice(seq) -> index
    uint32_t choice_index(uint32_t len) { return randbelow(len); }

private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER = 0x80000000u;
    static constexpr uint32_t LOWER = 0x7fffffffu;

    uint32_t mt_[N];
    int index_ = N;

    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (int i = 1; i < N; ++i)
            mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
        index_ = N;
    }

    // The constant half of the seeding. init_genrand(19650218) fills the
    // same 624 words on every construction, so it is a table wearing a
    // loop's clothes: computed once per process and copied thereafter.
    // Provably identical output, since it is the same recurrence with the
    // same constant. It matters because the engine builds a fresh
    // generator every in-game day, 30 times an episode, and this is a
    // third of that cost.
    // public so tools/test_rng_shared.cpp can check the table against the
    // recurrence it replaces, which is the only way to know it is right
    // rather than merely fast.
public:
    static const uint32_t* genrand_base() {
        static const std::vector<uint32_t> base = [] {
            std::vector<uint32_t> m(N);
            m[0] = 19650218u;
            for (int i = 1; i < N; ++i)
                m[i] = 1812433253u * (m[i - 1] ^ (m[i - 1] >> 30))
                       + static_cast<uint32_t>(i);
            return m;
        }();
        return base.data();
    }

private:
    void init_by_array(const uint32_t* key, int keylen) {
        std::memcpy(mt_, genrand_base(), sizeof(uint32_t) * N);
        index_ = N;
        int i = 1, j = 0;
        int k = (N > keylen) ? N : keylen;
        for (; k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) + key[j] + static_cast<uint32_t>(j);
            ++i; ++j;
            if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
            if (j >= keylen) j = 0;
        }
        for (k = N - 1; k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) - static_cast<uint32_t>(i);
            ++i;
            if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
        }
        mt_[0] = 0x80000000u;
        index_ = N;
    }
};

}  // namespace kag