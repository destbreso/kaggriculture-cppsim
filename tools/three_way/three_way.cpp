// Three engines in one process, measured INTERLEAVED so drift hits all
// three equally. A = nikital7's pristine port (1.32.6, unmodified),
// B = ours before this iteration (1.32.7 patched, no parity shortcut),
// C = ours after. Same loop, same episode count, same seeds.
#include "orig_sim.hpp"
#include "sim_noparity.hpp"
#include "sim.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>          // getloadavg
#include <cstring>
using namespace std::chrono;
#define TIMER(NS, NAME) static double NAME(int N){ NS::Action e; e.clear(); \
  volatile double s=0; auto t0=high_resolution_clock::now(); \
  for(int i=0;i<N;++i){NS::Config c;c.seed=i;NS::Sim sim(c); \
    while(!sim.st.done)sim.step(e,e); s+=sim.reward(0);} \
  auto t1=high_resolution_clock::now(); (void)s; \
  return duration<double>(t1-t0).count()/N*1e6;}
TIMER(kag_orig, tA)
TIMER(kag_np,   tB)
TIMER(kag,      tC)
static void report(const char* n, std::vector<double>& v, double base){
  std::sort(v.begin(), v.end());
  double med=v[v.size()/2];
  printf("%-42s %8.1f us   %8.0f eps/sec   spread %5.1f%%   %5.2fx\n",
         n, med, 1e6/med, 100*(v.back()-v.front())/med, base/med);
}
int main(int argc, char** argv){
  int N = argc>1 ? atoi(argv[1]) : 1500, R = argc>2 ? atoi(argv[2]) : 9;
  // The guard the rest of this work runs under, and the stamp that lets a
  // reader tell. This file carried the headline 1.82x for a day without
  // recording the conditions it was taken in, which is how a contaminated
  // number survives review: nothing in the artifact says to doubt it.
  double la[3] = {0, 0, 0};
  getloadavg(la, 3);
  double max_load = argc > 3 ? atof(argv[3]) : 1.5;
  bool forced = (argc > 4 && strcmp(argv[4], "--force") == 0);
  printf("load average %.2f (threshold %.2f)\n", la[0], max_load);
  if (la[0] > max_load && !forced) {
    printf("REFUSING: throughput measured here is not a property of the code.\n");
    return 2;
  }
  tA(150); tB(150); tC(150);                          // warm all three
  std::vector<double> A,B,C;
  for(int r=0;r<R;++r){ A.push_back(tA(N)); B.push_back(tB(N)); C.push_back(tC(N)); }
  std::vector<double> Bc=B; std::sort(Bc.begin(),Bc.end()); double base=Bc[Bc.size()/2];
  printf("\n%d episodes x %d interleaved repetitions, single thread\n\n", N, R);
  report("A  nikital7 pristine port (1.32.6)", A, base);
  report("B  ours before this iteration (1.32.7)", B, base);
  report("C  ours after  (parity shortcut)", C, base);
  FILE* f = fopen("/tmp/bench3/three.json", "w");
  fprintf(f, "{\n \"episodes\": %d,\n \"reps\": %d,\n \"load_average\": %.4f,\n \"forced\": %s,\n", N, R, la[0], forced ? "true" : "false");
  const char* names[3] = {"orig_pristine", "ours_before", "ours_after"};
  std::vector<double>* vs[3] = {&A, &B, &C};
  for (int i = 0; i < 3; ++i) {
    std::sort(vs[i]->begin(), vs[i]->end());
    fprintf(f, " \"%s\": {\"median_us\": %.3f, \"min_us\": %.3f, "
               "\"max_us\": %.3f, \"samples\": [", names[i],
            (*vs[i])[vs[i]->size()/2], vs[i]->front(), vs[i]->back());
    for (size_t k = 0; k < vs[i]->size(); ++k)
      fprintf(f, "%s%.3f", k ? ", " : "", (*vs[i])[k]);
    fprintf(f, "]}%s\n", i < 2 ? "," : "");
  }
  fprintf(f, "}\n"); fclose(f);
  printf("\nwritten /tmp/bench3/three.json\n");
  return 0;
}
