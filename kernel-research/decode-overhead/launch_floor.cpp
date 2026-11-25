// Launch floor of the HIP runtime: what one dispatch costs when the kernel itself does (almost) nothing.
//   hipcc -O2 -o launch_floor launch_floor.cpp && ./launch_floor [n_kernels=2600] [chunk=512] [seconds=12]
// Modes, each timed per "step" of n_kernels launches followed by one stream synchronize:
//   direct       hipLaunchKernel for every kernel
//   graph        the step captured once as ceil(n/chunk) HIP graphs, then replayed with hipGraphLaunch
//   sync-each    hipLaunchKernel + hipStreamSynchronize per kernel (what GGML_CUDA_OP_STATS=2 adds to every op)
// It prints host enqueue time (launch calls return) and total time (after the synchronize) per step, median of the
// steps in the last third of the run (the CPU clock drops after a few seconds of GPU load on Strix Halo).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(1); } } while (0)

// a chain: every launch reads what the previous one wrote, like the glue ops of a decode step
__global__ void tiny(const float * src, float * dst, const float a, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] * a + 1.0f;
    }
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct stat_t { std::vector<double> enq, tot; };

static void report(const char * name, stat_t & s, int n) {
    const size_t from = s.tot.size() * 2 / 3;
    std::vector<double> e(s.enq.begin() + from, s.enq.end()), t(s.tot.begin() + from, s.tot.end());
    std::sort(e.begin(), e.end());
    std::sort(t.begin(), t.end());
    const double em = e[e.size() / 2], tm = t[t.size() / 2];
    printf("%-10s steps %6zu  enqueue %8.1f us (%5.2f us/kernel)  total %8.1f us (%5.2f us/kernel)\n",
           name, s.tot.size(), em, em / n, tm, tm / n);
}

int main(int argc, char ** argv) {
    const int    n       = argc > 1 ? atoi(argv[1]) : 2600;
    const int    chunk   = argc > 2 ? atoi(argv[2]) : 512;
    const double seconds = argc > 3 ? atof(argv[3]) : 12.0;
    const int    n_elem  = argc > 4 ? atoi(argv[4]) : 1024; // elements each kernel touches

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));
    float * a, * b;
    CHECK(hipMalloc(&a, n_elem * sizeof(float)));
    CHECK(hipMalloc(&b, n_elem * sizeof(float)));
    CHECK(hipMemset(a, 0, n_elem * sizeof(float)));
    CHECK(hipMemset(b, 0, n_elem * sizeof(float)));

    const dim3 grid((n_elem + 255) / 256), block(256);
    const auto launch_all = [&](int from, int to, bool sync_each) {
        for (int i = from; i < to; ++i) {
            tiny<<<grid, block, 0, stream>>>(i % 2 ? b : a, i % 2 ? a : b, 0.5f, n_elem);
            if (sync_each) {
                CHECK(hipStreamSynchronize(stream));
            }
        }
    };

    // warm up
    launch_all(0, n, false);
    CHECK(hipStreamSynchronize(stream));

    stat_t direct, graph, sync_each;

    std::vector<hipGraphExec_t> execs;
    size_t n_graph_nodes = 0;
    double t_capture = 0, t_instantiate = 0;
    for (int from = 0; from < n; from += chunk) {
        hipGraph_t g;
        double t0 = now_us();
        CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
        launch_all(from, std::min(n, from + chunk), false);
        CHECK(hipStreamEndCapture(stream, &g));
        double t1 = now_us();
        size_t nn = 0;
        CHECK(hipGraphGetNodes(g, nullptr, &nn));
        n_graph_nodes += nn;
        hipGraphExec_t e;
        CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
        double t2 = now_us();
        CHECK(hipGraphDestroy(g));
        execs.push_back(e);
        t_capture += t1 - t0;
        t_instantiate += t2 - t1;
    }
    printf("n_kernels %d chunk %d elements %d: %zu graphs, %zu graph nodes, capture %.1f us, instantiate %.1f us\n",
           n, chunk, n_elem, execs.size(), n_graph_nodes, t_capture, t_instantiate);

    const double t_end = now_us() + seconds * 1e6;
    int round = 0;
    while (now_us() < t_end) {
        for (int mode = 0; mode < 3; ++mode) {
            if (mode == 2 && round % 8 != 0) {
                continue; // sync-each is slow, sample it less often
            }
            const double t0 = now_us();
            if (mode == 0) {
                launch_all(0, n, false);
            } else if (mode == 1) {
                for (hipGraphExec_t e : execs) {
                    CHECK(hipGraphLaunch(e, stream));
                }
            } else {
                launch_all(0, n, true);
            }
            const double t1 = now_us();
            CHECK(hipStreamSynchronize(stream));
            const double t2 = now_us();
            stat_t & s = mode == 0 ? direct : mode == 1 ? graph : sync_each;
            s.enq.push_back(t1 - t0);
            s.tot.push_back(t2 - t0);
        }
        round++;
    }
    report("direct", direct, n);
    report("graph", graph, n);
    report("sync-each", sync_each, n);

    for (hipGraphExec_t e : execs) {
        CHECK(hipGraphExecDestroy(e));
    }
    CHECK(hipFree(a));
    CHECK(hipFree(b));
    CHECK(hipStreamDestroy(stream));
    return 0;
}
