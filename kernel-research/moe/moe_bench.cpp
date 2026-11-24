// moe_bench: time one MUL_MAT_ID (MoE expert matmul) at Inkling 2 Mini's shapes through the ggml backend.
//
// Links against build-rocm-mix's libggml, so it times exactly the kernel path llama.cpp takes for this op, with
// synthetic weights; no model is loaded. Reports ms per op, effective TFLOPS and the weight-read bandwidth the time
// implies. --check runs the same op on the CPU backend and prints the NMSE.
//
//   build: see build.sh;  run under quiet.sh for timings.
//   e.g.   moe_bench --type iq4_nl --m 2304 --k 2304 --n 512 --bcast 1      (gate+up fused tensor, broadcast x)
//          moe_bench --type iq4_nl --m 2304 --k 1152 --n 512 --bcast 0      (down projection, per-expert x)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct opts {
    std::string type = "iq4_nl";
    int64_t m = 2304, n = 512, k = 2304;
    int nmat = 128, nused = 8;
    bool bcast = true;
    double skew = 0.0;      // expert popularity ~ (e+1)^-skew; 0 = uniform
    int reps = 20, warm = 3;
    int graph_reps = 4;     // copies of the op per graph
    bool check = false;
    unsigned seed = 1;
    std::string backend = "ROCm0";
    bool quiet = false;
};

static ggml_type type_by_name(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * nm = ggml_type_name((ggml_type) t);
        if (nm && s == nm) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s.c_str());
    exit(1);
}

struct graph_set {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * as = nullptr, * ids = nullptr, * b = nullptr, * out = nullptr;
    ggml_cgraph * gf = nullptr;
};

// graph_reps independent copies of the op in one graph (own outputs, shared inputs): the GPU runs them back to back
// like the 44 layers of a ubatch do, without a host sync and pool bookkeeping between them.
static graph_set build(const opts & o, ggml_type ta, ggml_backend_t backend, int graph_reps = 1) {
    graph_set g;
    ggml_init_params ip = { ggml_tensor_overhead()*(16 + 2*graph_reps) + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.as  = ggml_new_tensor_3d(g.ctx, ta, o.k, o.m, o.nmat);
    g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, o.nused, o.n);
    g.b   = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, o.k, o.bcast ? 1 : o.nused, o.n);
    g.gf  = ggml_new_graph(g.ctx);
    for (int r = 0; r < graph_reps; ++r) {
        g.out = ggml_mul_mat_id(g.ctx, g.as, g.b, g.ids);
        ggml_build_forward_expand(g.gf, g.out);
    }
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (!g.buf) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return g;
}

int main(int argc, char ** argv) {
    opts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if      (a == "--type")    o.type    = next();
        else if (a == "--m")       o.m       = atoll(next());
        else if (a == "--n")       o.n       = atoll(next());
        else if (a == "--k")       o.k       = atoll(next());
        else if (a == "--nmat")    o.nmat    = atoi(next());
        else if (a == "--nused")   o.nused   = atoi(next());
        else if (a == "--bcast")   o.bcast   = atoi(next()) != 0;
        else if (a == "--skew")    o.skew    = atof(next());
        else if (a == "--reps")    o.reps    = atoi(next());
        else if (a == "--warm")    o.warm    = atoi(next());
        else if (a == "--graph-reps") o.graph_reps = atoi(next());
        else if (a == "--seed")    o.seed    = (unsigned) atoi(next());
        else if (a == "--backend") o.backend = next();
        else if (a == "--check")   o.check   = true;
        else if (a == "--quiet")   o.quiet   = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    const ggml_type ta = type_by_name(o.type);

    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name(o.backend.c_str(), nullptr);
    if (!be) { fprintf(stderr, "backend %s not found\n", o.backend.c_str()); return 1; }

    graph_set g = build(o, ta, be, o.graph_reps);

    std::mt19937 rng(o.seed);
    // weights: a few distinct random expert matrices, quantized once each and cycled over the experts
    {
        const int nvar = std::min(o.nmat, 4);
        const size_t row_bytes = ggml_row_size(ta, o.k);
        std::vector<float> src((size_t) o.m * o.k);
        std::vector<std::vector<uint8_t>> q(nvar, std::vector<uint8_t>(row_bytes * o.m));
        std::uniform_real_distribution<float> U(-1.0f, 1.0f);
        for (int v = 0; v < nvar; ++v) {
            for (auto & x : src) x = U(rng);
            if (ggml_is_quantized(ta)) {
                ggml_quantize_chunk(ta, src.data(), q[v].data(), 0, o.m, o.k, nullptr);
            } else if (ta == GGML_TYPE_F32) {
                memcpy(q[v].data(), src.data(), q[v].size());
            } else if (ta == GGML_TYPE_F16) {
                ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) q[v].data(), src.size());
            } else if (ta == GGML_TYPE_BF16) {
                ggml_fp32_to_bf16_row(src.data(), (ggml_bf16_t *) q[v].data(), src.size());
            } else { fprintf(stderr, "unsupported type\n"); return 1; }
        }
        for (int e = 0; e < o.nmat; ++e) {
            ggml_backend_tensor_set(g.as, q[e % nvar].data(), (size_t) e * g.as->nb[2], q[0].size());
        }
    }
    // activations
    std::vector<float> bh(ggml_nelements(g.b));
    { std::uniform_real_distribution<float> U(-1.0f, 1.0f); for (auto & x : bh) x = U(rng); }
    ggml_backend_tensor_set(g.b, bh.data(), 0, bh.size() * sizeof(float));
    // routing: nused distinct experts per token, expert e with weight (e+1)^-skew
    std::vector<int32_t> idh((size_t) o.nused * o.n);
    std::vector<int> load(o.nmat, 0);
    {
        std::vector<double> w(o.nmat);
        for (int e = 0; e < o.nmat; ++e) w[e] = std::pow(e + 1.0, -o.skew);
        std::discrete_distribution<int> D(w.begin(), w.end());
        for (int64_t t = 0; t < o.n; ++t) {
            int got = 0;
            while (got < o.nused) {
                const int e = D(rng);
                bool dup = false;
                for (int j = 0; j < got; ++j) dup |= idh[t*o.nused + j] == e;
                if (!dup) { idh[t*o.nused + got++] = e; load[e]++; }
            }
        }
    }
    ggml_backend_tensor_set(g.ids, idh.data(), 0, idh.size() * sizeof(int32_t));
    int lmax = 0, lnz = 0; double tiles32 = 0;
    for (int e = 0; e < o.nmat; ++e) { lmax = std::max(lmax, load[e]); lnz += load[e] > 0; tiles32 += (load[e] + 31) / 32; }

    for (int i = 0; i < o.warm; ++i) ggml_backend_graph_compute(be, g.gf);
    ggml_backend_synchronize(be);
    std::vector<double> ts;
    for (int i = 0; i < o.reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, g.gf);     // synchronous for the CUDA/HIP backend
        auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / o.graph_reps);
    }
    std::sort(ts.begin(), ts.end());
    const double med = ts[ts.size()/2], mn = ts.front();
    const double flops   = 2.0 * o.m * o.k * o.n * o.nused;
    const double wbytes  = (double) ggml_nbytes(g.as);             // whole expert tensor (all experts touched at n=512)
    const double wactive = wbytes * lnz / o.nmat;                   // experts that have at least one token
    const double obytes  = (double) ggml_nbytes(g.out) + ggml_nbytes(g.b);
    printf("[x%d] %-7s m=%lld k=%lld n=%lld nused=%d nmat=%d bcast=%d skew=%.2f | med %.3f ms (min %.3f) | %.2f TFLOPS | weights %.0f MB -> %.0f GB/s (active %.0f GB/s), +io %.0f MB | load max %d, experts hit %d, 32-col tiles %.0f (ideal %.0f)\n",
           o.graph_reps, o.type.c_str(), (long long) o.m, (long long) o.k, (long long) o.n, o.nused, o.nmat, (int) o.bcast, o.skew,
           med, mn, flops / (med * 1e-3) / 1e12, wbytes / 1e6, wbytes / (med * 1e-3) / 1e9, wactive / (med * 1e-3) / 1e9,
           obytes / 1e6, lmax, lnz, tiles32, (double) o.n * o.nused / 32);

    if (o.check) {
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        graph_set c = build(o, ta, cpu);
        std::vector<uint8_t> tmp(ggml_nbytes(g.as));
        ggml_backend_tensor_get(g.as, tmp.data(), 0, tmp.size());
        ggml_backend_tensor_set(c.as, tmp.data(), 0, tmp.size());
        ggml_backend_tensor_set(c.b, bh.data(), 0, bh.size() * sizeof(float));
        ggml_backend_tensor_set(c.ids, idh.data(), 0, idh.size() * sizeof(int32_t));
        ggml_backend_graph_compute(cpu, c.gf);
        std::vector<float> og(ggml_nelements(g.out)), oc(ggml_nelements(c.out));
        ggml_backend_tensor_get(g.out, og.data(), 0, og.size() * sizeof(float));
        ggml_backend_tensor_get(c.out, oc.data(), 0, oc.size() * sizeof(float));
        double num = 0, den = 0, maxd = 0; size_t nbad = 0;
        for (size_t i = 0; i < og.size(); ++i) {
            const double d = og[i] - oc[i];
            num += d*d; den += (double) oc[i]*oc[i]; maxd = std::max(maxd, std::fabs(d));
            nbad += !std::isfinite(og[i]);
        }
        printf("check vs CPU: NMSE %.3e (test-backend-ops limit 5e-4), max |diff| %.4f, non-finite %zu\n", num / den, maxd, nbad);
        ggml_backend_buffer_free(c.buf); ggml_free(c.ctx); ggml_backend_free(cpu);
    }
    ggml_backend_buffer_free(g.buf); ggml_free(g.ctx); ggml_backend_free(be);
    return 0;
}
