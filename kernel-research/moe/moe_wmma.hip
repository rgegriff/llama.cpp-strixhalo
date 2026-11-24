// moe_wmma: standalone harness for the dedicated IQ4_NL MUL_MAT_ID WMMA kernel (moe_wmma_kernel.cuh), next to
// the MMQ path it is meant to replace (same inputs, same ids helper and activation quantization, same data generation
// as moe_mmq.hip). --kernel wmma|mmq picks the matmul, --check compares against a host dequant+dot reference, --cmp runs
// both kernels and reports the GPU-vs-GPU difference.
//
// It compiles the kernels from the copies in mmq/ (mmq.cuh, the RDNA3.5 config table, the tile loaders and vec dots,
// quantize.cu) against the rest of the ggml-cuda headers, and re-implements the host side of
// ggml_cuda_mul_mat_q's ids branch (mmq.cu) on raw device buffers: ids helper -> src1 quantization -> mul_mat_q.
// Everything else (input generation, checking) is self-contained; no model, no libggml.
//
//   moe_wmma --kernel wmma --m 2304 --k 2304 --n 512 --bcast 1 [--skew 0] [--reps 20] [--check] [--cmp]
#include <hip/hip_runtime.h>

#include "mmq/mmq.cuh"      // the local copy; pulls common.cuh etc. from ggml/src/ggml-cuda via -I
#include "mmq/quantize.cu"  // local copy (its "mmq.cuh" must be the local one)
#include "mmid.cu"          // original (only needs common.cuh/mmid.cuh)
#include "moe_wmma_kernel.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// ---- stubs for the ggml-cuda runtime symbols that the included code references ---------------------------------
static ggml_cuda_device_info g_info;
const ggml_cuda_device_info & ggml_cuda_info() { return g_info; }
int  ggml_cuda_get_device() { return 0; }
void ggml_cuda_set_device(int) {}
[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "HIP error: %s\n  %s in %s:%d: %s\n", msg, stmt, file, line, func);
    exit(1);
}

extern "C" [[noreturn]] void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort at %s:%d: %s\n", file, line, fmt);
    abort();
}

#define HIP_OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---- host-side launcher: the tiled branch of launch_mul_mat_q (no fusion, no stream-k on RDNA3.5) ---------------
template <ggml_type type, int J>
static void launch_tiled(const mmq_args & args, cudaStream_t stream, int * nblocks_out) {
    constexpr bool fallback = false;
    constexpr ggml_prec prec = GGML_PREC_Q8;
    const int cc = g_info.devices[0].cc;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec);
    if (config.type == GGML_TYPE_COUNT) { fprintf(stderr, "no config for J=%d\n", J); exit(1); }
    const int nwarps = config.nthreads / 32;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);
    const dim3 block_dims(32, nwarps, 1);
    static bool raised = false;
    if (!raised) { HIP_OK(hipFuncSetAttribute((const void *) mul_mat_q<type, J, fallback, prec, false>, hipFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared)); raised = true; }

    const int nti  = (args.nrows_x + config.I - 1) / config.I;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 grid(nti, args.ntj, ntzw);   // rows innermost for MoE on HIP
    *nblocks_out = nti * args.ntj * ntzw;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const uint3 ntj_fd             = init_fastdiv_values(args.ntj);
    const uint3 nchannels_y_fd     = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd      = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd   = init_fastdiv_values(1);
    const uint3 sample_ratio_fd    = init_fastdiv_values(1);

    mul_mat_q<type, J, fallback, prec, false><<<grid, block_dims, nbytes_shared, stream>>>
        (args.x, nullptr, args.y, args.ids_dst, args.tile_spec, args.dst, nullptr,
         nullptr, GGML_GLU_OP_SWIGLU, 0.0f, nullptr, nullptr,
         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
         ntj_fd);
}

template <ggml_type type>
static void launch_J_t(int J, const mmq_args & args, cudaStream_t stream, int * nblocks) {
    switch (J) {
        case  16: launch_tiled<type,  16>(args, stream, nblocks); break;
        case  32: launch_tiled<type,  32>(args, stream, nblocks); break;
        case  48: launch_tiled<type,  48>(args, stream, nblocks); break;
        case  64: launch_tiled<type,  64>(args, stream, nblocks); break;
        case  96: launch_tiled<type,  96>(args, stream, nblocks); break;
        case 128: launch_tiled<type, 128>(args, stream, nblocks); break;
        default: fprintf(stderr, "J=%d not instantiated\n", J); exit(1);
    }
}
static void launch_J(ggml_type type, int J, const mmq_args & args, cudaStream_t stream, int * nblocks) {
    switch (type) {
        case GGML_TYPE_IQ4_NL: launch_J_t<GGML_TYPE_IQ4_NL>(J, args, stream, nblocks); break;
        case GGML_TYPE_Q4_0:   launch_J_t<GGML_TYPE_Q4_0>  (J, args, stream, nblocks); break;
        case GGML_TYPE_Q4_K:   launch_J_t<GGML_TYPE_Q4_K>  (J, args, stream, nblocks); break;
        case GGML_TYPE_Q8_0:   launch_J_t<GGML_TYPE_Q8_0>  (J, args, stream, nblocks); break;
        default: fprintf(stderr, "type not instantiated\n"); exit(1);
    }
}

// the J choice of ggml_cuda_mul_mat_q: fewest column tiles for the per-expert average column count
static int pick_J(ggml_type type, int64_t ncols, int cc, size_t smpbo) {
    int J_best = 0, ntiles_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, J, false, cc, GGML_PREC_Q8);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        const int nt = (ncols + c.J - 1) / c.J;
        if (nt < ntiles_best) { J_best = J; ntiles_best = nt; }
    }
    return J_best;
}

static const int8_t kv_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static float fp16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m | 0x400), (int) e - 25);
    return s ? -v : v;
}
static uint16_t f32_to_fp16(float f) {   // round-to-nearest, normal range only (we only need small positive scales)
    int e; const float m = std::frexp(f, &e);  // f = m * 2^e, m in [0.5, 1)
    const int ee = e - 1 + 15;
    const uint32_t mant = (uint32_t) std::lround((m * 2.0f - 1.0f) * 1024.0f) & 0x3ff;
    return (uint16_t) ((ee << 10) | mant);
}

int main(int argc, char ** argv) {
    std::string type_s = "iq4_nl";
    int64_t m = 2304, n = 512, k = 2304;
    int nmat = 128, nused = 8, J_force = 0, reps = 20, warm = 3;
    bool bcast = true, check = false, cmp = false, plan = false;
    std::string kernel = "wmma";
    double skew = 0;
    unsigned seed = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if      (a == "--type")  type_s = next();
        else if (a == "--m")     m = atoll(next());
        else if (a == "--n")     n = atoll(next());
        else if (a == "--k")     k = atoll(next());
        else if (a == "--nmat")  nmat = atoi(next());
        else if (a == "--nused") nused = atoi(next());
        else if (a == "--bcast") bcast = atoi(next()) != 0;
        else if (a == "--J")     J_force = atoi(next());
        else if (a == "--skew")  skew = atof(next());
        else if (a == "--reps")  reps = atoi(next());
        else if (a == "--warm")  warm = atoi(next());
        else if (a == "--seed")  seed = (unsigned) atoi(next());
        else if (a == "--check") check = true;
        else if (a == "--cmp")   cmp = true;
        else if (a == "--plan")  plan = true;
        else if (a == "--kernel") kernel = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    ggml_type type = GGML_TYPE_COUNT; int qk = 0; size_t bs = 0;   // no libggml here: the four instantiated types by hand
    if      (type_s == "iq4_nl") { type = GGML_TYPE_IQ4_NL; qk = QK4_NL; bs = sizeof(block_iq4_nl); }
    else if (type_s == "q4_0")   { type = GGML_TYPE_Q4_0;   qk = QK4_0;  bs = sizeof(block_q4_0); }
    else if (type_s == "q4_K")   { type = GGML_TYPE_Q4_K;   qk = QK_K;   bs = sizeof(block_q4_K); }
    else if (type_s == "q8_0")   { type = GGML_TYPE_Q8_0;   qk = QK8_0;  bs = sizeof(block_q8_0); }
    else { fprintf(stderr, "type %s not instantiated\n", type_s.c_str()); return 1; }
    if (k % qk) { fprintf(stderr, "k must be a multiple of %d for %s\n", qk, type_s.c_str()); return 1; }
    const bool is_iq4 = type == GGML_TYPE_IQ4_NL;

    // device info stub
    {
        hipDeviceProp_t prop; HIP_OK(hipGetDeviceProperties(&prop, 0));
        auto & d = g_info.devices[0];
        d.cc = GGML_CUDA_CC_OFFSET_AMD + 0x1151; d.nsm = prop.multiProcessorCount; d.smpb = prop.sharedMemPerBlock;
        d.smpbo = prop.sharedMemPerBlock; d.warp_size = 32; d.integrated = true;
        g_info.device_count = 1; g_info.physical_device_count = 1;
        printf("device %s, %d CUs, %zu B LDS per block\n", prop.gcnArchName, d.nsm, d.smpb);
    }
    const int cc = g_info.devices[0].cc;

    std::mt19937 rng(seed);
    // weights: iq4_nl gets random nibbles with scales ~ U(0.5, 1.5)/128 (values O(1), checkable on the host);
    // the other types get random bytes (timing only, --check is iq4_nl-only)
    const int64_t kb = k / qk;
    std::vector<uint8_t> hx((size_t) nmat * m * kb * bs);
    {
        std::uniform_int_distribution<int> B(0, 255);
        std::uniform_real_distribution<float> S(0.5f / 127, 1.5f / 127);
        if (is_iq4) {
            for (int64_t row = 0; row < (int64_t) nmat * m; ++row) {
                for (int64_t b = 0; b < kb; ++b) {
                    uint8_t * blk = hx.data() + (row * kb + b) * bs;
                    const uint16_t d = f32_to_fp16(S(rng));
                    blk[0] = d & 0xff; blk[1] = d >> 8;
                    for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) B(rng);
                }
            }
        } else {
            for (auto & v : hx) v = (uint8_t) B(rng);
            if (check) { fprintf(stderr, "--check is iq4_nl-only\n"); check = false; }
        }
    }
    const int64_t ne11 = bcast ? 1 : nused;
    std::vector<float> hy((size_t) k * ne11 * n);
    { std::uniform_real_distribution<float> U(-1.0f, 1.0f); for (auto & v : hy) v = U(rng); }
    std::vector<int32_t> hids((size_t) nused * n);
    std::vector<int> load(nmat, 0);
    {
        std::vector<double> w(nmat);
        for (int e = 0; e < nmat; ++e) w[e] = std::pow(e + 1.0, -skew);
        std::discrete_distribution<int> D(w.begin(), w.end());
        for (int64_t t = 0; t < n; ++t) {
            int got = 0;
            while (got < nused) {
                const int e = D(rng);
                bool dup = false;
                for (int j = 0; j < got; ++j) dup |= hids[t*nused + j] == e;
                if (!dup) { hids[t*nused + got++] = e; load[e]++; }
            }
        }
    }

    // device buffers
    char * dx; float * dy; int32_t * dids; float * ddst;
    HIP_OK(hipMalloc(&dx, hx.size()));   HIP_OK(hipMemcpy(dx, hx.data(), hx.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dy, hy.size()*4)); HIP_OK(hipMemcpy(dy, hy.data(), hy.size()*4, hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dids, hids.size()*4)); HIP_OK(hipMemcpy(dids, hids.data(), hids.size()*4, hipMemcpyHostToDevice));
    const size_t ndst = (size_t) m * nused * n;
    HIP_OK(hipMalloc(&ddst, ndst*4)); HIP_OK(hipMemset(ddst, 0, ndst*4));

    // ---- host side of ggml_cuda_mul_mat_q, ids branch ----
    const int64_t ne_get_rows = n * nused;
    const int64_t ncols_opt   = (ne_get_rows + nmat - 1) / nmat;
    const bool use_wmma = kernel == "wmma";
    if (use_wmma && (type != GGML_TYPE_IQ4_NL || m % moe_wmma::BM || k % moe_wmma::BK)) { fprintf(stderr, "wmma kernel: iq4_nl, m%%128==0, k%%128==0 only\n"); return 1; }
    const int J = use_wmma ? moe_wmma::BN : (J_force ? J_force : pick_J(type, ncols_opt, cc, g_info.devices[0].smpbo));
    if (!use_wmma && kernel != "mmq") { fprintf(stderr, "--kernel wmma|mmq\n"); return 1; }
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, cc, GGML_PREC_Q8);
    const int ntj = (ne_get_rows + J - 1) / J + nmat;

    int32_t * ids_src1; int32_t * ids_dst; ggml_cuda_mm_id_tile_spec * tile_spec;
    HIP_OK(hipMalloc(&ids_src1, ne_get_rows*4));
    HIP_OK(hipMalloc(&ids_dst, (ne_get_rows + J - 1)*4));
    HIP_OK(hipMalloc(&tile_spec, ntj*sizeof(ggml_cuda_mm_id_tile_spec)));
    const bool dedup_bcast = ne11 == 1 && nused > 1;

    const int64_t ne10_padded = GGML_PAD(k, MATRIX_ROW_PADDING);
    const size_t src1_load_chunk = config.nthreads * sizeof(int);
    const size_t src1_q8_1_padding = ((J * sizeof(block_q8_1_mmq) + src1_load_chunk - 1) / src1_load_chunk) * src1_load_chunk;
    const size_t nbytes_src1_q8_1 = ne_get_rows * ne10_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ + src1_q8_1_padding;
    char * dyq; HIP_OK(hipMalloc(&dyq, nbytes_src1_q8_1));

    hipStream_t stream; HIP_OK(hipStreamCreate(&stream));

    const int64_t s01 = kb, s02 = m * kb;                 // in blocks
    const int64_t s1 = m, s2 = m * nused;                 // dst strides in floats
    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const mmq_args args = {
        dx, type, (const int *) dyq, ids_dst, tile_spec, ddst, nullptr,
        k, m, ne_get_rows, s01, ne_get_rows, s1,
        nmat, 1, s02, s12, s2,
        1, 1, 0, 0, 0,
        ntj, J, {} };

    auto run_prep = [&]() {
        ggml_cuda_launch_mm_ids_helper(dids, ids_src1, ids_dst, tile_spec, nmat, n, nused, ne11, J,
                                       /*si1=*/nused, /*sis1=*/ne11, dedup_bcast, stream);
        if (plan) {
            moe_wmma::plan_tiles<<<1, moe_wmma::PLAN_THREADS, 0, stream>>>(dids, (int) n, nused, nused, nmat, tile_spec, ntj);
        }
        if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(dy, ids_src1, dyq, type, k, /*stride_token=*/k*ne11, ne10_padded, n, ne_get_rows, nused, stream);
        } else {
            quantize_mmq_q8_1_cuda(dy, ids_src1, dyq, type, k, /*s11=*/k, /*s12=*/k*ne11, /*s13=*/k*ne11*n, ne10_padded, ne_get_rows, 1, 1, stream);
        }
    };
    int nblocks = 0;
    static bool lds_raised = false;
    if (!lds_raised) { HIP_OK(hipFuncSetAttribute((const void *) moe_wmma::mul_mat_id_iq4nl, hipFuncAttributeMaxDynamicSharedMemorySize, moe_wmma::LDS_BYTES)); lds_raised = true; }
    auto run_wmma = [&]() {
        const dim3 grid(m / moe_wmma::BM, ntj, 1);
        nblocks = grid.x * grid.y;
        moe_wmma::mul_mat_id_iq4nl<<<grid, moe_wmma::NTHREADS, moe_wmma::LDS_BYTES, stream>>>(
            dx, (const char *) dyq, ids_dst, tile_spec, ddst, (int) k, (int) (kb * bs), (int64_t) m * kb * bs, (int) ne_get_rows, (int) m);
    };
    auto run_mmq = [&]() { if (use_wmma) run_wmma(); else launch_J(type, J, args, stream, &nblocks); };

    for (int i = 0; i < warm; ++i) { run_prep(); run_mmq(); }
    HIP_OK(hipStreamSynchronize(stream));
    HIP_OK(hipGetLastError());

    hipEvent_t e0, e1, e2; HIP_OK(hipEventCreate(&e0)); HIP_OK(hipEventCreate(&e1)); HIP_OK(hipEventCreate(&e2));
    std::vector<float> t_all, t_mmq;
    for (int i = 0; i < reps; ++i) {
        HIP_OK(hipEventRecord(e0, stream));
        run_prep();
        HIP_OK(hipEventRecord(e1, stream));
        run_mmq();
        HIP_OK(hipEventRecord(e2, stream));
        HIP_OK(hipEventSynchronize(e2));
        float a, b; HIP_OK(hipEventElapsedTime(&a, e0, e1)); HIP_OK(hipEventElapsedTime(&b, e1, e2));
        t_all.push_back(a + b); t_mmq.push_back(b);
    }
    std::vector<ggml_cuda_mm_id_tile_spec> hts(ntj);
    HIP_OK(hipMemcpy(hts.data(), tile_spec, ntj*sizeof(ggml_cuda_mm_id_tile_spec), hipMemcpyDeviceToHost));
    double slots16 = 0, slotsJ = 0; int tiles_ne = 0;
    for (auto & t : hts) { const int nt = t.col_high - t.col_low; if (nt > 0) { ++tiles_ne; slots16 += (nt + 15) / 16 * 16; slotsJ += J; } }
    std::sort(t_all.begin(), t_all.end()); std::sort(t_mmq.begin(), t_mmq.end());
    const double med = t_all[reps/2], med_mmq = t_mmq[reps/2];
    const double flops = 2.0 * m * k * n * nused;
    int lmax = 0; double tiles = 0;
    for (int e = 0; e < nmat; ++e) { lmax = std::max(lmax, load[e]); tiles += (load[e] + J - 1) / J; }
    printf("%s %s m=%lld k=%lld n=%lld nused=%d nmat=%d bcast=%d skew=%.2f | J=%d (I=%d, %d thr) %d blocks | prep+mmq med %.3f ms, mmq alone %.3f ms (min %.3f) | %.2f TFLOPS, weights %.0f MB -> %.0f GB/s | tiles %d: J-slots x%.2f, 16-slots x%.2f%s\n",
           kernel.c_str(), type_s.c_str(), (long long) m, (long long) k, (long long) n, nused, nmat, (int) bcast, skew, J, use_wmma ? moe_wmma::BM : config.I, use_wmma ? moe_wmma::NTHREADS : config.nthreads,
           nblocks,
           med, med_mmq, (double) t_mmq.front(), flops / (med * 1e-3) / 1e12, hx.size() / 1e6, hx.size() / (med * 1e-3) / 1e9,
           tiles_ne, slotsJ / ne_get_rows, slots16 / ne_get_rows, plan ? " (expert-aligned plan)" : "");

    if (cmp && use_wmma) {
        std::vector<float> hw(ndst), hm(ndst);
        HIP_OK(hipMemcpy(hw.data(), ddst, ndst*4, hipMemcpyDeviceToHost));
        HIP_OK(hipMemset(ddst, 0, ndst*4));
        run_prep();
        launch_J(type, J, args, stream, &nblocks);   // MMQ with the same J (=BN) tiles and buffers
        HIP_OK(hipStreamSynchronize(stream));
        HIP_OK(hipMemcpy(hm.data(), ddst, ndst*4, hipMemcpyDeviceToHost));
        double num = 0, den = 0, maxd = 0; size_t nbad = 0, nex = 0;
        for (size_t i = 0; i < ndst; ++i) { const double d = hw[i] - hm[i]; num += d*d; den += (double) hm[i]*hm[i]; maxd = std::max(maxd, std::fabs(d)); nbad += !std::isfinite(hw[i]); nex += hw[i] == hm[i]; }
        printf("cmp wmma vs mmq(J=%d), all %zu outputs: NMSE %.3e, max |diff| %.3e, bit-identical %.1f%%, non-finite %zu\n", J, ndst, num / den, maxd, 100.0 * nex / ndst, nbad);
        run_prep(); run_wmma(); HIP_OK(hipStreamSynchronize(stream));   // restore the wmma result for --check
    }
    if (check) {
        std::vector<float> hd(ndst);
        HIP_OK(hipMemcpy(hd.data(), ddst, ndst*4, hipMemcpyDeviceToHost));
        // dequantize-and-dot on a sample of (token, slot, row) triples
        std::uniform_int_distribution<int64_t> T(0, n - 1), R(0, m - 1);
        std::uniform_int_distribution<int> Sl(0, nused - 1);
        double num = 0, den = 0, maxd = 0; size_t nbad = 0;
        std::vector<float> wrow(k);
        for (int s = 0; s < 4000; ++s) {
            const int64_t t = T(rng), r = R(rng); const int j = Sl(rng);
            const int e = hids[t*nused + j];
            const uint8_t * row = hx.data() + ((size_t) e * m + r) * kb * bs;
            for (int64_t b = 0; b < kb; ++b) {
                const uint8_t * blk = row + b * bs;
                const float d = fp16_to_f32(blk[0] | (blk[1] << 8));
                for (int i = 0; i < 16; ++i) { wrow[b*qk + i] = d * kv_iq4nl[blk[2+i] & 0xf]; wrow[b*qk + 16 + i] = d * kv_iq4nl[blk[2+i] >> 4]; }
            }
            const float * x = hy.data() + ((size_t) t * ne11 + (bcast ? 0 : j)) * k;
            double ref = 0; for (int64_t i = 0; i < k; ++i) ref += (double) wrow[i] * x[i];
            const float got = hd[(size_t) t * nused * m + (size_t) j * m + r];
            const double df = got - ref; num += df*df; den += ref*ref; maxd = std::max(maxd, std::fabs(df)); nbad += !std::isfinite(got);
        }
        printf("check vs host reference (4000 samples): NMSE %.3e, max |diff| %.4f, non-finite %zu  (Q8_1 activations: expect ~1e-5)\n", num / den, maxd, nbad);
    }
    return 0;
}
