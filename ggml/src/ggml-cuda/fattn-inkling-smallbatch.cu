#include "fattn-inkling.cuh"

#include <cstdlib>

// Small-batch decode attention for Inkling 2: 2..16 queries (the verify step of speculative decoding)
// over a long q8_0 or q4_0 KV cache. Ported from kernel-research/fa-smallbatch/fa_smallbatch.hip at kern 972d6052 (fa_small_wmma;
// the standalone prototype with a double-precision CPU reference, see its NOTES.md). The n_q = 1 shape stays on fattn-inkling.cu.
//
// Design: one block per (chunk of positions, group of HPB KV heads); WPH = ceil(3*NQ/16) waves per KV head, each owning 16
// (query, head) columns (col = g*NQ + qi). Per 32 keys a wave loads the 32 K rows (lane = key) and, per 16-key tile, the V bytes
// of its 8 dims x 16 keys plus the mask/index words of its query. K is dequantized to f16 in registers, Q is f16 in LDS,
// S^T = K.Q^T and O^T += V^T.P^T with the f16 WMMA (gfx11 wave32); online softmax in the WMMA output layout. Split-K + combine.
// Half-lane property (wmma_half_probe.hip): the even output rows come from the lower 16 lanes' A/B rows, the odd rows from the
// upper 16, so A = the 32 K rows as loaded gives the keys {2e} / {17 + 2e} and A = the halves exchanged gives {16 + 2e} / {1 + 2e}.
// n_q between the template sizes {2, 4, 8, 16} is padded up: the padding columns use the last query's data and are never stored.
// GGML_CUDA_FATTN_INKLING_SMALL=0 turns the path off; GGML_CUDA_FATTN_INKLING_SMALL_MIN_KV sets the shortest context it takes
// (default 32768: below that the tile kernel is fine).

#if defined(GGML_USE_HIP)

namespace {

constexpr int FS_D = 128, FS_HKV = 8, FS_G = 3, FS_HQ = FS_HKV*FS_G, FS_QK = 32, FS_NB = FS_D/FS_QK, FS_WAVE = 32;
constexpr int FS_NQ_MAX = 16;

template <int BLK> struct fs_type { static constexpr int blk = BLK; static constexpr bool q4 = BLK == 18; static constexpr int row = FS_NB*BLK; };
using fs_q8_0 = fs_type<34>;   // half d, int8 qs[32]
using fs_q4_0 = fs_type<18>;   // half d, uint8 qs[16]: low nibble = value i, high = value i+16, minus 8

typedef __attribute__((ext_vector_type(16))) _Float16 fs_halfx16;
typedef __attribute__((ext_vector_type(8)))  float    fs_floatx8;
typedef __attribute__((ext_vector_type(2)))  _Float16 fs_half2;
typedef __attribute__((ext_vector_type(2)))  uint32_t fs_u32x2;
typedef __attribute__((ext_vector_type(4)))  uint32_t fs_u32x4;
typedef __attribute__((ext_vector_type(4)))  int32_t  fs_i32x4;
typedef const __attribute__((address_space(1))) char * fs_gptr;

struct fs_params {
    const float * Q; int64_t qs_q, qs_h; float scale;        // Q element (d, qi, hq) at Q[qi*qs_q + hq*qs_h + d] (floats)
    const char * K; const char * V; int64_t nb_pos, nb_head;  // bytes
    const half * mask; int64_t mask_stride;                   // halves per query row
    const int32_t * idx; int64_t idx_stride;                  // ints per query row
    const float * table; int64_t table_stride_q; int n_bins;  // floats per query (= FS_HQ*n_bins when contiguous)
    int n_q, n_kv, chunk;
    float * part_acc; float2 * part_ml;                       // [chunk][col = hq*NQ + qi][D], [chunk][col]
};

template <typename U> static __device__ __forceinline__ U fs_gload(fs_gptr base, uint32_t off) {
    return *(const __attribute__((address_space(1))) U *) (base + off);
}
static __device__ __forceinline__ uint32_t fs_word_at(const uint32_t * w, int byte) {   // 4 bytes at a 2-byte aligned offset
    return (byte % 4 == 0) ? w[byte/4] : __builtin_amdgcn_alignbyte(w[byte/4 + 1], w[byte/4], byte % 4);
}
static __device__ __forceinline__ float fs_half_at(const uint32_t * w, int byte) {
    return __half2float(__ushort_as_half((unsigned short) (w[byte/4] >> (8*(byte % 4)))));
}
static __device__ __forceinline__ float    fs_half_bits(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short) bits)); }
static __device__ __forceinline__ uint32_t fs_h2_bits(fs_half2 v) { return __builtin_bit_cast(uint32_t, v); }
static __device__ __forceinline__ fs_half2 fs_bits_h2(uint32_t v) { return __builtin_bit_cast(fs_half2, v); }
// value of lane^16: v_permlanex16 is a VALU op (ds_bpermute would be an LDS round trip)
static __device__ __forceinline__ uint32_t fs_xchg16(uint32_t v) { return __builtin_amdgcn_permlanex16(0u, v, 0x76543210u, 0xfedcba98u, true, true); }
static __device__ __forceinline__ float    fs_xchg16(float v)    { return __uint_as_float(fs_xchg16(__float_as_uint(v))); }
static __device__ __forceinline__ uint32_t fs_launder(uint32_t v) { asm volatile("" : "+v"(v)); return v; }
// readfirstlane: the result is known-uniform (SGPR pair), opaque to LSR/LICM; the explicit global address space keeps the
// loads global_load (saddr + 32-bit lane offset), not flat
static __device__ __forceinline__ fs_gptr fs_uniform_ptr(const char * ptr) {
    const uint64_t v = (uint64_t) ptr;
    const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) v), hi = __builtin_amdgcn_readfirstlane((uint32_t) (v >> 32));
    return (fs_gptr) (((uint64_t) hi << 32) | lo);
}
// mask + indexed bias, rounded to half as fattn_indexed_bias::add does
static __device__ __forceinline__ float fs_bias(uint32_t mask_bits, int bin, const float * tbl, int n_bins) {
    const float mk = fs_half_bits(mask_bits);
    return (uint32_t) bin < (uint32_t) n_bins ? __half2float(__float2half(mk + tbl[bin])) : mk;
}
// two bytes (b_lo, b_hi) of a dword -> half2 (1024 + b_lo, 1024 + b_hi) via v_perm with the magic 0x6400 high bytes
constexpr uint32_t FS_MAGIC_H2 = 0x64006400u, FS_SEL_LO01 = 0x07010500u, FS_SEL_HI23 = 0x07030502u;

template <typename T, int NQ, int HPB>
__launch_bounds__(HPB*((FS_G*NQ + 15)/16)*FS_WAVE)
static __global__ void fs_wmma(const fs_params p) {
    constexpr int ROW = T::row, NC = FS_G*NQ, WPH = (NC + 15)/16, WAVES = HPB*WPH, NCOL = FS_HQ*NQ;
    static_assert(WAVES*16*FS_D*2 <= 65536, "Q f16 tiles exceed the LDS");
    static_assert(ROW % 16 == 8, "K row load pattern");
    const int lane = threadIdx.x % FS_WAVE, wave = __builtin_amdgcn_readfirstlane(threadIdx.x / FS_WAVE), hl = wave / WPH, w = wave % WPH;
    const int h = blockIdx.y*HPB + hl, c = blockIdx.x, k0 = c*p.chunk;
    const int nb_pos = (int) p.nb_pos;    // row pitch in bytes (fits 32 bits; lane offsets below stay 32-bit)
    const int l16 = lane & 15, hh = lane >> 4;

    __shared__ uint32_t q_f16[WAVES][16][FS_D/2];   // per wave: its 16 columns' scaled Q in f16; 16 B quads XOR-swizzled by (col & 7)

    // ---- setup: Q columns (col = g*NQ + qi) of this wave to f16, scaled, like llama.cpp's f16 kernels; qi clamped to n_q - 1
    for (int cc = 0; cc < 16; ++cc) {
        const int col = 16*w + cc;
        float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (col < NC) {
            const int g = col/NQ, qi = min(col % NQ, p.n_q - 1);
            x = ((const float4 *) (p.Q + qi*p.qs_q + (h*FS_G + g)*p.qs_h))[lane];
        }
        const fs_half2 lo = {(_Float16) (x.x*p.scale), (_Float16) (x.y*p.scale)}, hi = {(_Float16) (x.z*p.scale), (_Float16) (x.w*p.scale)};
        uint32_t * dst = &q_f16[wave][cc][4*((lane/2) ^ (cc & 7)) + 2*(lane % 2)];
        dst[0] = fs_h2_bits(lo); dst[1] = fs_h2_bits(hi);
    }
    __syncthreads();

    const int col = 16*w + l16, colc = min(col, NC - 1), qi = min(colc % NQ, p.n_q - 1), g = colc/NQ, hq = h*FS_G + g;
    const float * tbl = p.table + qi*p.table_stride_q + hq*p.n_bins;

    // V operand: lane l16 owns dims 8*l16..8*l16+7 (block vb = l16/4), 2 keys per load (lanes 0-15: even key, 16-31: odd)
    const int vb = l16/4, vo = T::q4 ? vb*T::blk + 2 + 8*((l16 % 4) % 2) : vb*T::blk + 2 + 8*(l16 % 4);
    const bool vhi = T::q4 && (l16 % 4) >= 2;
    const uint32_t koff = lane*nb_pos;
    const uint32_t moff = (uint32_t) (qi*p.mask_stride)*2u, ioff = (uint32_t) (qi*p.idx_stride)*4u;    // byte offsets
    // v_perm selectors: interleave (x = this half's keys, y = the other half's) into key order; low/high half-words
    const uint32_t sel_lo = hh ? 0x05040100u : 0x01000504u, sel_hi = hh ? 0x07060302u : 0x03020706u;
    auto interleave = [&](const uint32_t * x, const uint32_t * y, fs_halfx16 & out) {
        union { uint32_t u[8]; fs_halfx16 h; } r;
#pragma unroll
        for (int i = 0; i < 4; ++i) { r.u[2*i] = __builtin_amdgcn_perm(x[i], y[i], sel_lo); r.u[2*i + 1] = __builtin_amdgcn_perm(x[i], y[i], sel_hi); }
        out = r.h;
    };
    const fs_half2 koffs = fs_bits_h2(T::q4 ? 0xE408E408u : 0xE480E480u);     // -(1024 + 8) / -(1024 + 128)

    fs_floatx8 O[8];                           // O[t][e] = O[col][d = 8*(2e + hh) + t]
#pragma unroll
    for (int t = 0; t < 8; ++t) O[t] = (fs_floatx8) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -1e30f, l = 0.0f;                // l: this lane's 8-of-16 keys; summed with lane^16 at the end

    const int kend = min(k0 + p.chunk, p.n_kv);

    // V rows of one key tile: lane holds keys 2jj + hh (8 keys), 8 bytes of dims + the block scale each; and the lane's
    // query's mask halves / bias indices of those keys
    struct Tile { fs_u32x2 vraw[8]; uint32_t vsc[8]; uint32_t mb[8]; int bin[8]; };
    // tile 0/1 of the 32 keys at t0: this lane's keys are t0 + kofs(tile) + 2jj, kofs = 16*(hh ^ tile) + hh
    auto load_tile = [&](int t0, int tile, Tile & T_) {
        const int kofs = 16*(hh ^ tile) + hh;
        const uint32_t voff = fs_launder(kofs*nb_pos + vo), soff = fs_launder(kofs*nb_pos + vb*T::blk);
        const uint32_t mo = fs_launder(moff + 32*(hh ^ tile)), io = fs_launder(ioff + 64*(hh ^ tile));
        const char * vbase = p.V + (int64_t) t0*p.nb_pos + (int64_t) h*p.nb_head;
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {       // per-key uniform base (SALU) + one lane offset: no hoisted per-key VGPR offsets
            const fs_gptr bj = fs_uniform_ptr(vbase + 2*jj*nb_pos);
            T_.vraw[jj] = fs_gload<fs_u32x2>(bj, voff);
            T_.vsc[jj]  = fs_gload<unsigned short>(bj, soff);
        }
        const fs_gptr mp = fs_uniform_ptr((const char *) (p.mask + t0));
        const fs_u32x4 m0 = fs_gload<fs_u32x4>(mp, mo), m1 = fs_gload<fs_u32x4>(mp, mo + 16);
        const uint32_t md[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) T_.mb[e] = hh ? (md[e] >> 16) : (md[e] & 0xffffu);
        const fs_gptr ip = fs_uniform_ptr((const char *) (p.idx + t0));
#pragma unroll
        for (int i = 0; i < 4; ++i) { const fs_i32x4 v = fs_gload<fs_i32x4>(ip, io + 16*i); T_.bin[2*i] = hh ? v.y : v.x; T_.bin[2*i + 1] = hh ? v.w : v.z; }
    };
    // bias + online softmax + PV of one key tile; S holds this lane's 8 logits (keys t0 + kofs + 2e); chunk and n_kv are
    // multiples of 32, so every tile is full
    auto process_tile = [&](const Tile & T_, float * S) {
        bool uni = true;
#pragma unroll
        for (int e = 1; e < 8; ++e) uni = uni && T_.bin[e] == T_.bin[0];
        if (uni) {                             // the model's far field: one table load for the 8 keys
            const bool valid = (uint32_t) T_.bin[0] < (uint32_t) p.n_bins;
            const float tb = valid ? tbl[T_.bin[0]] : 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) { const float mkf = fs_half_bits(T_.mb[e]); S[e] += valid ? __half2float(__float2half(mkf + tb)) : mkf; }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) S[e] += fs_bias(T_.mb[e], T_.bin[e], tbl, p.n_bins);
        }
        float mt = S[0];
#pragma unroll
        for (int e = 1; e < 8; ++e) mt = fmaxf(mt, S[e]);
        mt = fmaxf(mt, fs_xchg16(mt));
        const float mn = fmaxf(m, mt), alpha = __expf(m - mn);
        m = mn;
        float pr[8], ps = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) { pr[e] = __expf(S[e] - mn); ps += pr[e]; }
        l = l*alpha + ps;
        if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
            for (int t = 0; t < 8; ++t) O[t] *= alpha;
        }
        // P as the B operand: lane holds P[col][16 keys in order]
        fs_halfx16 Pb;
        {
            uint32_t x[4], y[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) x[e2] = __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(pr[2*e2], pr[2*e2 + 1]));    // keys 4e2 + hh, 4e2 + 2 + hh
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) y[e2] = fs_xchg16(x[e2]);
            interleave(x, y, Pb);
        }
        // V as the A operand, one dim slot t at a time: lane holds V[16 keys][d = 8*l16 + t]
        float dv[8], ndv[8]; uint32_t vx[8][2];
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            dv[jj]  = fs_half_bits(T_.vsc[jj]);
            ndv[jj] = T::q4 ? -8.0f*dv[jj] : -128.0f*dv[jj];
            const fs_u32x2 v = T_.vraw[jj];
            if constexpr (!T::q4) { vx[jj][0] = v.x ^ 0x80808080u; vx[jj][1] = v.y ^ 0x80808080u; }
            else { vx[jj][0] = (vhi ? v.x >> 4 : v.x) & 0x0F0F0F0Fu; vx[jj][1] = (vhi ? v.y >> 4 : v.y) & 0x0F0F0F0Fu; }
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t x[4], y[4];
#pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) {        // keys 4j2 + hh, 4j2 + 2 + hh: v_cvt_f32_ubyte + v_fma_mixlo/hi
                fs_half2 r;
                r.x = (_Float16) fmaf((float) ((vx[2*j2][t/4] >> (8*(t % 4))) & 0xffu), dv[2*j2], ndv[2*j2]);
                r.y = (_Float16) fmaf((float) ((vx[2*j2 + 1][t/4] >> (8*(t % 4))) & 0xffu), dv[2*j2 + 1], ndv[2*j2 + 1]);
                x[j2] = fs_h2_bits(r);
            }
#pragma unroll
            for (int j2 = 0; j2 < 4; ++j2) y[j2] = fs_xchg16(x[j2]);
            fs_halfx16 Va;
            interleave(x, y, Va);
            O[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Va, Pb, O[t]);
            __builtin_amdgcn_sched_barrier(0);
        }
    };

    // K rows, lane = key (n_kv % 32 == 0: no clamping)
    uint32_t w_[ROW/4 + 1];
    auto load_K = [&](int t0) {
        const fs_gptr row = fs_uniform_ptr(p.K + (int64_t) t0*p.nb_pos + (int64_t) h*p.nb_head);
        const uint32_t ko = fs_launder(koff);
#pragma unroll
        for (int i = 0; i < ROW/16; ++i) { const fs_u32x4 v = fs_gload<fs_u32x4>(row, ko + 16*i); w_[4*i] = v.x; w_[4*i + 1] = v.y; w_[4*i + 2] = v.z; w_[4*i + 3] = v.w; }
        { const fs_u32x2 v = fs_gload<fs_u32x2>(row, ko + ROW - 8); w_[ROW/4 - 2] = v.x; w_[ROW/4 - 1] = v.y; }   // ROW % 16 == 8
        w_[ROW/4] = 0;
    };
#pragma unroll 1
    for (int t0 = k0; t0 < kend; t0 += FS_WAVE) {
        load_K(t0);
        // launder the Q swizzle so the B-operand LDS reads stay in the loop (hoisted they would cost 64 VGPRs)
        int sw = l16 & 7;
        asm volatile("" : "+v"(sw));

        // ---- S^T = K.Q^T for both key tiles: lane holds S[kt][key = 16*kt + 2e + hh][col]
        fs_floatx8 S0 = (fs_floatx8) {0, 0, 0, 0, 0, 0, 0, 0}, S1 = S0;
#pragma unroll
        for (int dt = 0; dt < FS_D/16; ++dt) {
            const int b = dt/2, half_ = dt % 2;
            const fs_half2 dk2 = {(_Float16) fs_half_at(w_, b*T::blk), (_Float16) fs_half_at(w_, b*T::blk)};
            // the lane's 16 dims of this d-tile as f16 (own); the halves exchanged (oth) give the second key tile (see above)
            uint32_t own[8], oth[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t kq;
                if constexpr (!T::q4) kq = fs_word_at(w_, b*T::blk + 2 + 16*half_ + 4*j) ^ 0x80808080u;
                else { const uint32_t raw = fs_word_at(w_, b*T::blk + 2 + 4*j); kq = (half_ ? raw >> 4 : raw) & 0x0F0F0F0Fu; }
                const fs_half2 lo = (fs_bits_h2(__builtin_amdgcn_perm(FS_MAGIC_H2, kq, FS_SEL_LO01)) + koffs)*dk2;
                const fs_half2 hi = (fs_bits_h2(__builtin_amdgcn_perm(FS_MAGIC_H2, kq, FS_SEL_HI23)) + koffs)*dk2;
                own[2*j] = fs_h2_bits(lo); own[2*j + 1] = fs_h2_bits(hi);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) oth[j] = fs_xchg16(own[j]);
            union { uint32_t u[8]; fs_halfx16 h; } bq;
            *(uint4 *) &bq.u[0] = *(const uint4 *) &q_f16[wave][l16][4*((2*dt) ^ sw)];
            *(uint4 *) &bq.u[4] = *(const uint4 *) &q_f16[wave][l16][4*((2*dt + 1) ^ sw)];
            S0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const fs_halfx16 *) own, bq.h, S0);
            S1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(*(const fs_halfx16 *) oth, bq.h, S1);
            __builtin_amdgcn_sched_barrier(0);
        }
        float S[2][8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { S[0][e] = S0[e]; S[1][e] = S1[e]; }

        Tile T0;
        load_tile(t0, 0, T0);
        process_tile(T0, S[0]);
        if (t0 + FS_WAVE <= kend) {    // always true (chunk % 32 == 0): a real branch splits the scheduling regions, which keeps the register allocator sane
            Tile T1;
            load_tile(t0, 1, T1);
            process_tile(T1, S[1]);
        }
    }

    const float ltot = l + fs_xchg16(l);
    if (col < NC && (col % NQ) < p.n_q) {     // padding columns (col % NQ >= n_q) are never stored
        const int gcol = hq*NQ + qi;         // global column = hq*NQ + qi (as the combine reads it)
        float * dst = p.part_acc + ((int64_t) c*NCOL + gcol)*FS_D + 8*hh;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            ((float4 *) (dst + 16*e))[0] = make_float4(O[0][e], O[1][e], O[2][e], O[3][e]);
            ((float4 *) (dst + 16*e))[1] = make_float4(O[4][e], O[5][e], O[6][e], O[7][e]);
        }
        if (hh == 0) p.part_ml[(int64_t) c*NCOL + gcol] = make_float2(m, ltot);
    }
}

// grid.x = HQ*n_q real columns (blockIdx = hq*n_q + qi; partial column = hq*NQ + qi), block = D: combine the chunks and the
// sink logit of the head; out = ggml dst [n_q][HQ][D]
template <int NQ>
static __global__ void fs_combine(const float * __restrict__ part_acc, const float2 * __restrict__ part_ml, int chunks, int n_q,
        const float * __restrict__ sinks, float * __restrict__ out) {
    constexpr int NCOL = FS_HQ*NQ;
    const int hq = blockIdx.x / n_q, qi = blockIdx.x % n_q, col = hq*NQ + qi, d = threadIdx.x;
    const float sink = sinks ? sinks[hq] : -INFINITY;
    float M = sinks ? sink : -1e30f;
    for (int c = 0; c < chunks; ++c) M = fmaxf(M, part_ml[(int64_t) c*NCOL + col].x);
    float L = sinks ? __expf(sink - M) : 0.0f, acc = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float2 ml = part_ml[(int64_t) c*NCOL + col];
        const float w = __expf(ml.x - M);
        L += ml.y*w; acc += part_acc[((int64_t) c*NCOL + col)*FS_D + d]*w;
    }
    out[((int64_t) qi*FS_HQ + hq)*FS_D + d] = acc/L;
}

template <typename T, int NQ, int HPB>
void fs_launch(cudaStream_t st, const fs_params & p, int chunks, const float * sinks, float * out) {
    constexpr int WPH = (FS_G*NQ + 15)/16;
    fs_wmma<T, NQ, HPB><<<dim3(chunks, FS_HKV/HPB), HPB*WPH*FS_WAVE, 0, st>>>(p);
    fs_combine<NQ><<<FS_HQ*p.n_q, FS_D, 0, st>>>(p.part_acc, p.part_ml, chunks, p.n_q, sinks, out);
}

int fs_env(const char * name, int dflt) {
    const char * e = getenv(name);
    return e ? atoi(e) : dflt;
}

} // namespace

#endif // defined(GGML_USE_HIP)

bool ggml_cuda_flash_attn_ext_inkling_smallbatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    static const bool enabled = fs_env("GGML_CUDA_FATTN_INKLING_SMALL", 1) != 0;
    static const int  min_kv  = fs_env("GGML_CUDA_FATTN_INKLING_SMALL_MIN_KV", 32768);
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    const ggml_tensor * table = dst->src[6], * idx = dst->src[7];

    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));

    const int64_t n_kv = K->ne[1], n_q = Q->ne[1];
    const bool shape_ok =
        n_q >= 2 && n_q <= FS_NQ_MAX && n_kv >= min_kv && n_kv % FS_WAVE == 0 &&
        Q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && Q->ne[0] == FS_D && Q->ne[2] == FS_HQ && Q->ne[3] == 1 &&
        Q->nb[0] == 4 && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && (uintptr_t) Q->data % 16 == 0 &&
        K->ne[0] == FS_D && K->ne[2] == FS_HKV && K->ne[3] == 1 && V->ne[0] == FS_D && V->ne[1] == n_kv && V->ne[2] == FS_HKV &&
        K->type == V->type && (K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0) &&
        (uintptr_t) K->data % 8 == 0 && K->nb[1] % 8 == 0 && K->nb[2] % 8 == 0 && K->nb[1]*FS_WAVE < (int64_t) 1 << 31 &&
        (uintptr_t) V->data % 8 == 0 && V->nb[1] == K->nb[1] && V->nb[2] == K->nb[2] &&    // one pitch for both (the cache allocates K and V alike)
        mask && mask->type == GGML_TYPE_F16 && mask->ne[0] == n_kv && mask->ne[1] >= n_q && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        mask->nb[1] % 16 == 0 && (uintptr_t) mask->data % 16 == 0 &&
        table && idx && !dst->src[5] && table->type == GGML_TYPE_F32 && table->ne[0] > 0 && table->ne[1] == FS_HQ && table->ne[2] >= n_q &&
        ggml_is_contiguous(table) && idx->type == GGML_TYPE_I32 && idx->ne[0] == n_kv && idx->ne[1] >= n_q && idx->nb[0] == 4 &&
        idx->nb[1] % 16 == 0 && (uintptr_t) idx->data % 16 == 0 &&
        (!sinks || (sinks->type == GGML_TYPE_F32 && sinks->ne[0] == FS_HQ)) &&
        max_bias == 0.0f && softcap == 0.0f && ggml_get_op_params_i32(dst, 4) == 0 && ggml_is_contiguous(dst);
    static const bool debug = fs_env("GGML_CUDA_FATTN_INKLING_DEBUG", 0) != 0;
    if (!shape_ok) {
        if (debug && n_q >= 2 && n_q <= FS_NQ_MAX) {
            fprintf(stderr, "inkling smallbatch: rejected n_q %lld n_kv %lld K %s V %s rows %d mask[%lld,%lld,%lld] tbl[%lld,%lld,%lld] idx[%lld,%lld] Qnb %zu/%zu Knb %zu/%zu sinks %d mb %.1f sc %.1f p4 %d\n",
                (long long) n_q, (long long) n_kv, ggml_type_name(K->type), ggml_type_name(V->type), dst->src[5] != nullptr,
                mask ? (long long) mask->ne[0] : -1, mask ? (long long) mask->ne[1] : -1, mask ? (long long) mask->ne[2] : -1,
                table ? (long long) table->ne[0] : -1, table ? (long long) table->ne[1] : -1, table ? (long long) table->ne[2] : -1,
                idx ? (long long) idx->ne[0] : -1, idx ? (long long) idx->ne[1] : -1, Q->nb[1], Q->nb[2], K->nb[1], K->nb[2],
                sinks != nullptr, max_bias, softcap, ggml_get_op_params_i32(dst, 4));
        }
        return false;
    }

    // measured-best configurations (NOTES.md): n_q <= 8: 8 heads per block, chunk 512 (1024 at 262k); n_q 16: 4 heads per block
    const int nq_t  = n_q <= 2 ? 2 : n_q <= 4 ? 4 : n_q <= 8 ? 8 : 16;
    const int chunk = nq_t <= 8 && n_kv >= 262144 ? 1024 : 512;
    const int chunks = (int) ((n_kv + chunk - 1)/chunk);
    if (debug) fprintf(stderr, "inkling smallbatch: taken n_q %lld (template %d) n_kv %lld K %s chunk %d\n", (long long) n_q, nq_t, (long long) n_kv, ggml_type_name(K->type), chunk);

    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool(), (size_t) chunks*FS_HQ*nq_t*FS_D);
    ggml_cuda_pool_alloc<float2> part_ml (ctx.pool(), (size_t) chunks*FS_HQ*nq_t);
    cudaStream_t st = ctx.stream();

    fs_params p;
    p.Q = (const float *) Q->data; p.qs_q = Q->nb[1]/4; p.qs_h = Q->nb[2]/4; p.scale = scale;
    p.K = (const char *) K->data; p.V = (const char *) V->data; p.nb_pos = K->nb[1]; p.nb_head = K->nb[2];
    p.mask = (const half *) mask->data; p.mask_stride = mask->nb[1]/2;
    p.idx = (const int32_t *) idx->data; p.idx_stride = idx->nb[1]/4;
    p.table = (const float *) table->data; p.table_stride_q = table->ne[0]*table->ne[1]; p.n_bins = (int) table->ne[0];
    p.n_q = (int) n_q; p.n_kv = (int) n_kv; p.chunk = chunk;
    p.part_acc = part_acc.ptr; p.part_ml = part_ml.ptr;
    const float * sk = sinks ? (const float *) sinks->data : nullptr;
    float * out = (float *) dst->data;

    const auto run = [&](auto tag) {
        using T = decltype(tag);
        switch (nq_t) {
            case 2:  fs_launch<T, 2,  8>(st, p, chunks, sk, out); break;
            case 4:  fs_launch<T, 4,  8>(st, p, chunks, sk, out); break;
            case 8:  fs_launch<T, 8,  8>(st, p, chunks, sk, out); break;
            default: fs_launch<T, 16, 4>(st, p, chunks, sk, out); break;
        }
    };
    if (K->type == GGML_TYPE_Q8_0) {
        run(fs_q8_0{});
    } else {
        run(fs_q4_0{});
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#endif
}
