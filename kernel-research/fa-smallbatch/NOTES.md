# fa-smallbatch: Inkling 2 Mini decode attention for n_q = 1..16 queries

Standalone HIP prototype of the attention kernel the n-gram speculative-decoding verify step needs: `n_q` drafted
tokens (2..16) attend to a long q8_0/q4_0 KV cache in one pass over the cache. `fa_smallbatch.hip` holds the kernels,
a double-precision CPU reference and the timing harness; `run_sweep.sh` + `table.py` produce the table below.

```
/usr/lib/llvm-21/bin/clang++ -x hip --offload-arch=gfx1151 -O3 -std=c++17 fa_smallbatch.hip -o fa_smallbatch -lamdhip64 -pthread
./fa_smallbatch <q8_0|q4_0> <n_kv> <n_q> [chunk ...]       # env: REPS, NO_SINKS, REF_QUANT=1, IDX_SPRINKLE=all, STAGE=1
```

## Semantics (what the kernel takes)

Same op as the integrated n_q = 1 path (`ggml/src/ggml-cuda/fattn-inkling.cu`), generalized to `n_q` queries:

- Q f32, element (d, qi, hq) at `Q[qi*qs_q + hq*qs_h + d]` (ggml ne = [128, n_q, 24]; the llama.cpp view has
  `qs_h = 128`, `qs_q = 24*128`); scaled on device by `scale`.
- K/V q8_0 (136 B rows) or q4_0 (72 B rows) in the cache layout: row of (position k, KV head h) at `k*nb_pos + h*nb_head`.
  Q head hq uses KV head hq/3.
- mask f16 `[n_q][mask_stride]`: the queries are the last n_q positions, causal among themselves (the host builds the mask;
  the kernel just adds it).
- indexed bias: `idx` i32 `[n_q][idx_stride]`, `table` f32 `[n_q][24][n_bins]`; per (qi, hq, key):
  `bin = idx[qi][k]; bias = (uint32)bin < n_bins ? half(float(mask) + table[qi][hq][bin]) : mask` (half rounding as
  `fattn_indexed_bias::add`). Exact for arbitrary indices; a tile whose 8 bins (per lane) are all equal takes a 1-load fast
  path, which is the model's far field (bin 1024).
- sinks f32 [24] or null: denominator `+= exp(sink - max)`.
- out f32 `[n_q][24][128]` = ggml dst (ne0 = 128, ne1 = 24, ne2 = n_q).
- Split-K: `fa_small_*` writes per-chunk partials `[chunk][col = hq*n_q + qi][128]` and (m, l); `fa_combine` merges them
  and the sink.

Constraints: `n_kv % 32 == 0` and `chunk % 32 == 0` for n_q >= 2 (llama.cpp pads the KV length to 256); `nb_pos`,
`nb_head` multiples of 8; Q 16-byte aligned. The n_q = 1 path has no n_kv constraint (it is fa_decode2 with the bias
rounding and sinks).

## Design

- **n_q = 1** (`fa_small_valu`): the fa_decode2 / fattn-inkling.cu kernel (one wave per KV head, int8 dot products,
  Q quantized to q8_0 blocks). 202 VGPRs, 1.56 ms at 131k q8_0 -- identical to fa_decode2.
- **n_q = 2..16** (`fa_small_wmma`): WPH = ceil(3*n_q/16) waves per KV head, each owning 16 (query, head) columns
  (col = g*n_q + qi). Per 32 keys a wave loads the 32 K rows (lane = key; 9 x 16 B loads) and, per 16-key tile, the V
  bytes of its 8 dims x 16 keys (2 keys per 8 B load) plus the mask/index words of its query. K is dequantized to f16 in
  registers (v_perm magic-number trick + v_pk_add/v_pk_mul: 1.5 VALU per value), Q is f16 in LDS (4 KB per wave, quads
  XOR-swizzled), S^T = K.Q^T with `wmma_f32_16x16x16_f16`, online softmax in the WMMA output layout, P and V to f16,
  O^T += V^T.P^T with f16 WMMA (64 accumulator VGPRs per wave). Block = HPB heads x WPH waves; HPB 8 for n_q <= 8,
  4 for n_q 16 (LDS: 4 KB x waves).
- **Half-lane WMMA property** (probed in `wmma_half_probe.hip`): on gfx11 wave32 the even output rows D[2e][*] are
  computed from the lower 16 lanes' A and B rows, the odd rows from the upper 16 lanes'. With B identical in both halves,
  A = the 32 loaded K rows gives S for keys {2e} (lower lanes) and {17 + 2e} (upper), and A = the halves exchanged
  (`v_permlanex16`) gives {16 + 2e} and {1 + 2e}: two 16-key tiles with no operand selects. The V/mask/index loads of a
  tile follow the same key mapping (`key = t0 + 16*(hh ^ tile) + 2e + hh`).
- Errors vs the double reference (REF_QUANT=1 feeds the reference the Q the kernel uses): n_q = 1 path 8e-7 (exact int8
  arithmetic; without REF_QUANT 1% from the q8_0 Q quantization at the synthetic logit scale), WMMA path 3e-4 to 1e-3
  (K/V/P f16 rounding; Q itself is f16 there, so it is the *more* accurate path). Causal-within-batch, nonzero sinks,
  invalid bins and random near bins are all covered by the synthetic data (`IDX_SPRINKLE`).

## What limits it

Ablations at n_q 16, 131k, q8_0 (all timings in one `quiet.sh` burst; the ablation builds give wrong results):

| variant | ms | note |
| --- | --- | --- |
| full | 2.9 | 3 waves per head |
| no QK + no PV compute (loads kept) | 2.96 | compute is hidden |
| no K/V loads (compute + mask/index kept) | 1.68 | |
| duplicate waves read a cache-resident region | 2.87 | DRAM/MALL traffic of the duplicates is not it |
| all waves read different heads (3x DRAM) | 3.49 | |
| block barrier per iteration / per load phase | 2.84 / 3.18 | L0 locality is not it |
| tile loads moved before QK (one load phase) | 2.81-2.87 | latency structure is not it |

So the time is the memory-instruction path itself: each wave of a head issues 53 VMEM instructions per 32 keys
(9 K-row loads with 32 distinct 64 B lines each, 32 V loads, 12 mask/index loads), and with 3 waves per head that is
3x the L0/TA work of the n_q = 1 kernel, which is already at the DRAM limit. n_q = 4 (1 wave per head) stays at the
memory floor; n_q = 8 (2 waves) costs 1.2x; n_q = 16 (3 waves) 1.8x.

Compiler notes that mattered (LLVM 21, gfx1151): the register allocator sits at the 256 VGPR edge; the things that
removed spills were (1) reading loop-invariant LDS operands with a laundered index so they are not hoisted into VGPRs,
(2) `__builtin_amdgcn_readfirstlane` on the uniform row base + a 32-bit lane offset (`global_load ... saddr`), with the
lane offsets laundered inside the loop so LICM does not hoist one VGPR per (offset + constant), (3) a real (always-true)
branch between the two key tiles, which splits the scheduling regions (212 VGPRs, 0 spills, vs 256 + 40 spills
without it), (4) `v_permlanex16` with fi/bc set so the tied old operand needs no copy. `global_load` saddr is only
emitted when the offset is laundered as a VGPR.

### Experimental: staged kernel (`fa_small_stage`, env STAGE=1, n_q >= 8)

The fix for the memory-instruction bound is to load each K/V row once per block: the block stages the raw rows of its
4 heads for 32 positions (34 KB) plus the mask/index rows in LDS with coalesced 16 B loads (6 per lane per iteration
instead of 53), prefetched one iteration ahead, and uses the int8 WMMA on the raw K bytes with an int8 Q (2 KB per wave,
so it fits: 61 KB). It is correct (REF_QUANT rel err 3e-4) but the compiler spills ~100 VGPRs (the 64 per-iteration dk
reads from the staged rows; with them replaced by a constant it is at 2 spills) and runs 2x slower than `fa_small_wmma`.
Next step: a transposed per-wave dk table written from the K row each lane already holds (16 x `ds_read_b64` instead of
64 x `ds_read_u16`), or f16 WMMA with the int8 Q converted per d-tile.

## Results

See the table at the end (`python3 table.py results/sweep-*.log`); times are one full-attention visit, `REPS=20`
median-free averages, synthetic data with the model's index pattern.

```

q8_0: best time in us (kernel, chunk, hpb; GB/s of K+V; max rel err vs double reference)
n_q                                 n_kv 32768                             n_kv 131072                             n_kv 262144
1          413 (valu 1024 8; 172.8 GB/s; 2.2e-03)    1562 (valu 512 8; 182.5 GB/s; 1.4e-03)    3022 (valu 1024 8; 188.8 GB/s; 1.3e-03)
2          379 (wmma 1024 8; 188.1 GB/s; 3.1e-04)    1553 (wmma 512 8; 183.6 GB/s; 3.1e-04)    2972 (wmma 1024 8; 192.0 GB/s; 3.1e-04)
4          424 (wmma 512 2; 168.2 GB/s; 3.0e-04)    1649 (wmma 512 8; 172.9 GB/s; 3.0e-04)    3160 (wmma 1024 8; 180.5 GB/s; 3.0e-04)
8          480 (wmma 1024 8; 148.5 GB/s; 2.8e-04)    1869 (wmma 512 8; 152.6 GB/s; 3.2e-04)    3617 (wmma 512 8; 157.7 GB/s; 2.8e-04)
16         732 (wmma 512 4;  97.5 GB/s; 3.7e-04)    2898 (wmma 512 4;  98.4 GB/s; 3.0e-04)    6042 (wmma 512 4;  94.4 GB/s; 2.7e-04)
ratio to n_q = 1:
1                                         1.00                                    1.00                                    1.00
2                                         0.92                                    0.99                                    0.98
4                                         1.03                                    1.06                                    1.05
8                                         1.16                                    1.20                                    1.20
16                                        1.77                                    1.85                                    2.00

q4_0: best time in us (kernel, chunk, hpb; GB/s of K+V; max rel err vs double reference)
n_q                                 n_kv 32768                             n_kv 131072                             n_kv 262144
1          298 (valu 256 8; 126.9 GB/s; 2.1e-03)    1139 (valu 512 8; 132.6 GB/s; 2.2e-03)    2048 (valu 512 8; 147.4 GB/s; 1.7e-03)
2          228 (wmma 512 4; 165.5 GB/s; 3.1e-04)     979 (wmma 512 8; 154.2 GB/s; 3.1e-04)    1906 (wmma 512 8; 158.5 GB/s; 3.1e-04)
4          210 (wmma 512 4; 180.2 GB/s; 3.0e-04)    1110 (wmma 512 8; 136.1 GB/s; 3.0e-04)    2042 (wmma 512 8; 147.9 GB/s; 3.0e-04)
8          360 (wmma 512 2; 104.7 GB/s; 2.8e-04)    1439 (wmma 512 8; 105.0 GB/s; 2.8e-04)    2841 (wmma 512 8; 106.3 GB/s; 2.9e-04)
16         542 (wmma 512 4;  69.6 GB/s; 2.7e-04)    2471 (wmma 512 4;  61.1 GB/s; 2.7e-04)    4996 (wmma 512 4;  60.5 GB/s; 2.7e-04)
ratio to n_q = 1:
1                                         1.00                                    1.00                                    1.00
2                                         0.77                                    0.86                                    0.93
4                                         0.70                                    0.97                                    1.00
8                                         1.21                                    1.26                                    1.39
16                                        1.82                                    2.17                                    2.44
```

Sweep log: results/sweep-20261124-0036.log (not committed). Targets from the task: n_q 1 = fa_decode2 (1.56 ms at 131k q8_0) met; n_q 4 and 8 within 1.06x and 1.20x of it; n_q 16 at 1.85x (2.9 ms) misses the 2.5 ms goal -- see 'What limits it'.
