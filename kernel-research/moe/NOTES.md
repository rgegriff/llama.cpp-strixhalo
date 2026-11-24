# Dedicated IQ4_NL MUL_MAT_ID (MoE expert matmul) kernel for gfx1151

Shapes: Inkling 2 Mini, 128 experts, 8 used, 512-token ubatch -> ne_get_rows 4096 (~32 tokens per expert).
gate_up: m=2304 k=2304 (382 MB of IQ4_NL weights), down: m=2304 k=1152 (191 MB). Baseline = MMQ with the J=16
tile forced for MUL_MAT_ID on RDNA3.5 (f09c7307): 4.3-4.4 ms / 2.3-2.4 ms in the standalone harness.

Floors at 512 tokens: weights once from DRAM at ~230 GB/s = 1.66 / 0.83 ms; WMMA iu8 at the measured 50 TOPS
peak = 0.87 / 0.43 ms (x1 padding; x1.25 with 16-token tiles).

## Result

`moe_wmma` (v1 + expert-aligned tile plan), ported as `ggml/src/ggml-cuda/mmq-inkling-moe.cu`:

| tokens | shape   | MMQ J=16 (ms) | wmma+plan (ms) | speedup |
|---|---|---|---|---|
| 256  | gate_up | 2.63-2.67 | 2.00-2.03 | 1.31x |
| 256  | down    | 1.56-1.58 | 1.05-1.06 | 1.49x |
| 512  | gate_up | 4.23-4.44 | 2.98-3.12 | 1.43x |
| 512  | down    | 2.31-2.75 | 1.45-1.61 | 1.58x |
| 2048 | gate_up | 14.0-14.1 | 9.02-9.07 | 1.55x |
| 2048 | down    | 7.39-7.55 | 4.64-4.74 | 1.59x |
| 512, skew 0.5 | gate_up | 4.51 | 3.01 | 1.50x |
| 512, skew 0.5 | down    | 2.50 | 1.72 | 1.46x |

(bursts 15-17: median of 12-15 reps, 2-3 interleaved rounds, GPU at ~2.2 GHz under load per GRBM_GUI_ACTIVE.)
At 512 tokens the dedicated kernel runs at 13.7 / 13.2 TFLOPS and reads weights at 120 / 116 GB/s -- roughly 1/3 of
either ceiling, up from 1/4-1/5. Not the hoped-for 2x: see "What limits it".

Correctness: vs the MMQ kernel on the same buffers, all 9.4M outputs: NMSE 1.1e-14, max |diff| 1.9e-5 (summation
order only); vs a host f64 dequant+dot reference NMSE 1.35e-5 = the Q8_1 activation error, identical to MMQ's.
Through libggml (moe_bench --check vs the CPU backend): NMSE 3.5e-8. test-backend-ops -o MUL_MAT_ID: 1087/1087
with the kernel on (11 of the new IQ4_NL cases take it, GGML_CUDA_INKLING_MOE_DEBUG=1) and off (burst 19).

## Files

- `moe_wmma_kernel.cuh` -- the kernel (`moe_wmma::mul_mat_id_iq4nl`) and the expert-aligned tile planner
  (`plan_tiles`); compile-time variants: `MOE_NO_LOAD`, `MOE_NO_MMA`, `MOE_NO_DEQ` (ablations, wrong results),
  `MOE_HALF_DEQ` (+`MOE_HALF_DEQ2`), `MOE_DY_T`, `MOE_OCC=n`, `MOE_WMMA_SWAP_AB` (operand order probe).
  `moe_wmma_kernel_v4.cuh` -- later experiments (NTT-templated body, `MOE_NWARPS=8`, `MOE_BUNROLL`), all equal or slower.
- `moe_wmma.hip` -- harness: moe_mmq.hip's prep (mm_ids_helper + quantize_mmq_q8_1, same data) with
  `--kernel wmma|mmq`, `--plan` (expert-aligned tiles), `--check` (host dequant+dot reference), `--cmp` (all outputs
  vs the MMQ kernel on the same buffers). `build_wmma.sh` (VARIANT=, DEFS=, SUFFIX=). `moe_bench_moe` = moe_bench
  linked against build-rocm-moe.
- `burst9..19.sh`, `logs/burst*.log`, `logs/pmc/moe_wmma_*` (rocprofv3 counters), `isa/` (ISA dumps).
- Port: `ggml/src/ggml-cuda/mmq-inkling-moe.cu/.cuh` (the v1 kernel + plan_tiles, no variants), dispatched from
  `ggml_cuda_mul_mat_q` (mmq.cu) after the ids helper and the Q8_1 quantization when `GGML_CUDA_INKLING_MOE` != 0,
  IQ4_NL, RDNA3.5, prec Q8, no fusion, ne00 % 128 == 0, ne01 % 128 == 0, <= 1024 experts. test-backend-ops got
  IQ4_NL MUL_MAT_ID cases with 128-multiple shapes (none existed).

## Design

- Workgroup = 128 threads (4 waves) x (BM=128 expert rows, BN=32 compact tokens of one expert), whole K.
  Wave w owns rows 32w..32w+31 = 2 x 16-row WMMA tiles, x 2 x 16-token tiles (the second only when the tile has
  > 16 tokens: `ntt`).
- K loop in chunks of 128 (= one block_q8_1_mmq per token = 4 IQ4_NL blocks per row). Thread t stages row t:
  9 x 8-byte global loads of the raw 72 bytes (rows are only 8B-aligned for k=1152), split in registers into the 4
  x 16 nibble bytes (16B-aligned in LDS, row stride 80 B -> conflict-free b128 fragment reads) and the 4 fp16 scales.
  The Q8_1_MMQ blocks of the 32 tokens (4608 B, contiguous in global memory) are copied verbatim.
  Double-buffered LDS (2 x 15872 B = 31.75 KB -> 2 workgroups per CU), the loads of chunk c+1 are issued before the
  compute of chunk c, one barrier per chunk.
- Fragments: WMMA A = 16 tokens x 16 K int8 (ds_read_b128 straight from the Q8_1 block), B = 16 expert rows x 16 K
  built in registers from 16 nibble bytes with the v_perm 16-entry table lookup (`table16`, 15 VALU per 8 values;
  the low nibbles are K 0..15, the high nibbles K 16..31 of the block, so one block = 2 WMMAs into one int32
  accumulator). Per 32-block: sum[a][t][e] += float(acc[e]) * (d_w[row] * d_y[token]) -- the MMQ numerics.
- WMMA operand layout on gfx11 (wave32), verified by `--check`: with A = tokens, B = rows, lane l of D holds expert
  row l%16 for tokens 2e + l/16 (e = 0..7); the swapped order fails. Both 16-lane halves hold replicated
  operands, and a per-half K permutation of A and B together is harmless (HALF_DEQ passes), so each half uses only
  its own data.
- `plan_tiles`: ceil(n_e/32) tiles per expert in expert order. mm_ids_helper's tiles are already expert-aligned
  (I first misread them as seam-aligned), but it puts each expert's second tile in a slot after all first tiles
  (index n_tokens*n_used/J + expert), so the two tiles of one expert run ~128 column tiles apart and the weights come
  from DRAM twice; adjacent tiles in launch order hit L2/MALL. Worth 5-7% (burst 15). WMMA padding is x1.25 with
  16-token granularity (printed as "16-slots" by the harness), not the 2x claimed in an earlier draft.

## What limits it (burst 11, 14; hardware counters per gate_up dispatch)

- Ablations of v1 at 3.2 ms: no global loads 2.65 ms, no WMMA+scaling 2.85 ms, no table lookup 2.95 ms. No single
  component dominates; the parts are nearly additive.
- Counters: GRBM_GUI_ACTIVE 7.06e6 cycles (2.2 GHz at 3.2 ms; the pp_dpm_sclk samples of 1.2-1.9 GHz are taken
  between kernels and mislead). SQ_INSTS_VALU 1.99e8 -- the same count as MMQ J=16 (which needs 9.57e6 cycles);
  SQ_WAIT_INST_LDS 4.9e6 (LDS is not the limiter); SQ_WAIT_INST_ANY 8.1e8 wave-cycles.
  WMMAs: 6.6e6 x 32 cycles / 80 SIMDs = 37% of the cycles; VALU issue (dequant 120 + scaling 96 + staging per
  32-row x 32-token x 32-K wave step, vs 8 WMMAs) another ~35%. The SIMD issue port is the bottleneck; the loads are
  hidden behind it.
- Things that did not help: HALF_DEQ (each half looks up 8 of the 16 bytes, exchange via v_permlanex16; -16% VALU
  instructions but +11% cycles with ds_read_b64 Y fragments; +-2% with b128 reads and a cndmask swap); DY_T (Y
  scales transposed in LDS, 2 b128 instead of 16 b32 reads per tile: no gain, LDS was not the limiter); 8 waves per
  workgroup (16 rows per wave: equal or slower, and the VGPR cap for 8 waves/SIMD spills); templating the body on
  the token-tile count (the compiler hoists everything and spills: 2.6x slower; unroll 1 of the block loop ties v1);
  launch_bounds forcing 192 VGPRs (spills, 1.4x slower). 251 VGPRs is where the compiler lands with 4 waves/SIMD.
- Not tried: per-128 activation scales (coarser than Q8_1's per-32, cuts the scaling VALU by ~25% but changes the
  numerics); f16 scale products via v_fma_mix (-16 VALU per step, not MMQ-identical); dequantizing in the staging
  pass into int8 LDS (halves the lookup work but needs BM=64 or one workgroup per CU); a planner that also
  reorders experts by size.

## Still to run (needs the model, main session)

- llama-perplexity --kl-divergence vs the f16 base logits with GGML_CUDA_INKLING_MOE=1 vs 0 (the J=16 change was
  bit-identical; this one differs in summation order only, expect KL at the noise floor).
- e2e prefill tok/s at depth 0 and 16k with the kernel on/off (GGML_CUDA_INKLING_MOE_DEBUG=1 to confirm it is taken
  for all 44 matmuls: ne01 2304, ne00 2304/1152, ne02 128).
