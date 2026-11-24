# Small-batch verify batches of Inkling 2 Mini on gfx1151: MUL_MAT_ID roofline and what was changed

Roofline first (sections "Which path", "Measured"), then the two changes that came out of it ("Changes").

## Question

Verify batch = 33 ms + 8-10 ms per extra token (2..8 tokens). Is the per-token cost inherent (more unique expert
bytes read, bandwidth-bound) or implementation (expert weights re-read per token, launch/occupancy)?

## Which path MUL_MAT_ID takes today on gfx1151 (RDNA3.5 -> RDNA3 tables), Inkling shapes

Dispatch is `ggml_cuda_mul_mat_id` (ggml-cuda.cu). The upstream "few-row MMQ" (`mmq-few.cu`, 2..24 tokens, groups
tokens per expert) is `#if !GGML_USE_HIP` -- NVIDIA only; on HIP `ggml_cuda_should_use_mmq_few` returns false. So:

| src0 type | n_tokens | path | kernel | how tokens are grouped |
|---|---|---|---|---|
| IQ4_NL (experts, layers 0-19) | 1 | MMVQ | `mul_mat_vec_q<type, ncols_dst=1>` with ids (mmvq.cu) | one block per (expert slot, row group) |
| IQ4_NL | 2..6 | MMVQ | `mul_mat_vec_q_moe<type, rows_per_block=2>` (mmvq.cu:949) | **none**: block = (32 lanes, n_tokens warps), grid = (nrows/2, 8 slots); warp = one (token, slot) pair, reads ITS expert's 2 rows independently |
| IQ4_NL | 7, 8 (..) | MMQ -> Inkling WMMA kernel (`mmq-inkling-moe.cu`, `ggml_cuda_mmq_inkling_moe_applies` has no token gate) | per expert: ceil(n_e/32) tiles of BM=128 rows x 32 tokens, whole K | weights read once per expert; compute padded to 32 tokens |
| Q8_0 (experts, layers 20-23) | 1 | MMVQ ncols_dst=1 | as above | |
| Q8_0 | 2..8 | MMVQ `mul_mat_vec_q_moe` | as above | **none** (per (token, slot) pair) |

`get_mmvq_mmid_max_batch_rdna3`: IQ4_NL 6, Q8_0 default 8 (MMVQ_MAX_BATCH_SIZE). gfx1151 is GGML_CUDA_CC_IS_RDNA3
(RDNA3_0 || RDNA3_5), so the RDNA3 table applies.

Consequence for the hypothesis: in `mul_mat_vec_q_moe` the bytes REQUESTED are n_tokens x 8 x expert_bytes whatever
the sharing (a token's 8 (token, slot) pairs each stream their expert's rows; two tokens routed to the same expert are
in different warps of the same block -- same blockIdx.x row group, different threadIdx.y -- so their duplicate row
reads are close in time and may be served by L2 (gfx1151: 2 MB L2 per XCD-ish, 32 MB MALL). Whether DRAM traffic is
n_tokens x or unique x is exactly what burst1 measures (pool 8 vs pool 0 at equal n). The per-expert bytes:
gate_up IQ4_NL 2304x2304 = 2.99 MB, down 2304x1152 = 1.49 MB (IQ4_NL 18 B / 32 values); Q8_0 (34 B / 32): 5.64 / 2.82 MB.
Unique-bytes floor per matmul at ~230 GB/s (measured DRAM ceiling in kernel-research/moe/NOTES.md): with D distinct
experts, gate_up IQ4_NL = D x 13 us, down = D x 6.5 us; 48 visits x 2 matmuls per token.

The 7->8 token step crosses the IQ4_NL path switch (MMVQ -> WMMA-MMQ), so the "8-10 ms per token" slope is a
mix of two paths; burst1 times n = 6, 7, 8 separately to see the discontinuity.

## Measured (logs/burst1.log, logs/burst2*.log, logs/ppl-ubN.log)

Read ceiling (`stream`, 285 MB, 4/8/16-byte loads): 223-242 GB/s, use 237. (32 MB fits the MALL: 500-940 GB/s.)

Real routing + real per-call time: llama-perplexity, IQ4_NL model, -c 2048 --chunks 1 -b N -ub N, OP_STATS=2
(synchronizes after every op: includes ~10-20 us of launch/sync per call). Distinct experts per call from
GGML_CUDA_MMID_ROUTE_STATS. Per-expert bytes: IQ4_NL gate_up 2.986 MB / down 1.493 MB, Q8_0 5.64 / 2.82 MB.
GB/s = distinct experts x bytes / time (unique bytes).

| n | distinct experts (iq4 / q8) | iq4 gate_up us (GB/s) | iq4 down us (GB/s) | q8 gate_up us (GB/s) | q8 down us (GB/s) | all MUL_MAT_ID ms/eval | floor ms (unique bytes at 237 GB/s) | x floor | whole eval ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 / 8         | 143 (167) MMVQ | 93 (129)  | 234 (193)  | 136 (165) | 11.88 | 7.79  | 1.52 | 51.1 |
| 2 | 12.85 / 13.1  | 236 (163) MMVQ-moe | 125 (154) | 425 (174)  | 205 (181) | 18.35 | 12.56 | 1.46 | 71.1 |
| 4 | 20.0 / 20.7   | 361 (165) MMVQ-moe | 189 (158) | 733 (159)  | 325 (180) | 28.36 | 19.60 | 1.45 | 84.5 |
| 6 | 25.5 / 26.8   | 489 (156) MMVQ-moe | 253 (150) | 1042 (145) | 446 (169) | 38.37 | 25.04 | 1.53 | 98.8 |
| 7 | 27.85 / 29.2  | 514 (162) WMMA-MMQ | 253 (164) | 1239 (133) | 529 (156) | 40.52 | 27.33 | 1.48 | 103.8 |
| 8 | 30.1 / 31.6   | 519 (173) WMMA-MMQ | 266 (169) | 1413 (126) | 557 (160) | 42.26 | 29.55 | 1.43 | 108.1 |

(44 IQ4_NL + 4 Q8_0 visits of each shape per evaluation. Tokens-per-expert histogram at n=8, iq4: 1:51% 2:22%
3:11% 4:6% 5:4% 6:2% 7:2% 8:2% of the distinct experts.)

Findings:
- An 8-token batch of consecutive tokens touches 30 distinct experts per matmul, not 64 and not ~8: sharing is
  moderate. Distinct experts grow 8 -> 12.9 -> 20 -> 25.5 -> 27.9 -> 30.1 for n = 1, 2, 4, 6, 7, 8.
- The synchronized evaluation grows 51.1 -> 108.1 ms (8.1 ms per extra token, matching the server's 8-10).
  MUL_MAT_ID is 30.4 ms of that 57 ms (53%); of those 30.4 ms, 21.8 ms is the growth of the unique-byte floor.
  So ~72% of the MoE slope is inherent (more distinct expert bytes), ~28% (8.6 ms over 7 tokens, 1.2 ms/token)
  is above the floor.
- The hypothesis "the small-batch MMVQ path re-reads an expert per token from DRAM" is REFUTED for IQ4_NL: the
  unique-byte bandwidth of mul_mat_vec_q_moe is flat at 150-167 GB/s for n = 1..6 although it requests n x 8
  expert reads; the duplicates are served from cache (the warps of one block read the same rows together).
  The ratio to the floor is the same at n = 1 (1.52x) as at n = 8 (1.43x): there is no small-batch-specific
  penalty, only the generic gap between these kernels (150-173 GB/s) and the stream ceiling (237 GB/s).
- The exception is Q8_0 gate_up on MMVQ-moe: 193 GB/s at n=1 falling to 126 GB/s at n=8 (1.41 ms per call, 5.6 ms
  per evaluation for 4 visits; floor 3.0 ms). Its duplicate reads stop being free (2304 x 34 B rows x 2 rows x 8
  warps per block). Small in absolute terms: 4 of 48 visits.
- The other 26.6 ms of the slope is not MoE: MUL_MAT 15.1 -> 20.0, and a step from n=1 to n=2 of ~9 ms that does not
  grow afterwards (CONCAT 0.06 -> 3.2, CPY 0.01 -> 2.2, SSM_CONV chains 0 -> 2.1, MUL 0.7 -> 1.6 ms): the batched
  graph takes a different (unfused, copying) short-conv/state path than one-token decode.

Standalone (burst1, moe_bench, 8 graph copies): random routing rows agree with the model (iq4 gate_up n=6, 42
experts: 0.590 ms = 213 GB/s; n=8, 52 experts: 0.925 ms med / 0.748 min). The pool-8 rows are NOT model-realistic:
8 experts x 2.99 MB = 24 MB sits in the 32 MB MALL across the graph copies (n=1: 0.049 ms = 490 GB/s, above DRAM).
Do not use them as a floor. They do show the path switch: pool 8, n=6 MMVQ 0.225 ms vs n=7 WMMA 0.098 ms.

## Verdict of the roofline

The expert matmuls are not within 15% of the unique-byte floor (1.43-1.53x), but the gap is not one a "read each
expert once" kernel closes for IQ4_NL: it already reads unique bytes once in effect, and the ratio is the same at one
token. Whole MoE headroom at 8 tokens: 12.7 ms of 108 at a perfect 237 GB/s, ~7 ms at 200 GB/s. Cheaper wins came
first: the non-MoE step from 1 to 2 tokens and the Q8_0 experts.

The upstream few-row MMQ with WMMA for RDNA3.5 (PR 32224) does not apply to this model: it excludes MUL_MAT_ID
(`has_ids` returns 0 columns, `GGML_ASSERT(!ids)`), and every plain MUL_MAT of the IQ4_NL file is Q8_0 or F32, which
it does not cover. It cherry-picks cleanly (mmq-few.cu, quantize.cu/.cuh) and was dropped unbuilt.

## Changes

### 1. Q8_0 MUL_MAT_ID on MMQ from 4 tokens (mmvq.cu, `GGML_CUDA_MMID_Q8_MMQ_MIN`, default 4, 0 = old table)

burst3, moe_bench, median ms, `--pool 32` (20-31 distinct experts, the model's sharing; weights larger than the MALL):

| n | gate_up MMVQ | gate_up MMQ | down MMVQ | down MMQ |
|---|---|---|---|---|
| 2 | 0.353 | 0.354 | 0.148 | 0.148 |
| 3 | 0.562 | 0.485 | 0.177 | 0.217 |
| 4 | 0.661 | 0.539 | 0.303 | 0.269 |
| 5 | 0.859 | 0.610 | 0.338 | 0.338 |
| 6 | 1.061 | 0.687 | 0.422 | 0.367 |
| 7 | 1.253 | 0.784 | 0.507 | 0.367 |
| 8 | 1.563 | 0.821 | 0.591 | 0.421 |

With random routing (`--pool 0`, 52 experts at n=8) MMQ wins less: 1.578 -> 1.417 / 0.748 -> 0.735 ms, and loses
slightly at 2-3 tokens. In the model: KL against the old path at ub 8 = 0.000014 (max 0.0007, same top 99.90%).

### 2. Multi-token short-conv fusion (inkling2.cu, `GGML_CUDA_INKLING2_CONV_MT`, bit 0 CONV_STATE, bit 1 CONV_DYN, default 1 since the rebase; the measurements below used 3)

Cause of the 1 -> 2 token step: the allocator puts the chain output in place of its inputs (`cur`, `r`, or the
residual). The per-element CONV_STATE kernel may only do that for one token (a thread reads other tokens' inputs), so
for 2+ tokens the matcher fell back to CONCAT + CPY + the CONV chain; CONV_DYN was one token only. Per evaluation:
192 CONCAT, 192 CPY and the 16 dynamic-tap convolutions (SSM_CONV, 64 CONT, 64 MUL, 80 ADD) unfused.

`inkling2_conv_mt_f32`: one thread per (channel, sequence) for 2..32 tokens, reads all of its inputs, then writes; taps
and residual optional. Dispatches per evaluation at ub 8: 3171.9 -> 2597.7.
`GGML_CUDA_INKLING2_CONV_MT_DEBUG=1` logs each kind of launch once with the aliasing it saw; in the model:
CONV_STATE n_c=1024 in place of cur and r, n_c=2304 in place of the residual, CONV_DYN n_c=1024 in place of cur and r.

Correctness:
- test-backend-ops (ROCm0 vs CPU): INKLING2_CONV_STATE 35/35, INKLING2_CONV_DYN 17/17 (new cases: 2..33 tokens, 3
  sequences, input computed by an op), INKLING2_CONV 10/10, MUL_MAT_ID 1087/1087. With CONV_MT=0 the earlier set
  (21 + 9 cases, before the computed-input cases were added) passed; the final set was not rerun with it off.
  The test graphs never get the in-place allocation (the debug log shows cur=0 r=0 res=0 for all of them); that
  case is only covered in the model, below.
- KL at ub 8 against the unfused path (llama-perplexity, 2048 tokens, one chunk; a repeated base run gives exactly 0):
  CONV_STATE only (mask 1): 0.000000, max 0.000052, same top 100% -- bit-identical logits, with in-place launches.
  CONV_DYN only (mask 2): 0.003756 +- 0.000396, max 0.166, same top 98.34%. Both (mask 3): the same.
- So CONV_DYN on the new kernel is NOT bit-identical to the unfused chain. Its size is that of any change of
  evaluation path in this model: against one-token decode logits, ub 8 unfused is 0.003567, ub 8 fused 0.004207,
  ub 2 unfused 0.003879, ub 2 fused 0.003265 (same top 98.5-98.7% for all four) -- the fused path is not
  systematically farther from one-token decode. The conv sum itself is bit-exact (CONV_STATE shares it); the
  difference is in the tap terms or the final add/scale. NOT established: which operation rounds differently, and
  that it is rounding rather than a small error. `GGML_CUDA_INKLING2_CONV_MT=1` keeps only the bit-identical part.

### Speed in the model (burst5/6: llama-perplexity -c 2048 --chunks 1 -b N -ub N, "seconds per pass" / 2048 * N, no
per-op synchronization; base = both switches off; rounds interleaved)

| n | base ms/batch (rounds 1-5) | new ms/batch (rounds 1-5) | means of rounds 2-5 |
|---|---|---|---|
| 1 | 31.35 30.33 | 30.56 30.46 | no change (30.8 / 30.5, 2 rounds) |
| 2 | 41.54 42.05 42.93 43.46 42.68 | 43.28 40.05 41.11 42.84 40.32 | 42.78 -> 41.08 (-1.7) |
| 4 | 59.80 57.56 58.98 57.83 57.79 | 66.50 54.98 58.20 55.80 55.88 | 58.04 -> 56.22 (-1.8) |
| 8 | 86.33 85.27 86.64 83.79 91.13 | 113.63 81.95 81.13 79.77 85.04 | 86.71 -> 81.97 (-4.7, 5.5%) |

Round 1 of "new" is slow at every n >= 2 (43.3, 66.5, 113.6) and is left out of the means; the cause was not found
(other GPU work on the box is not excluded for that window; rounds 2-5 never reproduce it). Run-to-run
spread is 2-3 ms at 8 tokens, so the split between the two changes is not resolved: single KL-run samples give
base 83.2 / 84.6, conv only 82.5, Q8_0 only 82.5, both 80.1; in a quieter window base 80.6, CONV_STATE only 79.5,
both conv chains 79.1. The synchronized per-op profile had put the conv step at 8-9 ms; unsynchronized it is ~1.5-2.

## Not done

- A bit-exact CONV_DYN, or the proof that its difference is one rounding.
- An IQ4_NL small-batch MUL_MAT_ID kernel (target >= 200 GB/s unique bytes; ~7 ms at 8 tokens if reached).
- Timing inside llama-server with a draft model; the numbers above are plain batched decodes at depth <= 2048.

## Tooling added (this directory + edits)

- `ggml/src/ggml-cuda/ggml-cuda.cu`: `GGML_CUDA_MMID_ROUTE_STATS=1` -- in `ggml_cuda_mul_mat_id`, for n_tokens <= 32
  copies ids to the host (stream sync; run with `GGML_CUDA_DISABLE_GRAPHS=1`) and at exit prints, per
  "type ne00xne01xne02 n_tokens=N": calls, average distinct experts per call, tokens-per-expert histogram. This is
  the "real routing" count .
- `kernel-research/moe/moe_bench.cpp`: `--pool P` routes every token within the first P experts (`--pool 8` with
  `--nused 8` = all tokens share the same 8 experts, the max-sharing case; `--pool 0` = random, ~8n distinct).
- `build.sh`: builds `moe_bench_sb` (moe_bench against build-rocm-moesb/bin, rpath set) and `stream` (the
  read-bandwidth probe from kernel-research/bw/stream.hip; already compiled).
- `burst1.sh`: stream ceiling, then moe_bench matrix: {iq4_nl, q8_0} x {gate_up 2304x2304 bcast, down 2304x1152} x
  n in {1,2,3,4,6,7,8,12,16} x pool in {8,16,0}; each call under quiet.sh (lock released between calls).
- `burst2.sh`: llama-perplexity on the IQ4_NL model, -c 2048 --chunks 1 -b N -ub N for N in 1 2 4 6 8 (consecutive
  tokens = the verify-batch shape), with GGML_CUDA_OP_STATS=2 (per-op synchronized wall time keyed by type+shape;
  MUL_MAT_ID rows give the real-routing per-call time per n) and GGML_CUDA_MMID_ROUTE_STATS=1. Logs to
  logs/ppl-ubN.log. `UBS="1 8" ./burst2.sh` to run a subset. Needs ~17 GiB of GPU memory + the model in page cache.

- `burst3.sh` .. `burst8.sh`: Q8_0 MMVQ vs MMQ; test-backend-ops; in-model A/B and KL; KL against one-token decode;
  in-place tests with the debug log and the per-chain KL split. Scripts take MODEL, TXT and optionally QUIET (a wrapper
  that serializes GPU runs) from the environment.
