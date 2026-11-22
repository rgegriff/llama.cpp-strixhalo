# Inkling 2 Mini on Strix Halo (gfx1151): where a decoded token's time goes, and the kernels that would cut it

2026-11-22. Worktree `~/src/llama.cpp-inkling-kern`, branch `inkling-kern` (PR #32740 head
808fd1c4 plus eight commits).

**Nothing in this report was measured on the GPU.** The GPU was booked for the whole run. The op and byte counts
are exact (taken from the real graph and the real GGUF header). Every millisecond and tok/s figure is either the
ticket's rough measurement or a prediction from it, and is labelled. `bench.sh` turns the predictions into
measurements in one slot.

## 1. Summary

- A decoded token is **7,380 dispatched GPU ops** on the PR build. 786 of them read weights, 48 are attention,
  and about 6,550 are glue on buffers of at most 100 KB, most of them one 2,304-float vector. The count
  reproduces the server log's graph size exactly (11,394 nodes).
- It reads **5.40 GB of weights**. That alone is 21 ms at the box's theoretical 256 GB/s and 27 to 41 ms at the
  133 to 204 GB/s that the measurements allow.
- Every attention layer walks its K and V **three times** per token (once per query head; 24 query heads share
  8 K/V heads). On the 8 full-attention visits that is 99.5 KB per token of context, which matches the measured
  0.49 ms per 1k tokens of depth. On the 40 window visits it is a constant 0.76 GB per token.
- Prototyped and committed, all building for ROCm, Vulkan and CPU:
  1. the fused glue-op kernels of realugbun's `inkling2-speed` branch, ported onto the PR head: 7,044 → 2,492
     dispatches per token by op pattern;
  2. a new fused kernel for the compact bias table, which the port left behind: 2,492 → 2,396, and no rocBLAS
     call left in the decode graph;
  3. a decode attention kernel that reads K/V once per K/V head;
  4. counters and a built-in per-op profile (`GGML_CUDA_OP_STATS`), with switches to turn each change off in one
     binary.
- Predicted from the ticket's two rough timings. Fused chains alone: 17.5 → 21 to 26 tok/s at 5k depth, 12.1 →
  13.6 to 15.6 at 57k. With the K/V kernel on top: up to 29 at 5k and up to 23 at 57k if it removes the full two
  thirds of the K/V traffic, nothing more if the repeated reads were already served from cache.
- CPU-side checks passed: the ported graph gives the same logits as the PR head (KLD 0, same top token 100%), and
  the test suites run. GPU correctness and every speed number wait for a slot.

## 2. The model as llama.cpp runs it

From the GGUF header (`inkling2_loop`): 24 stored layers, layers 6 to 17 run 3 times, so **48 layer visits** per
token, each with its own K/V and short-conv state. Hidden 2304. 24 query heads on 8 K/V heads, **head dim 128**
(`attention.key_length`; 2304/24 = 96 is not the head dim). 8 of 128 experts, expert FFN 1152. Layers 5, 11, 17, 23
are full attention, the rest a 1,024-token window. Relative-position bias: rank 16, extent 1,024. Short conv: 4
taps. Loop controller rank 64. Vocabulary padded to 201,024.

Which flash-attention kernel runs with the indexed bias (`ggml-cuda/fattn.cu`):

| Case | Kernel | Note |
|---|---|---|
| Decode, one query | VEC, `flash_attn_ext_vec<128, 1, …, indexed_bias>` | one block column per query head |
| Prefill | TILE with `ncols2 = 1` | no K/V sharing across the heads of a K/V head |
| Any | MMA / rocWMMA | never: it ignores the bias. `GGML_HIP_ROCWMMA_FATTN=ON` buys nothing here |

The PR's test covers widths 64 and 128, so it does cover this model.

## 3. Cost model of one decoded token

### 3.1 Ops (exact)

Method: a random-weight copy of the model with the true shapes (`make_synth_gguf.py`, 3.4 GB) run on the CPU
backend with `GGML_SCHED_DEBUG=2`, parsed by `census.py`. The CPU graph has 11,826 nodes: the 11,394 of the
server log plus the 9 nodes per visit of the dense-bias fallback that only the CPU takes. `census.py` removes
those and the two empty state nodes per visit.

| Visit kind | Visits | Dispatched ops per visit (min / median / max) | Sum |
|---|---|---|---|
| Window attention | 20 | 108 / 115 / 119 | 2,309 |
| Full attention | 4 | 149 / 153 / 160 | 614 |
| Window, looped pass | 20 | 169 / 180 / 182 | 3,594 |
| Full, looped pass | 4 | 208 / 210 / 219 | 845 |
| Output block | | | 18 |
| **Token** | 48 | | **7,380** |

The spread within a kind is the depth mix, which concatenates one more block output per block.

A plain window visit, by group:

| Group | Ops | What |
|---|---|---|
| Projections | 7 | Q, K, V, gate, O, bias projection, sink (Q8_0 weights) |
| Row clips | 26 | Q (9), K (8), bias projection (9): `MUL SQR SUM_ROWS SQRT SCALE FILL DIV CLAMP MUL` |
| Short convs | 20 | K, V, attention out, FFN out: `CONCAT CPY SSM_CONV ADD SCALE` each |
| Depth mixes | 14 to 21 | attention input and FFN input: norm, concat, 2 small matmuls, softmax |
| Bias table, attention | 5 | `PAD MUL_MAT SUB`, `FILL`, `FLASH_ATTN_EXT` (2 launches) |
| K/V and state stores | 4 | 2 `SET_ROWS`, state `GET_ROWS`, V norm |
| Norms, gate, residuals | 8 | |
| Router | 9 | matmul, sigmoid, bias, argsort, gather, normalise, scale |
| Experts | 2 | `MUL_MAT_ID`, 8 experts in one launch each |
| SiTU | 9 | |
| Weighted expert sum | 8 | `MUL` + 7 `ADD` (master already fuses this) |

A full-attention visit adds about 38: two input norms, two tap projections, and the dynamic-conv lag loop
(`CONT MUL ADD` × 4 lags × K and V = 24). A looped visit adds about 65: four controllers of 12 ops, two more input
norms, and the cross-pass message mix.

By op: `MUL_MAT` 977, `MUL` 858, `ADD` 824, `SCALE` 629, `CONCAT` 548, `RMS_NORM` 514, `CONT` 384, `TANH` 289.
Full list: `logs/census-pr-head.txt`.

### 3.2 Bytes (exact, from tensor shapes)

| What | Per token |
|---|---|
| Attention projections, 26.9M weights × 48 visits, Q8_0 | 1,371 MB |
| 8 experts, 63.7M weights × 48 visits, Q8_0 | 3,249 MB |
| Routers, F32, 48 visits | 57 MB |
| Loop controllers, F32, 96 of them | 226 MB |
| Output head, 201,024 × 2304, Q8_0 | 492 MB |
| **Weights** | **5,397 MB** |
| K/V of the 8 full-attention visits, F16, as implemented | 99.5 KB per token of depth |
| The same, read once per K/V head | 33.2 KB per token of depth |
| K/V of the 40 window visits, as implemented | 764 MB, constant once the window cache is full |
| The same, read once per K/V head | 255 MB |

The K/V rows: 8 K/V heads × 128 × 2 bytes × 2 (K and V) = 4,096 bytes per cell and layer, × 3 query heads, plus 6
bytes of bias index and mask per query head = 12,432 bytes. The window cache has 1,536 cells per layer (server
log: 240 MiB for 40 layers) and the kernel walks all of them, masked or not: 40 × 1,536 × 12,432 = 764 MB.

### 3.3 Time (from the ticket's measurements; the GPU was shared with a game stream, so rough)

Decode was 17.5 tok/s at about 5k depth (57.1 ms) and 12.1 tok/s at about 57k (82.6 ms). Model:

    T(depth) = (W + Kw)/B + N·c + s·depth
    W  = 5.40 GB of weights (exact)        B = bytes/s the kernels reach (unknown)
    Kw = 0.76 GB of window K/V (exact, if none of the repeated reads is served from cache)
    N  = dispatches (exact)                c = cost of one glue dispatch (unknown)
    s  = 0.49 µs per token of depth (from the two measurements)   →  T(0) ≈ 54.7 ms

Two unknowns, one equation. What bounds them:

| Assumed c | Glue, N = 7,044 | Leaves for bytes | Implied B, weights only | Implied B, with window K/V |
|---|---|---|---|---|
| 2 µs | 14.1 ms | 40.6 ms | 133 GB/s | 152 GB/s |
| 3 µs | 21.1 ms | 33.6 ms | 161 GB/s | 183 GB/s |
| 4 µs | 28.2 ms | 26.5 ms | 204 GB/s | 232 GB/s |

The memory's theoretical limit is 256 GB/s, which caps c at 4.3 to 4.8 µs.

- N = 7,044 rather than 7,380 because master already fuses the weighted expert sum (48 × 7 ops). Master's top-k
  router fusion may also match; the dump cannot tell, the stats counter can.
- The fused fork's README relays 31.1 → 20.4 ms per token on a Radeon R9700 under HIP. With about 5,000
  dispatches removed that is roughly 2.1 µs each on a desktop host. the host CPU is throttled under GPU load, so 2
  is a floor here and I use 2 to 4.
- s = 0.49 µs is 99.5 KB at 203 GB/s. Either the kernel is memory-bound and B is near 200 GB/s, or the three
  readers of a K/V head share cache and the time is something else. Not known.

## 4. Hypotheses, ranked

| # | Hypothesis | Evidence | Share of a 57 ms token (predicted) |
|---|---|---|---|
| 1 | Weight traffic | 5.40 GB per token, exact | 26 to 41 ms |
| 2 | Launch tax of glue ops | 6,550 of 7,380 ops touch ≤ 100 KB; fused fork +52% on an R9700 | 14 to 28 ms |
| 3 | K/V walked 3× per token | code (`K += nb12*(head/gqa_ratio)`, one block per query head); slope matches the bytes | full layers: 2.5 ms at 5k, 28 ms at 57k, 125 ms at 256k. Window layers: up to 3.3 to 5 ms, constant |
| 4 | Tiny matmuls through rocBLAS sgemm | 48 bias tables per token (`[16,1025] × [16,24]` misses every custom kernel), plus the 4-query and odd-state depth mixes | unknown; per-call cost not measured |
| 5 | Loop-controller matmuls | 288 dispatches, 226 MB of F32 | 1 to 2 ms |
| 6 | Dynamic-conv lag loop | 192 dispatches | 0.4 to 0.8 ms |
| 7 | MoE at batch 1 | experts are already 2 launches per visit; SiTU 9 ops, router 9 ops | 1.7 to 3.5 ms |
| 8 | Bias gather (`kq_b`) | removed by PR #32740; left: an O(depth) index and mask upload per token | below 0.3 ms until 256k |

Ruled out: the wrong attention kernel for head dim 96 (the head dim is 128, and the VEC kernel is the intended
one for a single query).

## 5. What each fix buys

Predictions use c = 2 to 4 µs and the ticket's timings. Nothing here is measured.

| Fix | State | Dispatches | 5k depth | 57k depth | 256k depth (extrapolated) |
|---|---|---|---|---|---|
| PR build today | measured (rough) | 7,044 | 57.1 ms, 17.5 tok/s | 82.6 ms, 12.1 | 180 ms, 5.6 |
| 1. Fused glue chains + bias table | prototyped | 2,396 | 38.5 to 47.8 ms, 21 to 26 | 64 to 73 ms, 13.6 to 15.6 | 162 to 171 ms, 5.9 to 6.2 |
| 2. K/V once per K/V head, alone | prototyped | 7,044 | 0 to −5 ms, up to 19 | 0 to −22 ms, up to 16.5 | 0 to −87 ms, up to 10.7 |
| 1 + 2 | prototyped | 2,396 | 35 to 48 ms, 21 to 29 | 43 to 73 ms, 13.6 to 23 | 76 to 171 ms, 5.9 to 13 |
| 3. Coarser fusion | design | about 1,300 | −2.2 to −4.4 ms more | same | same |
| 4. Q8_0 K/V cache | config | | | halves what is left of the slope | |
| 5. IQ4_NL weights | config | | −10 to −18 ms | same | same |

Arithmetic:

- **Fix 1.** 7,044 − 2,396 = 4,648 dispatches × 2 to 4 µs = 9.3 to 18.6 ms. The fused kernels do the same
  arithmetic on the same few kilobytes, so their own time is noise.
- **Fix 2.** It removes two thirds of the K/V bytes. Full layers: 0.327 µs × 57k = 18.6 ms, × 256k = 84 ms. Window
  layers: 0.51 GB at 152 to 232 GB/s = 2.2 to 3.4 ms at any depth. These are upper bounds. The three blocks that
  read one K/V head run at the same time over the same addresses, so the second and third reader may already be
  served from cache; then the kernel is not memory-bound and the gain shrinks, down to nothing. This is the least
  certain number in the report and the first thing the slot should measure.
- **Bound.** With fixes 1 and 2 at their best, a shallow token is (W + Kw/3)/B + 2,396·c: 42 ms if c = 2, 34 ms if
  c = 4, against a weights-only bound W/B of 36 and 23 ms. What is left is 786 weight dispatches, 96 attention
  launches and about 1,500 glue ops.
- **Fix 3**, not built. What is left outside the chains (`logs/census-kern.txt`): `RMS_NORM` + `MUL` pairs (about
  200), the router chain (9 → 1 or 2, if master's top-k fusion does not already take it), gate matmul + sigmoid +
  mul, the two `SET_ROWS`, the state `GET_ROWS`, and one launch for the projections that share an input.
- **Fix 5.** 4.5 bits per weight against 8.5 on the 5.11 GB of Q8_0 weights: about 2.4 GB less per token, at 133
  to 232 GB/s. ggml-org's README gives KL 0.0095 against 0.0030 for Q8_0.
- Not worth doing: F16 loop controllers (113 MB, under 1 ms).

## 6. The patches

All on `inkling-kern`. The first four are realugbun's commits (`ref/realugbun-inkling2-speed`), cherry-picked.

| Commit | What | Checked how |
|---|---|---|
| 78d0b231 | graph: normalise each depth state once (7,380 → 7,052 ops) | CPU logits equal the PR head |
| 81cc984d | graph: keep each glue chain contiguous. **Conflict resolved by me**: the dense-bias gather is gone on the PR head; I kept the early expansion of the bias projection and sinks so that `PAD MUL_MAT SUB` stays one run | CPU logits equal the PR head; chains contiguous in the dump |
| 18419dcf | CUDA: 11 fused chains (`inkling2.cu`) | builds for HIP; its 82 `INKLING2_*` cases pass CPU-vs-CPU; GPU run pending |
| dc115ceb | CUDA: pad kernel for narrow rows | builds |
| df1ba587 | **new** CUDA: `TABLE` fusion, `PAD MUL_MAT SUB` → one kernel | `INKLING2_TABLE` cases added (4); pass CPU-vs-CPU; GPU run pending |
| ced2bd61 | **new** CUDA: VEC kernel `cols_are_heads` mode, K/V once per K/V head | test extended to compare with the CPU backend; reference verified on CPU; GPU run pending |
| 7ed7f62e | **new** CUDA: `GGML_CUDA_OP_STATS`, `GGML_CUDA_INKLING2_FUSION` | builds |

Notes on each of mine:

- **`TABLE`.** On the PR head the fork's `RELBIAS` chain no longer exists on the GPU (the bias reaches attention as
  a compact table), so it never matches; what remains is the pad of a constant weight and a 16-wide sgemm, every
  visit. The kernel computes the table directly. Like the fork's `MIX`, it is not bit-identical to the library.
  The matcher refuses when the output overlaps an input; whether the allocator ever makes it overlap is only
  visible on the GPU (`fused PAD+2` should read 48.0 per evaluation).
- **K/V once per K/V head.** The VEC kernel already handles several query columns per block. The new mode makes
  the columns the 3 query heads of a K/V head: same K/V loads, per-head Q, softmax state, sink and bias row;
  shared mask row and bias indices. `launch_fattn` already had the grid for it (`ncols2`). Taken only for D = 128,
  GQA ratio 3, one query, no ALiBi, F16 or Q8_0 K/V. Not bit-identical to the old kernel, because fewer z blocks
  changes how the K/V range is split.
- **Test.** The PR's test compares with an analytic value where V is constant across the width, and its sinks grow
  with the head index, so it would not notice much at 24 heads. The added cases compare with the CPU backend
  computing the same attention from a dense per-head mask, with data built so that heads differ by more than 1.6
  and the tolerance is 0.02. Shapes: 24/8 heads, one query, 1k, 4k (Q8_0) and 32k cells, with and without rows,
  plus a prefill and a width-64 case that must not take the new path.
- **Switches**, all default on: `GGML_CUDA_INKLING2_FUSION=0`, `GGML_CUDA_FATTN_VEC_GQA=0`. `GGML_CUDA_OP_STATS=1`
  prints dispatches and fused chains per graph evaluation at exit; `=2` adds synchronized wall time per kind of
  op, with `MUL_MAT` keyed by operand type and shape. `opstats.py` turns that into c, B and GB/s per weight
  kernel.

Risks I could not retire without the GPU:

- The fused matchers check tensor memory overlap, which depends on the GPU allocator. A chain that fails to match
  costs speed, not correctness.
- The new attention mode has never executed. Its logic was reviewed line by line against the multi-column path it
  reuses, and it compiles for gfx1151. The test is the gate.
- `GGML_CUDA_OP_STATS=2` adds a stream synchronize per op; if HIP sleeps in it, the glue times are inflated. The
  summary prints the synchronized total next to the normal token time so this shows.

## 7. What was run, and how to repeat it

CPU only. Logs in `logs/`.

| Check | Result |
|---|---|
| Synthetic model loads and decodes on the CPU | 4.2 tok/s at 4 threads |
| Graph size against the server log | 11,826 − 432 = 11,394, exact |
| Ported graph against PR head, CPU logits, batch 1 and 64 (`llama-perplexity --kl-divergence`) | mean KLD 0.000000, max 0.000002, same top token 100% |
| `test-backend-ops -b CPU -o 'INKLING2.*'` | 86/86 |
| `test-flash-attn-indexed-bias`, CPU build | reference for all 10 new cases is finite, head distance 1.68 to 1.92 |
| Builds at `-j 8`: CPU, ROCm (gfx1151), Vulkan | all three clean |
| New kernels present in `libggml-hip.so` | 4 `flash_attn_ext_vec<128, 3, …>`, `inkling2_table_f32` |

    python3 kernel-research/make_synth_gguf.py ~/models/inkling-2-mini-gguf/Inkling-2-Mini-Q8_0.gguf \
        kernel-research/synth/inkling2-mini-synth-e8-Q8_0.gguf                       # 40 s, 3.4 GB
    kernel-research/build-kern.sh [cpu] [rocm] [vulkan]                              # -j 8
    GGML_SCHED_DEBUG=2 build-cpu/bin/llama-bench -m <synth> -p 0 -n 2 -r 1 -fa on -t 4 -ngl 0 --no-warmup -v \
        > kernel-research/logs/census-kern-cpu.log 2>&1
    python3 kernel-research/census.py kernel-research/logs/census-kern-cpu.log --fused
    # PR-head CPU logits (the baseline tree is an export of 808fd1c4 in build-base-src/):
    build-base-src/build/bin/llama-perplexity -m <synth> -f README.md -c 64 -b 1 --chunks 4 -t 8 -ngl 0 -fa on \
        --kl-divergence-base kernel-research/logs/synth-base-cpu-b1.kld
    build-cpu/bin/llama-perplexity <same> --kl-divergence-base <that file> --kl-divergence

The synthetic model has 8 experts instead of 128 and random weights. A decode step runs the same node list on the
same shapes and reads the same bytes as the real model, which also reads exactly 8 experts per visit. It is for
counting, correctness and speed, never for quality.

## 8. The GPU slot: `bench.sh`

`GPU_SLOT=1 kernel-research/bench.sh <stage>`, in this order. It uses no timeouts and sends no signals.

| Stage | Backend | Answers | Time |
|---|---|---|---|
| `preflight` | none | binaries, RAM, who is on the GPU | seconds |
| `vk-tests` | Vulkan | do the tests and the reordered graph hold on a GPU at all | 2 min |
| `vk-bench` | Vulkan | Vulkan decode speed at depth 0 and 16k | 3 min |
| `rocm-tests` | ROCm | GQA kernel on/off against the CPU; fused chains against the CPU; fused logits against unfused | 5 min |
| `rocm-stats` | ROCm | real dispatch counts, which chains matched, c, B, GB/s per weight kernel | 10 min |
| `rocm-bench` | ROCm | tok/s for `pr`, `unfused`, `fused`, `fused+gqa` at depth 0, 16k, 64k | 20 min |
| `real-bench` | ROCm | the same on the 27.8 GB model; needs its server stopped by its owner | 25 min |
| `rocprof` | ROCm | kernel trace; needs `ROCPROF_OK=1` | 5 min |

- 256k depth: `DEPTHS=262144 CONFIGS="fused fused+gqa"`. The fill alone is about 12 minutes per repetition.
- The only `rocprofv3` on the box is in the ROCm 10.1 wheels of two venvs; the system runtime is ROCm 7.1. It may
  refuse to attach. `rocm-stats` level 2 gives the same answers without it, so `rocprof` is last and optional.
- What decides the next step: if `opstats.py` shows the Q8_0 weight kernels well under the box's bandwidth, the
  next kernel to write is the matrix-vector one, not more fusion. If they are at it, the remaining levers are
  fix 3 and smaller weights.

## 9. Open points

- Issue #32757 and the PR thread were not re-read in this run (`gh` is denied to this agent). The earlier run
  read the PR thread; its findings are in context #978.
- Prefill was not studied. The TILE kernel also runs one K/V pass per query head with the bias; the same sharing
  would apply.
- The house rule about strided-transpose fp32 GEMMs: the PR build's decode graph calls rocBLAS sgemm (transposed
  operand, as ggml always does) at least 48 times per token and has not hung. After fix 1 the decode graph should
  call it zero times. `rocm-stats` level 2 shows any `MUL_MAT [f32 …] [f32 AxB]` row that remains.
- Upstream: `TABLE` and the K/V kernel are candidates for the PR's author and for realugbun's branch. Nothing was
  sent. Drafts on request, once the GPU tests have passed.
- Disk: this run added about 4.5 GB under the worktree (3.2 GB synthetic model, builds, logit files). `/` has
  27 GB free. `kernel-research/synth/` can be deleted and rebuilt in 40 seconds.
