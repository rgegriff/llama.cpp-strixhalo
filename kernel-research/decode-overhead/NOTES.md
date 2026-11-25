# Decode step overhead of Inkling 2 Mini on gfx1151: launches, idle time, HIP graphs

Question: of the ~31 ms that one decoded token of Inkling 2 Mini (IQ4_NL, q8_0 KV cache) takes on gfx1151, how much
is GPU idle time or launch overhead rather than kernel time, and are HIP graphs in use?

Short answers:

- **HIP graphs are in use and they buy nothing here.** The step is GPU-bound with or without them.
- **The GPU never waits for the host.** The host enqueues a step in 10.5 ms (direct) or 0.75 ms (graph replay) and
  the stream then needs another 21 to 31 ms to drain.
- **About a fifth of the step is the launch floor of the device**: 3,100 kernels at ~2 µs each. A kernel that does
  next to nothing costs 2.1 µs on the GPU side however it is submitted.
- Removing kernels is therefore the lever. Three changes take 3,128 kernels per step down to 2,400, all of them
  bit-identical to the unfused ops.

## Tools

All in `ggml/src/ggml-cuda/ggml-cuda.cu`, read from the environment:

- `GGML_CUDA_STEP_TIMING=1`: one line per graph evaluation: mode (`direct`, `capture`, `replay`), host time to
  enqueue, time for the stream to drain, and for a capture the number of kernel nodes in the HIP graphs. It
  synchronizes before and after every evaluation, nothing in between.
- `GGML_CUDA_OP_STATS=1` now also counts, while a graph is captured, the kernels each op launches
  (`hipStreamGetCaptureInfo_v2` + `hipGraphGetNodes` around the dispatch). `GGML_CUDA_OP_STATS_SHAPES=1` keys the
  matmuls by shape at level 1, `GGML_CUDA_OP_STATS_TRACE=<n>` prints every dispatch of evaluation n.
- `launch_floor.cpp` (this directory): a chain of trivial kernels, launched directly, replayed from HIP graphs, and
  launched with a synchronize after each one.

There is no kernel tracer on this machine: the packaged HIP runtime (7.1) is not linked against
rocprofiler-register, so rocprofv3 (only present in ROCm 10 wheels) has nothing to attach to. The numbers below
come from host timestamps around a synchronized stream, from kernel counts of the captured graphs, and from two
calibrations of what one launch costs.

## HIP graphs

Build has `GGML_HIP_GRAPHS=ON`. For this model nothing disables them:

- `ggml_cuda_graph_check_compability` only rejects a `MUL_MAT_ID` that takes the synchronizing fallback. At 1 and 8
  tokens the expert matmuls go to MMVQ (IQ4_NL) or MMQ, neither synchronizes.
- The node properties do not change between tokens: llama.cpp reuses the graph (`graphs reused` in the perf line),
  so the backend sees the same `uid` and skips the per-node comparison.
- The recurrent conv state, the indexed-bias attention and the fused chains are all plain kernel launches.

What happens (`GGML_CUDA_GRAPH_DEBUG=1`, `-v`): evaluation 1 runs directly, evaluation 2 is captured as 21 graphs
(the 10,521 ggml nodes in chunks of 512), every later evaluation of the same shape is a replay. A change of batch
size resets the warm-up: one direct evaluation, one capture, then replays again. `GGML_CUDA_OP_STATS=2` turns
graphs off (it synchronizes inside the step).

| evaluation, n=1 | host enqueue | drain | total |
|---|---|---|---|
| direct (graphs off) | 10.2 to 11.0 ms | 20.6 to 22.3 ms | 30.7 to 33.0 ms |
| capture | 34.7 to 37.3 ms | 4.4 to 6.3 ms | 41.0 to 41.7 ms |
| replay | 0.7 to 0.9 ms | 31.2 to 32.8 ms | 31.9 to 33.3 ms |

| evaluation, n=8 (pp8) | host enqueue | drain | total |
|---|---|---|---|
| direct (graphs off) | 16.5 to 21.7 ms | 56.8 to 72.8 ms | 74.4 to 93.0 ms |
| capture | 41.0 ms | 39.4 ms | 80.4 ms |
| replay | 0.6 to 1.2 ms | 71.9 to 79.1 ms | 73.1 to 79.7 ms |

(Totals include the two synchronizes of the timing mode, about 1 ms over an undisturbed step.)

A replay takes as long as a direct evaluation: the host is 10 to 20 ms ahead of the GPU either way. What graphs do
cost is the capture: +5 to +10 ms in the runs above, +1 ms in a later run with a faster host clock (the host
frequency under GPU load varies; the capture is host work). It is paid once per run of equal batch sizes. A decode
loop pays it once. A speculative-decoding verify whose batch size changes from step to step pays it every time two
batches of the same size follow each other, and gets nothing back. `GGML_CUDA_DISABLE_GRAPHS=1` is worth an A/B in
the server; llama-bench at a fixed batch size cannot show the difference (see the results).

## The launch floor

`launch_floor 2600 512 12` (2,600 kernels of 1,024 elements, each reading what the previous one wrote; median of the
last third of a 12 s run):

| mode | host enqueue per kernel | total per kernel |
|---|---|---|
| direct | 0.95 µs | 2.25 µs |
| HIP graphs, 512 kernels each | 0.10 µs | 2.07 µs |
| direct, one graph of 2,600 | 1.66 µs | 2.29 µs |
| synchronize after each launch | 8.57 µs | 8.57 µs |

The host is not the limit in either mode: the device completes a trivial kernel every ~2.1 µs and no faster. That
is the price of a dispatch on this GPU (packet, wave launch, completion), not of the 1,024 multiplications.

In the model the same number comes out. With the Inkling 2 fusions off (`GGML_CUDA_INKLING2_FUSION=0`) a step has
3,796 more kernels that do the same arithmetic, and tg128 goes from 31.3 to 25.55 t/s: 7.2 ms more, **1.9 µs per
kernel**.

A step of 3,128 kernels therefore carries about 3,128 x 2.0 µs = **6.3 ms of launch floor, 20% of 31.5 ms**. Of the
3,128 kernels roughly 740 do the heavy work (593 matrix-vector products, 96 expert matmuls, 48 attentions); the
other ~2,400 are glue and activation quantization, 4.7 ms.

`GGML_CUDA_OP_STATS=2` (synchronized per-op times) reports 53.3 ms for the same step. One launch with a synchronize
costs 8.6 µs in the microbenchmark and 9.8 µs in the model (`SCALE`, one tiny kernel), against 2.1 µs in a stream.
Subtracting 7.7 µs per dispatch (2,586 dispatches, 19.9 ms) gives 33.4 ms: the synchronized profile is the stream
profile plus a constant per dispatch, which is why it overstates small ops.

### Where the kernels were (before)

Per decoded token, 2,607 dispatches and 3,128 kernels; a layer visit is ~62 kernels, 48 visits per token.

| per token | dispatches | kernels |
|---|---|---|
| quantized `MUL_MAT` (q8_0 projections) | 353 | 706 |
| `MUL_MAT_ID` (expert gate_up, down) | 96 | 192 |
| `FLASH_ATTN_EXT` | 48 | 96 |
| f32 `MUL_MAT` (4 K/V rotations per visit, router logits) | 240 | 240 |
| `SCALE` (192 in the SiTU, 96 in the router weights) | 291 | 291 |
| `TANH`, `MUL` (SiTU) | 96 + 64 | 160 |
| `SUM_ROWS`, `DIV` (router weights) | 48 + 48 | 96 |
| `SET_ROWS` (KV cache store) | 96 | 96 |
| `FILL`, `GET_ROWS` (recurrent state, bias fill) | 72 + 49 | 121 |
| `RMS_NORM` and 9 single ops | 63 | 63 |
| fused chains (48 of them the `SIGMOID MUL` of the SiTU) | 1,043 | 1,043 |

Every quantized matrix-vector product is two kernels: the activations are quantized to Q8_1 by one kernel, then
multiplied. The six projections of an attention block (k, v, rel, sink, q, gate) read the same normalized vector and
quantized it six times.

## Changes

1. **Q8_1 activation cache** (`GGML_CUDA_MMVQ_Q8_CACHE`, default on; any model). The backend context keeps the
   quantization of the activations of the last quantized matrix-vector product in a buffer of its own. A product
   whose `src1` has the same data pointer, shape and strides takes it instead of quantizing again. It is dropped
   when an evaluation starts and whenever a computed node (any node of a fused chain) overlaps the source range, so
   an in-place rewrite of the activations is seen (`test-backend-ops -o MUL_MAT_VEC_SHARED`). The buffer is fixed
   (256 KiB, larger inputs are not cached) because captured graphs keep its address. 706 -> ~470 kernels.
2. **SiTU without a per-expert scale** (`GGML_CUDA_INKLING2_SITU_PLAIN`, default on). The SiTU fusion needed the
   `MUL` by the expert scale in front of the chain; Inkling 2 Mini has no such scale, so its chain
   (`SCALE TANH SCALE SIGMOID MUL VIEW SCALE TANH SCALE MUL`) ran as 8 kernels per visit. Now 1.
   `ggml_can_fuse_subgraph` refuses a chain that contains a view of a tensor computed outside it (the up half of the
   expert output), so the matcher checks the two halves and the joining nodes itself.
3. **Router weight normalization by `sum + eps`** (`GGML_CUDA_TOPK_MOE_NORM_EPS`, default on). The top-k MoE kernel
   fused the normalization only in its clamped form (`SUM_ROWS CLAMP DIV`); with `expert_weights_norm_eps` the graph
   has `SUM_ROWS SCALE(1, eps) DIV`, 4 more kernels per visit. In this form the kernel adds the weights up by rank
   (the lanes `SUM_ROWS` uses) and divides instead of multiplying by the reciprocal, which gives the bits of the
   unfused ops. A first version that reused the clamped form's arithmetic was 1 ulp off in the weights; this
   model amplifies that to KL 0.0020 (98.0% same top token) over 2,048 tokens, which is why the exact form is used.

Kernels per decoded token: 3,128 -> 2,400 (-23%). At 8 tokens (pp8): 3,669 -> 3,141 (-14%).

## Results

Before = the tree without the three changes, after = with them, same build flags. llama-bench, IQ4_NL, q8_0 KV,
`-r 3`, two rounds interleaved (before, after, after without graphs).

| | before | after | after, `GGML_CUDA_DISABLE_GRAPHS=1` |
|---|---|---|---|
| tg128, depth 0 | 31.61, 31.58 | 33.55, 32.12 | 33.53, 33.29 |
| tg128, depth 65536 | 25.42, 25.06 | 26.71, 26.69 | |
| pp8, depth 0, `-r 20` | 109.45, 107.73 | 106.36, 112.32 | 112.30, 112.38 (before: 109.82, 109.60) |
| pp8, depth 65536 | 105.47, 103.41 | 104.13, 106.37 | |

- Decode: +3.9% at depth 0 (mean of two; the second "after" run is low against the graphs-off run that followed
  it on the same binary, 33.29), +5.8% at depth 65536. In step time 31.65 -> 30.45 ms and 39.6 -> 37.45 ms: 1.2 to
  2.2 ms for 728 kernels, 1.6 to 3.0 µs each.
- 8 tokens: +2% without graphs, inside the spread of the runs with them (a pp8 repetition is one evaluation, and
  the first two of a run are the direct and the capture one). 528 kernels less of a 73 ms step is 1.1 ms.
- Graphs on or off makes no difference at a fixed batch size, as the step timing said.

Correctness: `llama-perplexity --kl-divergence` on wikitext-2 test, 2 chunks of 1,024 tokens, against the tree
without the changes, at batch size 1, 8 and 512: mean KL 0.000000, same top token 100.000% at all three (two runs
of the same build give the same figures; maximum KL 0.00005 is the resolution of the logits file). With all three
switches off, and with the SiTU fusion alone, at batch size 1: the same. `test-backend-ops -o INKLING2_SITU` (9), `-o TOPK_MOE` (704),
`-o MUL_MAT_VEC_SHARED` (12), `-o MUL_MAT` (2,724), `-o MUL_MAT_ID` (1,087) pass on gfx1151.

## What is left

Per layer visit ~49 kernels remain. Candidates, in order of kernels per token:

- the four f32 rotation matmuls around the KV cache (192): each could join the chain before or after it;
- the second kernel of the remaining quantized products (attention output, expert gate_up and down, ~160): the
  producer of the activations would have to write Q8_1 itself;
- the second kernel of `FLASH_ATTN_EXT` (48), the two `SET_ROWS` of a visit as one (48), the `FILL` and `GET_ROWS`
  of the recurrent state and of the bias (121).

All of them together are ~570 kernels, 1.1 ms. Beyond that the step is the 740 heavy kernels and the fused chains,
one launch each.
