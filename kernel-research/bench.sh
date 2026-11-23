#!/bin/bash
# Measurement plan for the GPU slot (Inkling 2 Mini kernels on gfx1151).
# Vulkan first, ROCm after, the 27.8 GB model and rocprofv3 last. Nothing here has been run on the GPU yet.
#
#   GPU_SLOT=1 kernel-research/bench.sh <stage> [<stage> ...]
#
# Stages, in the order to run them (each stops at its first failure; results go to kernel-research/results/<stamp>/):
#   preflight    no GPU work: binaries, model files, RAM, who is on the GPU
#   vk-tests     Vulkan: the indexed-bias test, the chain tests, logits of the synthetic model against the CPU
#   vk-bench     Vulkan: decode speed of the synthetic model at depth 0 and 16k
#   rocm-tests   ROCm: the same tests with the fusions and the GQA kernel on and off, logits against unfused ROCm
#   rocm-stats   ROCm: dispatch counts and the synchronized per-op profile (GGML_CUDA_OP_STATS), 4 configurations
#   rocm-bench   ROCm: llama-bench of the synthetic model at DEPTHS, CONFIGS
#   real-bench   ROCm: the same on the real 27.8 GB model (needs the :8291 server stopped by its owner)
#   rocprof      ROCm: rocprofv3 kernel trace (the only rocprofv3 on the box is a ROCm 10.1 wheel, the runtime is 7.1:
#                it may refuse to attach; rocm-stats level 2 is the profile that does not depend on it)
#
# Environment:
#   GPU_SLOT=1      required for every stage but preflight: you have been handed the GPU.
#                   The script cannot see a booking (the other sessions' servers come and go); this flag is the gate.
#   DEPTHS          default "0,16384,65536"; add 262144 for the long run (the fill alone is ~12 min per repetition)
#   CONFIGS         default "pr unfused fused fused+gqa" (see config_env below)
#   N_GEN, REPS     tokens per measurement (64) and repetitions (2)
#   KV              K/V cache type, f16 (default) or q8_0
#   FORCE=1         run although something answers on :8291 (the Inkling server), only if you were told it is idle
#   ROCPROF_OK=1    required for the rocprof stage
#
# HOUSE RULES, do not edit them out:
#   - Never SIGKILL a ROCm process and never wrap one in `timeout`: a ROCm job killed in mid-dispatch resets the GPU
#     and takes down every other ROCm job and any game stream. If a run hangs, Ctrl-C once (SIGINT) and wait;
#     if it still hangs, stop and ask. This script uses no timeouts and sends no signals.
#   - Host RAM: the halogen brain stalls when free RAM nears 2 GiB. Every load checks MemAvailable first.
#   - Nothing here writes outside this worktree.

set -u
set -o pipefail

KERN="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
FA_TREE="$HOME/src/llama.cpp-inkling-fa"          # PR #32740 as the server runs it; read and execute only
SYNTH="$KERN/kernel-research/synth/inkling2-mini-synth-e8-Q8_0.gguf"
REAL="$HOME/models/inkling-2-mini-gguf/Inkling-2-Mini-Q8_0.gguf"
TEXT="$KERN/README.md"
CPU_KLD="$KERN/kernel-research/logs/synth-base-cpu-b1.kld"   # CPU logits of the PR-head graph, written by this run
ROCPROF="${ROCPROF:-$HOME/.venvs/minicpmo-rocm10/bin/rocprofv3}"

DEPTHS="${DEPTHS:-0,16384,65536}"
CONFIGS="${CONFIGS:-pr unfused fused fused+gqa}"
N_GEN="${N_GEN:-64}"
REPS="${REPS:-2}"
KV="${KV:-f16}"

OUT="$KERN/kernel-research/results/$(date +%F-%H%M)"

say()  { printf '\n== %s\n' "$*"; }
die()  { printf 'bench.sh: %s\n' "$*" >&2; exit 1; }

avail_gib() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }

need_ram() { # need_ram <GiB> <what>
    local have; have=$(avail_gib)
    [ "$have" -ge "$1" ] || die "$2 needs $1 GiB available, there are $have. Free RAM first (see \`free -g\`)."
    echo "RAM: $have GiB available, $2 needs $1"
}

need_slot() {
    [ "${GPU_SLOT:-0}" = 1 ] || die "the GPU is booked by default. Run with GPU_SLOT=1 only once the slot is yours."
    if ss -ltn 2>/dev/null | grep -qE ':(8291|8293) ' && [ "${FORCE:-0}" != 1 ]; then
        die "something listens on :8291 or :8293 (an Inkling server). Its batch owns the GPU; FORCE=1 only if you were told it is idle."
    fi
    mkdir -p "$OUT"
}

# run <log name> <command...>: run in the foreground, keep the output, stop the stage on failure
run() {
    local log="$OUT/$1.log"; shift
    echo "+ $*" | tee "$log"
    "$@" >> "$log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || { tail -15 "$log"; die "exit $rc, see $log"; }
}

# The four configurations. `pr` is the untouched PR #32740 build the server runs. The other three are the kern build:
# one binary, switched by environment, so that a difference is the kernels and nothing else.
config_bin() { # config_bin <config> <tool>
    case "$1" in
        pr) echo "$FA_TREE/build-rocm/bin/$2" ;;
        *)  echo "$KERN/build-rocm/bin/$2" ;;
    esac
}
config_env() {
    case "$1" in
        pr)        echo "" ;;
        unfused)   echo "GGML_CUDA_INKLING2_FUSION=0 GGML_CUDA_FATTN_VEC_GQA=0" ;;   # the graph changes only
        fused)     echo "GGML_CUDA_INKLING2_FUSION=1 GGML_CUDA_FATTN_VEC_GQA=0" ;;   # + fused glue chains and bias table
        fused+gqa) echo "GGML_CUDA_INKLING2_FUSION=1 GGML_CUDA_FATTN_VEC_GQA=1" ;;   # + K/V read once per K/V head
        *) die "unknown config $1" ;;
    esac
}

stage_preflight() {
    say "preflight (no GPU work)"
    for f in "$KERN"/build-{cpu,vulkan,rocm}/bin/{llama-bench,llama-perplexity,test-flash-attn-indexed-bias,test-backend-ops} \
             "$FA_TREE/build-rocm/bin/llama-bench" "$SYNTH" "$REAL" "$TEXT" "$CPU_KLD"; do
        [ -e "$f" ] && echo "ok       $f" || echo "MISSING  $f"
    done
    echo "kern HEAD: $(git -C "$KERN" log --oneline -1)"
    echo "RAM available: $(avail_gib) GiB"
    echo "listening on :8291 (Inkling server): $(ss -ltn 2>/dev/null | grep -c ':8291 ')"
    echo "llama processes (the house models stay resident; a booked slot is not visible here, ask):"
    pgrep -a -x 'llama-server|llama-bench|llama-perplexit' | cut -c1-150 || echo "  none"
    [ -x "$ROCPROF" ] && echo "rocprofv3: $ROCPROF" || echo "rocprofv3: not found at $ROCPROF"
    echo "to rebuild the synthetic model: python3 kernel-research/make_synth_gguf.py $REAL $SYNTH"
    echo "to rebuild the CPU logits: see REPORT.md, section 7"
}

stage_vk_tests() {
    need_slot; need_ram 12 "the synthetic model on Vulkan"
    say "Vulkan: indexed-bias flash attention against the CPU"
    run vk-test-indexed-bias "$KERN/build-vulkan/bin/test-flash-attn-indexed-bias"
    grep -h 'max error' "$OUT/vk-test-indexed-bias.log" | tail -12
    say "Vulkan: the Inkling 2 op chains against the CPU (Vulkan runs them unfused)"
    run vk-test-chains "$KERN/build-vulkan/bin/test-backend-ops" -b Vulkan0 -o 'INKLING2.*'
    tail -3 "$OUT/vk-test-chains.log"
    say "Vulkan: logits of the synthetic model, decode path, against the CPU logits of the PR-head graph"
    run vk-kld "$KERN/build-vulkan/bin/llama-perplexity" -m "$SYNTH" -f "$TEXT" -c 64 -b 1 --chunks 4 -ngl 99 -fa on \
        --kl-divergence-base "$CPU_KLD" --kl-divergence
    grep -E 'Mean    KLD|Maximum KLD|Same top p' "$OUT/vk-kld.log"
    echo "Random weights amplify rounding, so GPU-vs-CPU KLD is a noise floor to compare the ROCm numbers with, not a pass mark."
}

stage_vk_bench() {
    need_slot; need_ram 12 "the synthetic model on Vulkan"
    say "Vulkan: decode speed of the synthetic model, depth 0 and 16k"
    run vk-bench "$KERN/build-vulkan/bin/llama-bench" -m "$SYNTH" -d 0,16384 -p 0 -n "$N_GEN" -r "$REPS" -fa on -ngl 99 \
        -ctk "$KV" -ctv "$KV" -o md
    grep -E '^\|' "$OUT/vk-bench.log"
}

stage_rocm_tests() {
    need_slot; need_ram 12 "the synthetic model on ROCm"
    say "ROCm: indexed-bias flash attention against the CPU, GQA decode kernel on, then off"
    run rocm-test-indexed-bias-gqa1 env GGML_CUDA_FATTN_VEC_GQA=1 "$KERN/build-rocm/bin/test-flash-attn-indexed-bias"
    run rocm-test-indexed-bias-gqa0 env GGML_CUDA_FATTN_VEC_GQA=0 "$KERN/build-rocm/bin/test-flash-attn-indexed-bias"
    paste -d'\n' <(grep -h 'vs CPU' "$OUT/rocm-test-indexed-bias-gqa1.log" | sed 's/^/gqa=1 /') \
                 <(grep -h 'vs CPU' "$OUT/rocm-test-indexed-bias-gqa0.log" | sed 's/^/gqa=0 /')
    say "ROCm: the fused Inkling 2 chains against the unfused CPU ops"
    run rocm-test-chains "$KERN/build-rocm/bin/test-backend-ops" -b ROCm0 -o 'INKLING2.*'
    tail -3 "$OUT/rocm-test-chains.log"
    say "ROCm: logits of the synthetic model. Reference = unfused kern build; then each configuration against it"
    run rocm-kld-ref env $(config_env unfused) "$KERN/build-rocm/bin/llama-perplexity" -m "$SYNTH" -f "$TEXT" -c 64 -b 1 \
        --chunks 4 -ngl 99 -fa on --kl-divergence-base "$OUT/rocm-unfused-b1.kld"
    for c in fused fused+gqa; do
        run "rocm-kld-$c" env $(config_env $c) "$KERN/build-rocm/bin/llama-perplexity" -m "$SYNTH" -f "$TEXT" -c 64 -b 1 \
            --chunks 4 -ngl 99 -fa on --kl-divergence-base "$OUT/rocm-unfused-b1.kld" --kl-divergence
        echo "$c vs unfused:"; grep -E 'Mean    KLD|Maximum KLD|Same top p' "$OUT/rocm-kld-$c.log"
    done
    run rocm-kld-cpu env $(config_env unfused) "$KERN/build-rocm/bin/llama-perplexity" -m "$SYNTH" -f "$TEXT" -c 64 -b 1 \
        --chunks 4 -ngl 99 -fa on --kl-divergence-base "$CPU_KLD" --kl-divergence
    echo "unfused ROCm vs CPU (the noise floor):"; grep -E 'Mean    KLD|Maximum KLD|Same top p' "$OUT/rocm-kld-cpu.log"
    echo "Pass: fused and fused+gqa differ from unfused by no more than unfused ROCm differs from the CPU."
    echo "The fusions claim the bits of the unfused ops except the MIX and TABLE kernels; the GQA kernel changes the split of K/V."
}

stage_rocm_stats() {
    need_slot; need_ram 12 "the synthetic model on ROCm"
    say "ROCm: dispatches per token (level 1) and the synchronized per-op profile (level 2), depth 0 and 16k"
    for c in unfused fused fused+gqa; do
        for d in 0 16384; do
            for level in 1 2; do
                local name="rocm-stats-$c-d$d-l$level"
                run "$name" env $(config_env $c) GGML_CUDA_OP_STATS=$level "$KERN/build-rocm/bin/llama-bench" -m "$SYNTH" \
                    -d "$d" -p 0 -n 32 -r 1 -fa on -ngl 99 -ctk "$KV" -ctv "$KV" --no-warmup -o md
                python3 "$KERN/kernel-research/opstats.py" "$OUT/$name.log" > "$OUT/$name.summary.txt"
            done
            echo "--- $c, depth $d"; head -2 "$OUT/rocm-stats-$c-d$d-l1.summary.txt"
            sed -n '/^group/,/^weights:/p' "$OUT/rocm-stats-$c-d$d-l2.summary.txt"
        done
    done
    echo "The depth fill of -d 16384 is in the same totals (prefill graphs): read the decode numbers from depth 0 and"
    echo "only the FLASH_ATTN rows from depth 16384. Full tables: $OUT/rocm-stats-*.summary.txt"
    echo "all master fusions off as well, for the raw count: GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_OP_STATS=1 ..."
}

bench_model() { # bench_model <tag> <model> <GiB needed>
    need_slot; need_ram "$3" "$1"
    say "ROCm: llama-bench $1, depths $DEPTHS, configs $CONFIGS, $N_GEN tokens x $REPS, K/V $KV"
    echo "depth fill at ~350 tok/s: 16k ~45 s, 64k ~3 min, 256k ~12 min, per repetition and configuration"
    for c in $CONFIGS; do
        run "bench-$1-$c" env $(config_env $c) "$(config_bin $c llama-bench)" -m "$2" -d "$DEPTHS" -p 0 -n "$N_GEN" \
            -r "$REPS" -fa on -ngl 99 -ctk "$KV" -ctv "$KV" -o md
        echo "--- $c"; grep -E '^\|' "$OUT/bench-$1-$c.log"
    done
}

stage_rocm_bench() { bench_model synth "$SYNTH" 14; }

stage_real_bench() {
    echo "The real model needs 28 GiB plus the K/V cache. Its server on :8291 must have been stopped by its owner."
    bench_model real "$REAL" 40
}

stage_rocprof() {
    need_slot; need_ram 12 "the synthetic model on ROCm"
    [ "${ROCPROF_OK:-0}" = 1 ] || die "rocprof stage: set ROCPROF_OK=1. The profiler is from a ROCm 10.1 wheel and the runtime is 7.1; a crash of a profiled ROCm process is the kind of event the house rules are about. Run it last."
    [ -x "$ROCPROF" ] || die "no rocprofv3 at $ROCPROF"
    for c in unfused fused+gqa; do
        say "rocprofv3 kernel trace, $c, 16 tokens at depth 0"
        run "rocprof-$c" env $(config_env $c) "$ROCPROF" --kernel-trace --stats --summary -d "$OUT/rocprof-$c" -o decode -- \
            "$KERN/build-rocm/bin/llama-bench" -m "$SYNTH" -d 0 -p 0 -n 16 -r 1 -fa on -ngl 99 --no-warmup -o md
        ls "$OUT/rocprof-$c" 2>/dev/null | head
    done
    echo "kernel_stats.csv has calls and total/average ns per kernel: launches per token = sum(calls)/16."
    echo "If no kernel rows were written, the profiler did not attach to this runtime; use the rocm-stats level 2 tables."
}

[ $# -ge 1 ] || { sed -n '2,32p' "$0"; exit 1; }
for stage in "$@"; do
    case "$stage" in
        preflight)  stage_preflight ;;
        vk-tests)   stage_vk_tests ;;
        vk-bench)   stage_vk_bench ;;
        rocm-tests) stage_rocm_tests ;;
        rocm-stats) stage_rocm_stats ;;
        rocm-bench) stage_rocm_bench ;;
        real-bench) stage_real_bench ;;
        rocprof)    stage_rocprof ;;
        *) die "unknown stage $stage" ;;
    esac
done
if [ -d "$OUT" ]; then printf '\nresults: %s\n' "$OUT"; fi
