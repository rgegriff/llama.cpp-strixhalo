#!/usr/bin/env python3
"""Op census of an Inkling 2 decode graph, from a GGML_SCHED_DEBUG=2 dump of the CPU backend.

  GGML_SCHED_DEBUG=2 build-cpu/bin/llama-bench -m <gguf> -p 0 -n 2 -r 1 -fa 1 -t 4 -ngl 0 --no-warmup -v > dump.log 2>&1
  python3 kernel-research/census.py dump.log [--fused]

The dump lists every node that is not a view (RESHAPE/VIEW/PERMUTE/TRANSPOSE), in evaluation order. On the GPU every
such node is one dispatched op (>= 1 kernel launch) unless its tensor is empty, so the count is the launch count of the
unfused graph. Two corrections turn the CPU graph into the ROCm graph of the PR #32740 build:
  - the CPU has no indexed-bias flash attention, so its graph still gathers the dense bias: CONT GET_ROWS CONT CPY ADD
    CPY per layer visit (the nodes between the bias table SUB and the FILL of the sinks). They are counted apart.
  - the two empty recurrent-state nodes per visit (FILL and CPY of zero rows) are no-ops.
--fused additionally counts the chains that ggml-cuda/inkling2.cu replaces by one kernel each, by their op sequence in
this dump. That is a count by pattern, not a run of the matcher: its shape and memory-overlap checks need the GPU run
(GGML_CUDA_OP_STATS=1 prints the real numbers).
"""
import argparse
import collections
import re
import sys

NODE = re.compile(r"^node #\s*(\d+) \(\s*([A-Z_0-9]+)\):\s+(.{20}) \(\s*(\S+)\) \[")

# the chains of ggml-cuda/inkling2.cu as sequences of dispatched ops (views dropped), longest first
CHAINS = [
    ("CONV_DYN",   ["CONCAT", "CPY", "SSM_CONV"] + ["CONT", "MUL", "ADD"]*4 + ["ADD", "SCALE"]),
    ("CONV_STATE", ["CONCAT", "CPY", "SSM_CONV", "ADD", "SCALE", "ADD"]),
    ("CONV_STATE", ["CONCAT", "CPY", "SSM_CONV", "ADD", "SCALE"]),
    ("CLIP",       ["MUL", "SQR", "SUM_ROWS", "SQRT", "SCALE", "FILL", "DIV", "CLAMP", "MUL"]),
    ("CLIP",       ["SQR", "SUM_ROWS", "SQRT", "SCALE", "FILL", "DIV", "CLAMP", "MUL"]),
    ("SITU",       ["SCALE", "TANH", "SCALE", "SIGMOID", "MUL", "SCALE", "TANH", "SCALE", "MUL"]),
    ("MIX",        ["RMS_NORM", "CONCAT", "CONT", "REPEAT", "MUL_MAT", "SOFT_MAX", "MUL_MAT"]),
    ("CTRL_DOWN",  ["RMS_NORM", "RMS_NORM", "CONCAT", "MUL_MAT"]),
    ("CTRL_OUT",   ["MUL_MAT", "MUL_MAT", "TANH", "MUL", "SUB", "TANH", "MUL", "ADD"]),
    # already in master (ggml_cuda_match_moe_weighted_reduction): the weighted sum of the expert outputs
    ("MOE_SUM*",   ["MUL"] + ["ADD"]*7),
]


def parse(path):
    graphs, cur = [], None
    for line in open(path, errors="replace"):
        if line.startswith("## SPLIT #0"):
            cur = []
            graphs.append(cur)
            continue
        m = NODE.match(line)
        if m and cur is not None:
            cur.append((int(m.group(1)), m.group(2), m.group(3).strip(), m.group(4)))
    return graphs


def visit_kind(slot):
    phys = slot if slot < 24 else 6 + (slot - 24) % 12
    return ("full" if phys % 6 == 5 else "swa") + ("+loop" if slot >= 24 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--fused", action="store_true")
    args = ap.parse_args()

    graphs = parse(args.log)
    if not graphs:
        sys.exit("no graph dump found: run with GGML_SCHED_DEBUG=2 and -v")
    nodes = graphs[-1]  # the last graph is a one-token decode
    print(f"{len(graphs)} graph dumps, using the last: {nodes[-1][0] + 1} nodes, {len(nodes)} of them not views")

    # the empty recurrent-state nodes and the dense-bias chain of the CPU graph, by tensor name
    empty = []
    ops, dense = [], []
    skip = False
    for idx, op, name, size in nodes:
        if name.startswith("cache_r_l") and size == "0K" and op in ("FILL", "CPY"):
            empty.append(op)
            continue
        if name.startswith("(permuted) (cont)") or name.startswith("kq_b-") or name.startswith("attn_inp_kq_mask (co") \
                or name.startswith("kq_mask_rel_pos-"):
            dense.append(op)
            skip = name.startswith("(permuted) (cont)") or skip
            continue
        if skip and op in ("GET_ROWS", "ADD"):  # the gather after the table CONT, the ADD of bias and mask
            dense.append(op)
            if op == "ADD":
                skip = False
            continue
        ops.append((idx, op, name))

    print(f"empty recurrent-state nodes (no-ops): {len(empty)}")
    print(f"dense-bias nodes, CPU graph only:     {len(dense)}  {dict(collections.Counter(dense))}")
    print(f"dispatched ops on the GPU graph:      {len(ops)}")

    hist = collections.Counter(op for _, op, _ in ops)
    print("\nops per token:")
    for op, n in hist.most_common():
        print(f"  {op:12s} {n:6d}")

    # per layer visit: a visit ends at its l_out
    per_kind = collections.defaultdict(list)
    start, first = 0, None
    for i, (_, op, name) in enumerate(ops):
        m = re.match(r"l_out-(\d+)$", name)
        if m:
            slot = int(m.group(1))
            if first is None:
                first = i + 1  # the input block (embedding, norm) is counted with visit 0
            per_kind[visit_kind(slot)].append(i + 1 - start)
            start = i + 1
    tail = len(ops) - start
    print("\ndispatched ops per layer visit (the visit of slot 0 includes the input block, cross-pass message mixes are"
          "\ncounted with the first looped visit of their pass):")
    total = 0
    for kind in ("swa", "full", "swa+loop", "full+loop"):
        v = per_kind[kind]
        total += sum(v)
        print(f"  {kind:10s} x{len(v):2d}: min {min(v)}, median {sorted(v)[len(v)//2]}, max {max(v)}, sum {sum(v)}")
    print(f"  output block: {tail}")
    print(f"  total: {total + tail}")

    if args.fused:
        seq = [op for _, op, _ in ops]
        i, launches, found = 0, 0, collections.Counter()
        replaced = collections.Counter()
        rest = collections.Counter()
        while i < len(seq):
            for name, chain in CHAINS:
                if seq[i:i + len(chain)] == chain:
                    found[name] += 1
                    replaced[name] += len(chain)
                    i += len(chain)
                    break
            else:
                rest[seq[i]] += 1
                i += 1
            launches += 1
        print("\nchains by op pattern (one kernel each; * = already fused in master):")
        for name, n in found.most_common():
            print(f"  {name:11s} x{n:4d}  replaces {replaced[name]:5d} ops")
        new = sum(n for k, n in found.items() if not k.endswith("*"))
        new_replaced = sum(n for k, n in replaced.items() if not k.endswith("*"))
        base = len(seq) - (replaced["MOE_SUM*"] - found["MOE_SUM*"])
        print("ops left outside the chains: " + ", ".join(f"{op} {n}" for op, n in rest.most_common()))
        print(f"dispatches per token: {base} with master's fusions, {launches} with the Inkling 2 chains"
              f" ({new} chains replace {new_replaced} ops)")


if __name__ == "__main__":
    main()
