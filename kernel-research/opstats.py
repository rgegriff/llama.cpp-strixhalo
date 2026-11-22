#!/usr/bin/env python3
"""Summarize the ggml_cuda_op_stats lines of a GGML_CUDA_OP_STATS=1|2 run (see ggml-cuda.cu).

  GGML_CUDA_OP_STATS=2 build-rocm/bin/llama-bench ... 2> run.log ; python3 kernel-research/opstats.py run.log

Level 1 gives dispatches per graph evaluation. Level 2 adds the synchronized wall time per kind of op, from which this
prints: the split of a token into weight-reading kernels, attention and glue; the cost of one glue dispatch (the `c`
of REPORT.md); and the bytes per second that each weight kernel reaches (the `B` of REPORT.md).
"""
import re
import sys

LINE = re.compile(r"ggml_cuda_op_stats:\s+(op|fused)\s+(.+?)\s+(\d+\.\d)(?:\s+(\d+\.\d+) ms\s+(\d+\.\d+) us each)?(?:\s+\(replaces (\d+\.\d) ops\))?\s*$")
SHAPE = re.compile(r"\[(\w+) ([\dx]+)\]")
BYTES = {"q8_0": 34 / 32, "f32": 4.0, "f16": 2.0, "q4_0": 18 / 32, "iq4_nl": 18 / 32, "bf16": 2.0}
N_EXPERT_USED = 8


def weight_bytes(name):
    """Bytes of the first operand that one dispatch reads, if it is a weight matrix worth counting."""
    m = SHAPE.search(name)
    if not m or m.group(1) not in BYTES:
        return 0
    dims = [int(x) for x in m.group(2).split("x")]
    if name.startswith("MUL_MAT_ID"):
        dims = dims[:2] + [N_EXPERT_USED]
    n = 1
    for d in dims:
        n *= d
    return n * BYTES[m.group(1)]


def main():
    rows, head = [], []
    for line in open(sys.argv[1], errors="replace"):
        if "ggml_cuda_op_stats:" not in line:
            continue
        m = LINE.search(line)
        if not m:
            head.append(line.strip())
            continue
        kind, name, count, ms, us, replaced = m.groups()
        rows.append((kind, name.strip(), float(count), float(ms) if ms else None, float(us) if us else None))
    if not rows:
        sys.exit("no ggml_cuda_op_stats lines: run with GGML_CUDA_OP_STATS=1 or 2 and keep stderr")
    print("\n".join(head))
    timing = rows[0][3] is not None

    groups = {"weights": [0.0, 0.0, 0.0], "attention": [0.0, 0.0, 0.0], "glue": [0.0, 0.0, 0.0]}
    kernels = []
    for kind, name, count, ms, us in rows:
        b = weight_bytes(name) if kind == "op" and name.startswith("MUL_MAT") else 0
        g = "attention" if name.startswith("FLASH_ATTN") else "weights" if b >= 1e6 else "glue"
        groups[g][0] += count
        groups[g][1] += ms or 0.0
        groups[g][2] += b * count
        if b >= 1e6 and timing:
            kernels.append((ms, name, count, us, b))

    print(f"\n{'group':10s} {'dispatches':>10s} {'ms/token':>9s} {'MB/token':>9s}")
    for g, (n, ms, b) in groups.items():
        print(f"{g:10s} {n:10.1f} {ms:9.3f} {b / 1e6:9.1f}")
    if timing:
        n, ms, _ = groups["glue"]
        print(f"\nglue: {ms * 1000 / n:.2f} us per dispatch (synchronized, an upper bound of c)")
        n, ms, b = groups["weights"]
        print(f"weights: {b / 1e9:.2f} GB per token in {ms:.2f} ms = {b / 1e9 / (ms / 1e3):.0f} GB/s (B)")
        print("\nweight kernels, slowest first:")
        for ms, name, count, us, b in sorted(kernels, reverse=True):
            print(f"  {name:58s} x{count:6.1f} {ms:8.3f} ms  {us:8.1f} us each  {b / 1e6 / (us / 1e6) / 1e3:6.0f} GB/s")
        print("\nglue, slowest first:")
        glue = [(ms, name, count, us) for kind, name, count, ms, us in rows
                if not name.startswith("FLASH_ATTN") and not (kind == "op" and weight_bytes(name) >= 1e6)]
        for ms, name, count, us in sorted(glue, reverse=True)[:25]:
            print(f"  {name:58s} x{count:6.1f} {ms:8.3f} ms  {us:8.1f} us each")


if __name__ == "__main__":
    main()
