#!/usr/bin/env python3
"""Best time per (type, n_kv, n_q) from a run_sweep.sh log: python3 table.py results/sweep-*.log"""
import re, sys
from collections import defaultdict

best = {}
hdr = re.compile(r"== (\w+) n_kv (\d+) n_q (\d+)")
row = re.compile(r"(valu|wmma|stage)\s+chunk\s+(\d+) hpb (\d) \(\s*(\d+) blocks x\s+(\d+) waves\):\s+([\d.]+) us,\s+([\d.]+) GB/s K\+V, max err ([\d.e+-]+) \(rel ([\d.e+-]+)")
key = None
for path in sys.argv[1:]:
    for line in open(path):
        m = hdr.search(line)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)))
            continue
        m = row.search(line)
        if m and key:
            us = float(m.group(6))
            cur = best.get(key)
            if cur is None or us < cur[0]:
                best[key] = (us, m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(7)), float(m.group(9)))

types = sorted({k[0] for k in best}, reverse=True)
nkvs = sorted({k[1] for k in best})
nqs = sorted({k[2] for k in best})
for t in types:
    print(f"\n{t}: best time in us (kernel, chunk, hpb; GB/s of K+V; max rel err vs double reference)")
    print("n_q   " + "".join(f"{'n_kv ' + str(n):>40}" for n in nkvs))
    for q in nqs:
        cells = []
        for n in nkvs:
            b = best.get((t, n, q))
            cells.append(f"{b[0]:8.0f} ({b[1]} {b[2]} {b[3]}; {b[4]:5.1f} GB/s; {b[5]:.1e})" if b else " " * 40)
        print(f"{q:<4}  " + "".join(f"{c:>40}" for c in cells))
    print("ratio to n_q = 1:")
    for q in nqs:
        print(f"{q:<4}  " + "".join(f"{(best[(t, n, q)][0]/best[(t, n, 1)][0] if (t, n, q) in best and (t, n, 1) in best else float('nan')):>40.2f}" for n in nkvs))
