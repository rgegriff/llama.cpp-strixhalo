#!/usr/bin/env python3
"""Write a random-weight Inkling 2 Mini with the true layer shapes and only N experts (default 8 = the number routed).

Why: the real GGUF is 27.8 GB and its slot on the GPU is booked. A decode step of this file runs the same graph (same
node list, same tensor shapes, 48 layer visits) and reads the same bytes of weights per token as the real model, because
the real model also reads exactly 8 experts per visit. It is ~3.4 GB, so it fits next to the house models and the CPU
backend can run it. Use it for: the op census, CPU-vs-GPU logit comparisons of kernel patches, and launch-overhead
benchmarks. Do NOT use it for anything about quality: the weights are noise.

All metadata (tokenizer included) is copied from the real file; only expert_count changes.

  python3 kernel-research/make_synth_gguf.py \
      ~/models/inkling-2-mini-gguf/Inkling-2-Mini-Q8_0.gguf kernel-research/synth/inkling2-mini-synth-e8-Q8_0.gguf
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf  # noqa: E402

EXPERT_AXIS = {  # tensor suffix -> index of the expert axis in the ggml shape
    "ffn_gate_inp.weight": 1,
    "exp_probs_b.bias": 0,
    "ffn_gate_up_exps.weight": 2,
    "ffn_down_exps.weight": 2,
}


def q8_0(rng: np.random.Generator, shape: list[int]) -> np.ndarray:
    """Random Q8_0 blocks (f16 scale + 32 int8) for a tensor of ggml shape `shape`, weights ~ U(-1, 1)*sqrt(3/fan_in)."""
    n = int(np.prod(shape))
    assert shape[0] % 32 == 0, shape
    blocks = np.empty((n // 32, 34), dtype=np.uint8)
    d = np.float16(np.sqrt(3.0 / shape[0]) / 127.0)
    blocks[:, 0:2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
    blocks[:, 2:] = np.frombuffer(rng.bytes(n), dtype=np.uint8).reshape(-1, 32)
    blocks[:, 2:][blocks[:, 2:] == 0x80] = 0  # -128 is not a Q8_0 code that the quantizer writes
    return blocks.reshape(-1)


def f32(rng: np.random.Generator, name: str, shape: list[int]) -> np.ndarray:
    n = int(np.prod(shape))
    if re.search(r"norm\.weight$|_clip\.weight$", name):
        out = 1.0 + 0.05 * rng.standard_normal(n)
    elif name.endswith("exp_probs_b.bias"):
        out = 0.01 * rng.standard_normal(n)
    elif re.search(r"_conv\.weight$", name):
        out = 0.5 * rng.standard_normal(n)
    elif name.endswith("attn_rel_bias.weight"):
        out = rng.standard_normal(n)
    else:  # queries, router, loop controllers
        out = rng.standard_normal(n) / np.sqrt(shape[0])
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="the real Inkling 2 Mini GGUF (only its header is read)")
    ap.add_argument("dst", type=Path)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1006)
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    n_used = reader.fields[f"{arch}.expert_used_count"].contents()
    assert args.experts >= n_used, f"need at least the {n_used} routed experts"

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(args.dst, arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == f"{arch}.expert_count":
            val = args.experts
        elif field.name == gguf.Keys.General.NAME:
            val = f"{val} (synthetic, random weights, {args.experts} experts)"
        writer.add_key_value(field.name, val, vtype, sub_type=sub)

    rng = np.random.default_rng(args.seed)
    total = 0
    for t in reader.tensors:
        shape = [int(x) for x in t.shape]
        suffix = t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name
        if suffix in EXPERT_AXIS:
            shape[EXPERT_AXIS[suffix]] = args.experts
        if t.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            data = q8_0(rng, shape)
        elif t.tensor_type == gguf.GGMLQuantizationType.F32:
            data = f32(rng, t.name, shape)
        else:
            raise SystemExit(f"{t.name}: unexpected type {t.tensor_type.name}")
        raw_shape = list(reversed(shape))
        if t.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            raw_shape[-1] = raw_shape[-1] // 32 * 34  # GGUFWriter takes the byte shape of quantized uint8 data
        writer.add_tensor(t.name, data, raw_shape=raw_shape, raw_dtype=t.tensor_type)
        total += data.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.dst}: {len(reader.tensors)} tensors, {total / 1e9:.2f} GB, {args.experts} experts")


if __name__ == "__main__":
    main()
