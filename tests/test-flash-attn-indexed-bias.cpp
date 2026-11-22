#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static void set_kv(ggml_tensor * t, const std::vector<float> & data) {
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, data.data(), bytes.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

static void test_indexed_bias(ggml_backend_t backend, int width, bool use_rows, int n_query, int cells = 768, ggml_type kv_type = GGML_TYPE_F16) {
    constexpr int heads = 6, kv_heads = 2, streams = 2, bins = 17;
    const int cols = use_rows ? std::min(512, cells) : cells;

    ggml_context * ctx = ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, n_query, heads, streams);
    auto * k = ggml_new_tensor_4d(ctx, kv_type, width, cells, kv_heads, use_rows ? 1 : streams);
    auto * v = ggml_dup_tensor(ctx, k);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cols, n_query, 1, streams);
    auto * rows = use_rows ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cols, streams) : nullptr;
    auto * table = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, bins, heads, n_query*streams);
    auto * indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cols, n_query*streams);
    auto * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    auto * out = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, rows, 1.0f/sqrtf(width), 0, 0);
    ggml_flash_attn_ext_add_sinks(out, sinks);
    const bool supports_plain = ggml_backend_supports_op(backend, out);
    ggml_flash_attn_ext_add_indexed_bias(out, table, indices);
    GGML_ASSERT(ggml_flash_attn_ext_has_indexed_bias(out));
    ggml_flash_attn_ext_add_indexed_bias(out, nullptr, nullptr);
    GGML_ASSERT(!ggml_flash_attn_ext_has_indexed_bias(out) && !out->src[6] && !out->src[7]);
    GGML_ASSERT(ggml_backend_supports_op(backend, out) == supports_plain);
    ggml_flash_attn_ext_add_indexed_bias(out, table, indices);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return;
    }

    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);
    ggml_backend_buffer_clear(buf, 0);

    std::vector<int32_t> row(cols*streams), idx(ggml_nelements(indices));
    std::vector<ggml_fp16_t> visibility(idx.size());
    std::vector<float> keys(ggml_nelements(k)), values(ggml_nelements(v));
    std::vector<float> queries(ggml_nelements(q));
    float sink_logits[heads];
    for (int h = 0; h < heads; ++h) {
        sink_logits[h] = float(2*h - 3);
    }
    for (size_t i = 0; i < queries.size(); i += width) {
        queries[i] = 4.0f;
    }
    std::vector<float> biases(ggml_nelements(table)), expected(n_query*streams*heads), output(ggml_nelements(out));
    for (int s = 0; s < v->ne[3]; ++s) {
        for (int h = 0; h < kv_heads; ++h) {
            for (int c = 0; c < cells; ++c) {
                const float x = float((c + 3*h + 5*s) % 23 - 11)/16;
                const int offset = ((s*kv_heads + h)*cells + c)*width;
                std::fill_n(values.data() + offset, width, x);
                keys[offset] = float((7*c + 2*h + 3*s) % 29 - 14)/4;
            }
        }
    }
    for (int s = 0; s < streams; ++s) {
        for (int c = 0; c < cols; ++c) {
            row[s*cols + c] = use_rows ? (c % 19 == 0 ? -1 : (13*c + 73*s) % cells) : c;
        }
        for (int j = 0; j < n_query; ++j) {
            const int query = s*n_query + j;
            for (int h = 0; h < heads; ++h) {
                for (int b = 0; b < bins; ++b) {
                    biases[(query*heads + h)*bins + b] = float((b*(3 + h) + query) % 13 - 6)/2.9f;
                }
                double sum = exp(double(sink_logits[h])), weighted = 0.0;
                for (int c = 0; c < cols; ++c) {
                    const int cell = row[s*cols + c];
                    const int i = query*cols + c;
                    idx[i] = (c + 3*query) % bins;
                    const bool visible = (s != 1 || j != 0) && cell >= 0 && (c + j) % 7 != 0;
                    visibility[i] = ggml_fp32_to_fp16(visible ? 0.25f : -INFINITY);
                    if (visible) {
                        const int vk = (use_rows ? 0 : s)*kv_heads + h/(heads/kv_heads);
                        const double qk = 4.0*keys[(vk*cells + cell)*width]/sqrt(width);
                        const float m = 0.25f + biases[(query*heads + h)*bins + idx[i]];
                        const double weight = exp(qk + ggml_fp16_to_fp32(ggml_fp32_to_fp16(m)));
                        weighted += weight*values[(vk*cells + cell)*width];
                        sum += weight;
                    }
                }
                expected[query*heads + h] = weighted/sum;
            }
        }
    }
    ggml_backend_tensor_set(q, queries.data(), 0, ggml_nbytes(q));
    set_kv(k, keys);
    set_kv(v, values);
    ggml_backend_tensor_set(mask, visibility.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(table, biases.data(), 0, ggml_nbytes(table));
    ggml_backend_tensor_set(sinks, sink_logits, 0, sizeof(sink_logits));
    ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
    if (rows) {
        ggml_backend_tensor_set(rows, row.data(), 0, ggml_nbytes(rows));
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
    float max_error = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        GGML_ASSERT(std::isfinite(output[i]));
        max_error = std::max(max_error, fabsf(output[i] - expected[i/width]));
    }
    printf("%s: width=%d, kv=%s, rows=%d, queries=%d, cells=%d, max error=%g\n", ggml_backend_name(backend), width, ggml_type_name(kv_type), use_rows, n_query, cells, max_error);
    GGML_ASSERT(max_error < 0.001f);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// The indexed bias against the CPU backend, which has no indexed bias: the CPU computes the same attention from the
// dense per-head F16 mask that the indexed bias stands for. Random Q and K; V rows are one of `bins` prototypes per
// K/V head and the bias of each query head has a spike on one bin, so the heads give outputs of O(1) that differ from
// each other, and a kernel that mixes up heads, K/V heads or rows fails by far more than the tolerance.
// Covers the decode shape of Inkling 2 (24 query heads on 8 K/V heads, one query, deep K/V).
static void test_indexed_bias_vs_cpu(ggml_backend_t backend, ggml_backend_t cpu, int width, int heads, int kv_heads, int streams,
        int n_query, int cells, bool use_rows, ggml_type kv_type) {
    constexpr int bins = 17;
    const int cols = cells; // one mask column per cell, in a shuffled order with holes when read through rows

    uint32_t seed = 1006u + 31u*heads + 7u*cells + 3u*n_query + use_rows;
    auto rnd = [&seed]() { // uniform in [-1, 1)
        seed = seed*1664525u + 1013904223u;
        return float(seed >> 8)/float(1 << 23) - 1.0f;
    };

    ggml_context * ctx = ggml_init({64*ggml_tensor_overhead() + 2*ggml_graph_overhead(), nullptr, true});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, n_query, heads, streams);
    auto * k = ggml_new_tensor_4d(ctx, kv_type, width, cells, kv_heads, use_rows ? 1 : streams);
    auto * v = ggml_dup_tensor(ctx, k);
    auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cols, n_query, 1, streams);
    auto * rows = use_rows ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cols, streams) : nullptr;
    auto * table = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, bins, heads, n_query*streams);
    auto * indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cols, n_query*streams);
    auto * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    auto * out = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, rows, 1.0f/sqrtf(width), 0, 0);
    ggml_flash_attn_ext_add_sinks(out, sinks);
    ggml_flash_attn_ext_add_indexed_bias(out, table, indices);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    const bool supported = ggml_backend_supports_op(backend, out);

    // the reference: K and V gathered through the rows, a dense mask per head
    ggml_context * ctx_ref = ggml_init({64*ggml_tensor_overhead() + 2*ggml_graph_overhead(), nullptr, true});
    auto * q_ref = ggml_dup_tensor(ctx_ref, q);
    auto * k_ref = ggml_new_tensor_4d(ctx_ref, kv_type, width, cols, kv_heads, streams);
    auto * v_ref = ggml_dup_tensor(ctx_ref, k_ref);
    auto * mask_ref = ggml_new_tensor_4d(ctx_ref, GGML_TYPE_F16, cols, n_query, heads, streams);
    auto * sinks_ref = ggml_dup_tensor(ctx_ref, sinks);
    auto * out_ref = ggml_flash_attn_ext(ctx_ref, q_ref, k_ref, v_ref, mask_ref, 1.0f/sqrtf(width), 0, 0);
    ggml_flash_attn_ext_add_sinks(out_ref, sinks_ref);
    ggml_prec_set_acc(out_ref, GGML_PREC_F32);
    GGML_ASSERT(ggml_backend_supports_op(cpu, out_ref));

    std::vector<float> queries(ggml_nelements(q)), keys(ggml_nelements(k)), values(ggml_nelements(v));
    std::vector<float> biases(ggml_nelements(table)), sink_logits(heads), protos(size_t(kv_heads)*bins*width);
    std::vector<int32_t> row(size_t(cols)*streams), idx(ggml_nelements(indices));
    std::vector<ggml_fp16_t> visibility(idx.size());

    for (float & x : queries) { x = 2.0f*rnd(); }
    for (float & x : keys)    { x = rnd(); }
    for (float & x : protos)  { x = rnd() < 0.0f ? -1.0f : 1.0f; }
    for (float & x : sink_logits) { x = 3.0f*rnd() + 1.0f; }
    for (int s = 0; s < int(v->ne[3]); ++s) {
        for (int h = 0; h < kv_heads; ++h) {
            for (int c = 0; c < cells; ++c) {
                for (int i = 0; i < width; ++i) {
                    values[((size_t(s)*kv_heads + h)*cells + c)*width + i] = protos[(size_t(h)*bins + (c + s) % bins)*width + i] + 0.125f*rnd();
                }
            }
        }
    }
    for (int s = 0; s < streams; ++s) {
        for (int c = 0; c < cols; ++c) {
            // rows: a permutation of the cells (7919 is coprime to the cell counts used) with a hole every 19 columns
            row[size_t(s)*cols + c] = use_rows ? (c % 19 == 5 ? -1 : int((7919ll*c + 73*s) % cells)) : c;
        }
        for (int j = 0; j < n_query; ++j) {
            const int query = s*n_query + j;
            for (int h = 0; h < heads; ++h) {
                for (int b = 0; b < bins; ++b) {
                    biases[(size_t(query)*heads + h)*bins + b] = 1.5f*rnd() + (b == (5*h + query) % bins ? 6.0f : 0.0f);
                }
            }
            for (int c = 0; c < cols; ++c) {
                const int cell = row[size_t(s)*cols + c];
                const size_t i = size_t(query)*cols + c;
                idx[i] = cell < 0 ? 0 : (cell + 3*query) % bins;
                const bool visible = cell >= 0 && (c + 3*j) % 11 != 0;
                visibility[i] = ggml_fp32_to_fp16(visible ? 0.25f : -INFINITY);
            }
        }
    }

    // the reference inputs
    std::vector<float> keys_ref(ggml_nelements(k_ref)), values_ref(ggml_nelements(v_ref));
    std::vector<ggml_fp16_t> dense(ggml_nelements(mask_ref));
    for (int s = 0; s < streams; ++s) {
        const int s_kv = use_rows ? 0 : s;
        for (int h = 0; h < kv_heads; ++h) {
            for (int c = 0; c < cols; ++c) {
                const int cell = std::max(row[size_t(s)*cols + c], 0);
                const size_t src = ((size_t(s_kv)*kv_heads + h)*cells + cell)*width;
                const size_t dst = ((size_t(s)*kv_heads + h)*cols + c)*width;
                std::copy_n(keys.data() + src, width, keys_ref.data() + dst);
                std::copy_n(values.data() + src, width, values_ref.data() + dst);
            }
        }
        for (int h = 0; h < heads; ++h) {
            for (int j = 0; j < n_query; ++j) {
                const int query = s*n_query + j;
                for (int c = 0; c < cols; ++c) {
                    const size_t i = size_t(query)*cols + c;
                    const float m = ggml_fp16_to_fp32(visibility[i]) + biases[(size_t(query)*heads + h)*bins + idx[i]];
                    dense[((size_t(s)*heads + h)*n_query + j)*cols + c] = ggml_fp32_to_fp16(m);
                }
            }
        }
    }

    auto * graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, out_ref);
    auto * buf_ref = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu);
    GGML_ASSERT(buf_ref);
    ggml_backend_tensor_set(q_ref, queries.data(), 0, ggml_nbytes(q_ref));
    set_kv(k_ref, keys_ref);
    set_kv(v_ref, values_ref);
    ggml_backend_tensor_set(mask_ref, dense.data(), 0, ggml_nbytes(mask_ref));
    ggml_backend_tensor_set(sinks_ref, sink_logits.data(), 0, ggml_nbytes(sinks_ref));
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph_ref) == GGML_STATUS_SUCCESS);
    std::vector<float> reference(ggml_nelements(out_ref));
    ggml_backend_tensor_get(out_ref, reference.data(), 0, ggml_nbytes(out_ref));

    // the reference must tell the heads apart, or the comparison below proves nothing: out is [width, heads, n_query, streams]
    float min_head_distance = INFINITY, max_reference = 0.0f;
    for (int h = 1; h < heads; ++h) {
        float distance = 0.0f;
        for (int i = 0; i < width; ++i) {
            distance = std::max(distance, fabsf(reference[size_t(h)*width + i] - reference[size_t(h - 1)*width + i]));
        }
        min_head_distance = std::min(min_head_distance, distance);
    }
    for (float x : reference) {
        GGML_ASSERT(std::isfinite(x));
        max_reference = std::max(max_reference, fabsf(x));
    }
    GGML_ASSERT(min_head_distance > 0.25f && max_reference > 0.5f);

    if (!supported) {
        printf("%s: no indexed bias, CPU reference only: width=%d, heads=%d/%d, streams=%d, kv=%s, rows=%d, queries=%d, cells=%d, min head distance=%g\n",
                ggml_backend_name(backend), width, heads, kv_heads, streams, ggml_type_name(kv_type), use_rows, n_query, cells, min_head_distance);
    } else {
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        auto * buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buf);
        ggml_backend_buffer_clear(buf, 0);
        ggml_backend_tensor_set(q, queries.data(), 0, ggml_nbytes(q));
        set_kv(k, keys);
        set_kv(v, values);
        ggml_backend_tensor_set(mask, visibility.data(), 0, ggml_nbytes(mask));
        ggml_backend_tensor_set(table, biases.data(), 0, ggml_nbytes(table));
        ggml_backend_tensor_set(sinks, sink_logits.data(), 0, ggml_nbytes(sinks));
        ggml_backend_tensor_set(indices, idx.data(), 0, ggml_nbytes(indices));
        if (rows) {
            ggml_backend_tensor_set(rows, row.data(), 0, ggml_nbytes(rows));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> output(ggml_nelements(out));
        ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
        GGML_ASSERT(output.size() == reference.size());
        float max_error = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            GGML_ASSERT(std::isfinite(output[i]));
            max_error = std::max(max_error, fabsf(output[i] - reference[i]));
        }
        printf("%s vs CPU: width=%d, heads=%d/%d, streams=%d, kv=%s, rows=%d, queries=%d, cells=%d, max error=%g, min head distance=%g\n",
                ggml_backend_name(backend), width, heads, kv_heads, streams, ggml_type_name(kv_type), use_rows, n_query, cells, max_error, min_head_distance);
        // the kernels accumulate V in F16, the outputs are O(1)
        GGML_ASSERT(max_error < 0.02f);
        ggml_backend_buffer_free(buf);
    }

    ggml_backend_buffer_free(buf_ref);
    ggml_free(ctx_ref);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend);
        // against the CPU backend: the shapes of the existing cases, then the decode shape of Inkling 2 (24 query heads on
        // 8 K/V heads, one query), short and deep, with and without rows, in the K/V types it is run with
        test_indexed_bias_vs_cpu(backend, cpu, 128,  6, 2, 2, 1,   768, false, GGML_TYPE_F16);
        test_indexed_bias_vs_cpu(backend, cpu, 128,  6, 2, 2, 7,   768, true,  GGML_TYPE_F16);
        test_indexed_bias_vs_cpu(backend, cpu, 128, 24, 8, 2, 1,  1024, false, GGML_TYPE_F16);
        test_indexed_bias_vs_cpu(backend, cpu, 128, 24, 8, 2, 1,  1024, true,  GGML_TYPE_F16);
        test_indexed_bias_vs_cpu(backend, cpu, 128, 24, 8, 1, 1, 32768, false, GGML_TYPE_F16);
        test_indexed_bias_vs_cpu(backend, cpu, 128, 24, 8, 1, 1, 32768, true,  GGML_TYPE_F16);
        test_indexed_bias_vs_cpu(backend, cpu, 128, 24, 8, 1, 1,  4096, false, GGML_TYPE_Q8_0);
        test_indexed_bias_vs_cpu(backend, cpu, 128, 24, 8, 1, 1,  4096, true,  GGML_TYPE_Q8_0);
        test_indexed_bias_vs_cpu(backend, cpu, 128, 24, 8, 1, 5,  1536, false, GGML_TYPE_F16); // prefill: the tile kernel
        test_indexed_bias_vs_cpu(backend, cpu,  64, 24, 8, 1, 1,  1024, false, GGML_TYPE_F16); // no GQA variant at this width
        for (int width : {64, 128}) {
            test_indexed_bias(backend, width, false, 7, 16);
            test_indexed_bias(backend, width, true, 7, 16);
            test_indexed_bias(backend, width, false, 1, 768, GGML_TYPE_Q4_0);
            test_indexed_bias(backend, width, true,  1, 768, GGML_TYPE_Q4_0);
            for (int n_query : {1, 7, 65}) {
                test_indexed_bias(backend, width, false, n_query);
                test_indexed_bias(backend, width, true, n_query);
            }
        }
        ggml_backend_free(backend);
    }
    ggml_backend_free(cpu);
}
