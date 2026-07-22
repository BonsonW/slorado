/* @file modcall.cpp
**
** runs base modification calling steps
** @@
******************************************************************************/

#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdlib.h>
#include <vector>

#include "torchbox.h"
#include "modcall.h"
#include "misc.h"
#include "error.h"

#include "dorado/modbase_model.h"

#include "dorado/modbase.h"
#include "dorado/tensor_chunk_utils.h"

#ifdef USE_GPU
#include <c10/core/DeviceGuard.h>
#endif

typedef struct {
    core_t* core;
    db_t* db;
    int32_t runner;
    int32_t start;
    int32_t end;
} model_thread_arg_t;

static void mod_accept_chunk(const int num_chunks, const torch::Tensor& signal, const std::vector<int8_t>& kmers, const core_t* core, const int runner_idx) {
    runner_t* runner = (*core->mod_runners)[runner_idx];

    auto& input_sigs = runner->input_sigs;
    auto& input_seqs = runner->input_seqs;

    // auto& input_sigs = m_input_sigs[model_id];
    // auto& input_seqs = m_input_seqs[model_id];
    if (signal.size(0) != input_sigs.size(2)) {
        ERROR("%s", "ModBaseRunner received signal and sequence chunks with different lengths.");
        exit(1);
    }
    input_sigs = input_sigs.contiguous(); // todo: move these somewhere else
    input_seqs = input_seqs.contiguous();

    const auto sig_len = signal.size(0);
    copy_tensor_elems(input_sigs, num_chunks * sig_len, signal, 0, sig_len);

    const auto kmer_elem_count = input_seqs.size(1) * input_seqs.size(2);
    assert(input_seqs.dtype() == torch::kInt8);
    int8_t* const input_seqs_ptr = input_seqs.data_ptr<int8_t>();
    std::memcpy(&input_seqs_ptr[num_chunks * kmer_elem_count], kmers.data(),  kmer_elem_count * sizeof(int8_t));
}

static void mod_call_chunks(
    const core_t* core,
    const std::vector<mod_chunk_t *> &chunks,
    const int runner_idx
) {
    runner_t* runner = (*core->mod_runners)[runner_idx];

#ifdef USE_GPU
    c10::DeviceGuard device_guard(runner->tensor_opts.device());
#endif
    torch::InferenceMode guard;

    LOG_DEBUG("%s", "mod calling chunks");
    const int64_t active_chunks = static_cast<int64_t>(chunks.size());
    auto active_input_sigs = runner->input_sigs.narrow(0, 0, active_chunks);
    auto active_input_seqs = runner->input_seqs.narrow(0, 0, active_chunks);

    auto scores = modbase_model_forward(
        (modbase_model_t *)runner->bc_model,
        active_input_sigs.to(runner->tensor_opts.device_opt().value()),
        active_input_seqs.to(runner->tensor_opts.device_opt().value())
    );
#ifdef USE_GPU
    if (runner->device != "cpu") torch::cuda::synchronize(runner->device_idx);
#endif

    auto scores_f16 = scores.cpu().contiguous();
    assert(scores_f16.is_contiguous());
    assert(scores_f16.dtype() == at::ScalarType::Half);

    const int64_t row_size = scores_f16.size(1);
    const auto* const scores_f16_ptr = scores_f16.data_ptr<c10::Half>();
    for (size_t i = 0; i < chunks.size(); ++i) {
        extract_mod_probs(core, chunks[i], scores_f16_ptr, static_cast<int64_t>(i) * row_size, row_size);
    }
}

void mod_basecall_chunks(
    const core_t* core,
    const int runner_idx,
    const std::vector<mod_chunk_t *> &results
) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    const auto chunk_size = core->modbase_config->context.chunk_size;
    const int kmer_size_per_sample = core->modbase_config->context.kmer_len * NUM_BASES;
    auto sequence_stride_ratio = general_stride_ratio(core->modbase_config->general);

    assert(is_chunked_input_model(*core->modbase_config));

    ts->time_modcall -= realtime();
    for (size_t i = 0; i < results.size(); ++i) {
        mod_chunk_t *chunk = results[i];
        read_dat_t *read_dat = chunk->read_dat;

        const int64_t start = chunk->signal_offset;
        const int64_t end = std::min(start + chunk_size, static_cast<int64_t>(read_dat->scaled_signal.size(0)));
        const int64_t len = end - start;
        assert(start <= end);

        auto signal_chunk = read_dat->scaled_signal.index({at::indexing::Slice(start, end)});

        const std::int64_t kmer_start = start / sequence_stride_ratio;
        const std::int64_t kmer_end = end / sequence_stride_ratio;

        auto encoded_kmers_chunk = std::vector<int8_t>(
            read_dat->encoded_kmers.begin() + kmer_start * kmer_size_per_sample,
            read_dat->encoded_kmers.begin() + kmer_end * kmer_size_per_sample
        );

        if (len < chunk_size) {
            // tile the signal tensor
            auto result = std::div(chunk_size, len);
            int n_tiles = result.quot;
            int n_overhang = result.rem;
            signal_chunk = at::concat({signal_chunk.repeat({n_tiles}),
                                       signal_chunk.index({at::indexing::Slice(0, n_overhang)})},
                                      -1);
            // tile the kmer vector (sequence resolution = chunk_size / sequence_stride_ratio)
            const int64_t original_size = static_cast<int64_t>(encoded_kmers_chunk.size());
            const int64_t extended_size = (chunk_size / sequence_stride_ratio) * kmer_size_per_sample;
            encoded_kmers_chunk.resize(extended_size);

            for (int64_t i = original_size; i < extended_size; ++i) {
                encoded_kmers_chunk[i] = encoded_kmers_chunk[i % original_size];
            }
        }

        mod_accept_chunk(i, signal_chunk, encoded_kmers_chunk, core, runner_idx);
    }

    mod_call_chunks(core, results, runner_idx);
    ts->time_modcall += realtime();
}

static void* pthread_single_mod_basecall(void* voidargs) {
    model_thread_arg_t* args = (model_thread_arg_t*)voidargs;
    db_t* db = args->db;
    core_t* core = args->core;
    const size_t runner_idx = args->runner;
    const size_t start = args->start;
    const size_t end = args->end;
    opt_t opt = core->opt;

    std::vector<mod_chunk_t *> results;

    LOG_DEBUG("%s", "loading chunks");

    for (size_t read_idx = start; read_idx < end; ++read_idx) {
        auto& chunks = (*db->mod_chunks)[read_idx];

        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            results.push_back(&chunks[chunk_idx]);

            if (results.size() == (size_t)opt.mod_gpu_batch_size) {
                mod_basecall_chunks(core, runner_idx, results);
                results.clear();
            }
        }
    }

    // leftover chunks
    if (results.size() > 0) {
        mod_basecall_chunks(core, runner_idx, results);
    }

    pthread_exit(0);
}

void mod_basecall_db(core_t* core, db_t* db) {
    int32_t n_reads = db->n_rec;
    int32_t num_threads = (*core->mod_runners).size();
    int32_t step = (n_reads + num_threads - 1) / num_threads;

    pthread_t tids[num_threads];
    model_thread_arg_t pt_args[num_threads];
    int32_t t, ret;
    int32_t i = 0;
    // set the data structures
    for (t = 0; t < num_threads; t++) {
        pt_args[t].core = core;
        pt_args[t].db = db;
        pt_args[t].start = i;
        pt_args[t].runner = t;
        i += step;
        if (i > n_reads) {
            pt_args[t].end = n_reads;
        } else {
            pt_args[t].end = i;
        }
    }

    double time_sync = 0;

    LOG_DEBUG("%s", "starting mod basecall");

    // modbase call
    // create threads
    for (t = 0; t < num_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_mod_basecall,
                                (void*)(&pt_args[t]));
        NEG_CHK(ret);
    }

    // pthread joining
    for (t = 0; t < num_threads; t++) {
        int ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
        if (t == 0) {
            time_sync -= realtime();
        }
        if (t == num_threads-1) {
            time_sync += realtime();
        }
    }

    core->time_sync += time_sync;
}
