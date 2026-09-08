/**
 * @file torchbox.c
 * @brief common functions for slorado that depends on torch
 * @author Hasindu Gamaarachchi (hasindu@unsw.edu.au)
 * @author Bonson Wong (bonson.ym@gmail.com)

MIT License

Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/
#include "error.h"
#include "misc.h"
#include "torchbox.h"
#include "dorado/tensor_chunk_utils.h"
#include "dorado/CRFModel.h"
#include "dorado/TxModel.h"
#include "dorado/ModBaseModel.h"
#include "dorado/modbase.h"
#include "dorado/simd.h"

#ifdef USE_GPU
#ifdef HAVE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#define CACHING_ALLOCATOR_NS c10::cuda::CUDACachingAllocator
#elif defined(HAVE_ROCM)
#include <c10/hip/HIPCachingAllocator.h>
#define CACHING_ALLOCATOR_NS c10::hip::HIPCachingAllocator
#endif
#endif

#ifdef USE_GPU
#include <c10/core/DeviceGuard.h>
#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#elif defined(HAVE_ROCM)
#include <hip/hip_runtime_api.h>
#endif
#endif

void free_read_dat(read_dat_t *read_dat) {
    delete read_dat;
}

#ifdef USE_GPU
static void gpu_mem_get_info(size_t *free_mem, size_t *total_mem) {
    *free_mem = 0;
    *total_mem = 0;
#ifdef HAVE_CUDA
    cudaMemGetInfo(free_mem, total_mem);
#elif defined(HAVE_ROCM)
    (void)hipMemGetInfo(free_mem, total_mem);
#endif
    // On multi-GCD ROCm setups (e.g. MI250X in unified partition mode) hipMemGetInfo reports the
    // combined HBM pool across both GCDs, but each GCD can only reach half of it at local
    // bandwidth; cap free at total/2 so we budget for one GCD's share. On CUDA free <= total by
    // definition, so the guard is just a sanity check.
#ifdef HAVE_ROCM
    if (*free_mem > *total_mem / 2) *free_mem = *total_mem / 2;
#else
    if (*free_mem > *total_mem) *free_mem = *total_mem;
#endif
}

// Probe whether a forward pass at batch size n fits in GPU memory.
static bool trial_fits(runner_t *runner, core_t *core, int est_chunk_size, int n,
                       size_t reserve_bytes, bool modbase) {
    const auto device_idx = (c10::DeviceIndex)runner->device_idx;
    bool ok = true;
    CACHING_ALLOCATOR_NS::emptyCache();
    try {
        torch::InferenceMode no_grad;
        at::Tensor reserve;
        if (reserve_bytes > 0) {
            reserve = torch::empty({(int64_t)reserve_bytes},
                torch::TensorOptions().dtype(torch::kUInt8).device(runner->tensor_opts.device()));
        }
        if (modbase) {
            const int channels = NUM_BASES * core->modbase_config->general.kmer_len;
            const int64_t seq_chunk = (int64_t)est_chunk_size / core->modbase_config->general.stride_ratio();
            auto sigs = torch::zeros({n, 1, (int64_t)est_chunk_size},
                torch::TensorOptions().dtype(runner->tensor_opts.dtype()).device(torch::kCPU));
            auto seqs = torch::zeros({n, seq_chunk, channels},
                torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
            auto out = runner->module->forward(sigs.to(runner->tensor_opts.device()),
                                               seqs.to(runner->tensor_opts.device()));
            out.contiguous();
        } else {
            auto in = torch::zeros({n, 1, (int64_t)est_chunk_size},
                torch::TensorOptions().dtype(runner->tensor_opts.dtype()).device(torch::kCPU));
            auto out = runner->module->forward(in.to(runner->tensor_opts.device()));
            // Replicate call_chunks: transpose(0,1).contiguous() allocates a second T*N*C copy
            // while the original scores tensor is still alive.
            out.transpose(0, 1).contiguous();
        }
        torch::cuda::synchronize(device_idx);
    } catch (const std::exception &) {
        ok = false;
    }
    CACHING_ALLOCATOR_NS::emptyCache();
    return ok;
}
#endif

std::vector<std::string> parse_cuda_device_string(std::string device_arg) {
    std::vector<std::string> devices;

    if (device_arg == "cuda:all" || device_arg == "cuda:auto") {
        for (int8_t i = 0; i < (int8_t)torch::cuda::device_count(); i++) {
            devices.push_back("cuda:" + std::to_string(i));
        }
        return devices;
    }

    std::string device_name = "";
    std::string delimiter = ":";
    size_t pos = device_arg.find(delimiter);
    device_name = device_arg.substr(0, pos + delimiter.length());
    device_arg.erase(0, pos + delimiter.length());

    delimiter = ",";
    while ((pos = device_arg.find(delimiter)) != std::string::npos) {
        devices.push_back(device_name + device_arg.substr(0, pos));
        device_arg.erase(0, pos + delimiter.length());
    }
    devices.push_back(device_name + device_arg.substr(0, pos));

    return devices;
}

lstm_stats_t *init_lstm_stats() {
    lstm_stats_t *lstm_stats = (lstm_stats_t *)calloc(1, sizeof(lstm_stats_t));
    MALLOC_CHK(lstm_stats);
    return lstm_stats;
}

tx_stats_t *init_tx_stats() {
    tx_stats_t *tx_stats = (tx_stats_t *)calloc(1, sizeof(tx_stats_t));
    MALLOC_CHK(tx_stats);
    return tx_stats;
}

/* initialise runners */
void init_runner(
    core_t* core,
    runner_t* runner,
    char *model_path,
    const std::string &device,
    int &batch_size,
    torch::ScalarType dtype,
    int runner_idx,
    bool modbase
) {
    LOG_TRACE("initializing model runner for device %s", device.c_str());
    runner->device = device;

    if (device != "cpu") {
#ifdef USE_GPU
        int64_t device_idx = device[device.size()-1] - '0'; // quick and dirty device index extraction
        runner->device_idx = device_idx;
        runner->tensor_opts = torch::TensorOptions().dtype(dtype).device(c10::kCUDA, device_idx);
#endif
    } else {
        runner->tensor_opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    }

    LOG_TRACE("%s", "device str parsed");

    // Load model first so we can query remaining GPU memory for auto batch size
    if (modbase == true) {
        LOG_TRACE("%s", "loading modbase model");
        runner->module = load_modbase_model(*core->modbase_config, runner->tensor_opts, batch_size);
    } else {
        if (core->model_config->tx != NULL) {
            LOG_TRACE("%s", "loading tx model");
            tx_stats_t *model_stats = init_tx_stats();
            runner->module = load_tx_model(*core->model_config, runner->tensor_opts, model_stats, (core->opt.flag & SLORADO_FLASH) != 0, core->opt.num_thread);
            (*core->runner_stats)[runner_idx]->model_stats = model_stats;
        } else {
            LOG_TRACE("%s", "loading lstm model");
            lstm_stats_t *model_stats = init_lstm_stats();
            runner->module = load_lstm_model(*core->model_config, runner->tensor_opts, model_stats);
            (*core->runner_stats)[runner_idx]->model_stats = model_stats;
        }
    }

    LOG_TRACE("%s", "model populated");

    // try powers of two descending from MAX_AUTO_GPU_BATCH_SIZE and stop at the first that fits
    if (device != "cpu" && batch_size == 0) {
#ifdef USE_GPU
        c10::DeviceGuard device_guard(runner->tensor_opts.device());
        const int est_chunk_size = modbase
            ? (int)core->modbase_config->context.chunk_size
            : (int)core->chunk_size;
        const int T = est_chunk_size / (modbase ? 1 : (int)core->model_stride);

        // upper bound on the search
        int hi = MAX_AUTO_GPU_BATCH_SIZE;
        const size_t max_input_len = 10000ULL * 6000ULL;
        const int max_batch = (int)(max_input_len / est_chunk_size);
        if (hi > max_batch) hi = max_batch;
        const bool cudnn_rnn = modbase || (core->model_config->tx == NULL && core->model_config->lstm_inner_dim < 0);
        if (cudnn_rnn) {
            const int mstride = (modbase && core->modbase_config->general.stride > 0) ? core->modbase_config->general.stride : 1;
            const int lstm_sz = modbase ? core->modbase_config->general.size : core->model_config->lstm_size;
            const int Tb = modbase ? (int)(core->modbase_config->context.chunk_size / mstride) : T;
            if (lstm_sz > 0 && Tb > 0) {
                const int max_rnn = (int)(2147483647LL / ((int64_t)Tb * lstm_sz));
                if (hi > max_rnn) hi = max_rnn;
            }
        }
        if (hi < 1) hi = 1;
        { int p = 1; while (p * 2 <= hi) p *= 2; hi = p; } // round hi down to a power of two

        // leave ~10% of total memory free for runtime fragmentation and other processes
        size_t free_mem = 0, total_mem = 0;
        gpu_mem_get_info(&free_mem, &total_mem);
        const size_t headroom = total_mem / 10;

        int chosen = 0;
        for (int n = hi; n >= 1; n /= 2) {
            const size_t gpubuf_bytes = modbase ? 0 : openfish_gpubuf_size(T, n, core->model_config->state_len);
            if (trial_fits(runner, core, est_chunk_size, n, gpubuf_bytes + headroom, modbase)) {
                chosen = n;
                break;
            }
        }
        if (chosen < 1) {
            WARNING("auto GPU batch size: no batch size fit on %s, falling back to %d",
                    device.c_str(), DEFAULT_GPU_BATCH_SIZE);
            batch_size = DEFAULT_GPU_BATCH_SIZE;
        } else {
            batch_size = chosen;
        }

        fprintf(stderr, "[%s] %.1f MB free / %.1f MB total on %s, auto GPU batch size: %d%s\n",
                __func__, free_mem / 1e6, total_mem / 1e6, device.c_str(),
                batch_size, modbase ? " [modbase]" : "");
#endif
    }

    // Allocate openfish GPU buffer and input tensor with the resolved batch size
    // (the decode buffer is basecall-only; the modbase path does not decode with openfish)
    if (device != "cpu" && !modbase) {
#ifdef USE_GPU
        c10::DeviceGuard device_guard(runner->tensor_opts.device());
        runner->gpubuf = openfish_gpubuf_init(core->chunk_size / core->model_stride, batch_size, core->model_config->state_len);
#endif
    }

    if (modbase) {
        const int channels = NUM_BASES * core->modbase_config->general.kmer_len;
        // The signal chunk is at signal resolution; the sequence (kmer) chunk is at the sequence
        // resolution = chunk_size / stride_ratio (ratio > 1 for conv_lstm_v3, whose sequence convs
        // are stride-1; ratio == 1 for v1/v2, which downsample the sequence in-conv).
        const int64_t seq_ratio = core->modbase_config->general.stride_ratio();
        const int64_t seq_chunk = (int64_t)core->modbase_config->context.chunk_size / seq_ratio;
        runner->input_sigs = torch::zeros({batch_size, 1, (int64_t)core->modbase_config->context.chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
        runner->input_seqs = torch::zeros({batch_size, seq_chunk, channels}, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
    } else {
        runner->input_tensor = torch::zeros({batch_size, 1, (int64_t)core->chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    }

    LOG_DEBUG("fully initialized model runner for device %s", device.c_str());
}

/* initialise runner_stat */
void init_runner_stat(runner_stat_t *time_stamps) {
    memset(time_stamps, 0, sizeof(runner_stat_t));
}

void init_runners(core_t* core, opt_t *opt, char *model) {
    core->runners = new std::vector<runner_t *>();
    core->mod_runners = new std::vector<runner_t *>();
    core->runner_stats = new std::vector<runner_stat_t *>();

    if (strcmp(opt->device, "cpu") == 0) {
        // No GPU memory to query; fall back to default batch size for CPU
        if (opt->gpu_batch_size == 0) {
            opt->gpu_batch_size = DEFAULT_GPU_BATCH_SIZE;
        }
        if (opt->mod_gpu_batch_size == 0) {
            opt->mod_gpu_batch_size = DEFAULT_GPU_BATCH_SIZE;
        }

        std::string device = opt->device;
        core->runner_stats->push_back((runner_stat_t *)malloc(sizeof(runner_stat_t)));
        init_runner_stat((*core->runner_stats).back());

        core->runners->push_back(new runner_t());
        init_runner(core, (*core->runners).back(), model, device, opt->gpu_batch_size, torch::kF32, 0, false);

        if (core->modbase_config != NULL) {
            LOG_DEBUG("adding mod_base runner for device %s", device.c_str());
            core->mod_runners->push_back(new runner_t());
            init_runner(core, (*core->mod_runners).back(), model, device, opt->mod_gpu_batch_size, torch::kF32, 0, true);
        }
    } else {
#ifdef USE_GPU
        std::vector<std::string> devices;
        std::string device_args = std::string(opt->device);
        devices = parse_cuda_device_string(device_args);
        if (devices.size() < 1) {
            ERROR("%s", "Could not locate any cuda devices");
            exit(EXIT_FAILURE);
        }

        int runner_idx = 0;
        int mod_runner_idx = 0;
        for (auto device: devices) {
            core->runner_stats->push_back((runner_stat_t *)malloc(sizeof(runner_stat_t)));
            init_runner_stat((*core->runner_stats).back());
            core->runners->push_back(new runner_t());
            // init_runner auto-detects when opt->gpu_batch_size == 0 and updates it in-place;
            // subsequent runners (including modbase and other GPUs) then inherit the computed value
            init_runner(core, (*core->runners).back(), model, device, opt->gpu_batch_size, torch::kF16, runner_idx++, false);

            if (core->modbase_config != NULL) {
                LOG_DEBUG("adding mod_base runner for device %s", device.c_str());
                core->mod_runners->push_back(new runner_t());
                init_runner(core, (*core->mod_runners).back(), model, device, opt->mod_gpu_batch_size, torch::kF16, mod_runner_idx++, true);
            }
        }
#else
        ERROR("Invalid device: %s. Please compile again for GPU", opt->device);
        exit(EXIT_FAILURE);
#endif
    }

    auto adjusted_chunk_size = core->chunk_size;
    if (opt->chunk_size != adjusted_chunk_size) {
        LOG_DEBUG("Adjusting chunk size to %zu", adjusted_chunk_size);
        opt->chunk_size = adjusted_chunk_size;
    }
}

void free_runners(core_t *core) {
    for (size_t i = 0; i < core->runner_stats->size(); ++i) {
        free((*core->runner_stats)[i]->model_stats);
        free((*core->runner_stats)[i]);
    }

    for (size_t i = 0; i < core->runners->size(); ++i) {
        runner_t *runner = (*core->runners)[i];
        if (runner->device != "cpu") {
#ifdef USE_GPU
            c10::DeviceGuard device_guard(runner->tensor_opts.device());
            openfish_gpubuf_free(runner->gpubuf);
#endif
        }
        delete runner;

        if (core->modbase_config != NULL) {
            runner_t *mod_runner = (*core->mod_runners)[i];
            delete mod_runner;
        }
    }

}

torch::Tensor tensor_from_record(slow5_rec_t *rec) {
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt16).requires_grad(false);
    return torch::from_blob(rec->raw_signal, rec->len_raw_signal, options);
}

size_t create_basecall_chunks(std::vector<basecall_chunk_t> &chunks, size_t num_samples, size_t chunk_size, size_t overlap, size_t stride, read_dat_t *read_dat) {
    ASSERT(chunks.size() == 0);
    std::size_t offset = 0;
    std::size_t last_offset = (num_samples > chunk_size) ? (num_samples - chunk_size) : 0;
    const std::size_t misalignment = last_offset % stride;
    if (misalignment != 0) {
        // Move last chunk start to the next stride boundary, we'll zero pad any excess samples required.
        last_offset += stride - misalignment;
    }
    const std::size_t chunk_step = chunk_size - overlap;

    size_t i = 1;
    chunks.push_back({0, 0, chunk_size, std::string(), std::string(), std::vector<uint8_t>(), read_dat});
    while ((offset + chunk_size) < num_samples) {
        offset = std::min(offset + chunk_step, last_offset);
        chunks.push_back({offset, i, chunk_size, std::string(), std::string(), std::vector<uint8_t>(), read_dat});
        i += 1;
    }
    
    return chunks.size();
}

// Per-read core, shared by the batch path (preprocess_signal_db) and the async pipeline.
void preprocess_signal(core_t *core, slow5_rec_t *rec, read_dat_t *read_dat, std::vector<basecall_chunk_t> &chunks) {
    uint64_t len_raw_signal = rec->len_raw_signal;
    opt_t opt = core->opt;

    chunks.clear();
    if (len_raw_signal > 0) {
        auto signal_norm_params = core->model_config->signal_norm_params;

        // if we are doing modbase calling, we need to keep the original signal for the modbase preproc,
        // so clone the tensor here to avoid in-place scaling modifying the original tensor.
        // if not doing modbase calling, we can save memory by not cloning and just using the same tensor for scaling and basecalling.
        if (opt.mod != NULL) {
            read_dat->scaled_signal = tensor_from_record(rec).clone();
        } else {
            read_dat->scaled_signal = tensor_from_record(rec);
        }

        scale_signal(core, read_dat->scaled_signal, rec->range / rec->digitisation, rec->offset, signal_norm_params);
        LOG_TRACE("%s", "scaled signal");

        // scale_signal front-trims the signal (RNA adapter / DNA pore-open). Record how much so the
        // modbase path can trim the raw signal identically and stay aligned with the moves/sequence.
        read_dat->basecall_trim_start = (int64_t)len_raw_signal - read_dat->scaled_signal.size(0);

        create_basecall_chunks(chunks, read_dat->scaled_signal.size(0), core->chunk_size, opt.overlap, core->model_stride, read_dat);
    }
}

void preprocess_signal_db(core_t *core, db_t *db, int32_t i) {
    read_dat_t *read_dat = (*db->read_dats)[i];
    if (read_dat == NULL && db->slow5_rec[i]->len_raw_signal > 0) {
        read_dat = new read_dat_t;
        (*db->read_dats)[i] = read_dat;
    }
    preprocess_signal(core, db->slow5_rec[i], read_dat, (*db->basecall_chunks)[i]);
}

// Reverse a seq_to_sig_map for RNA modbase models: reverse the order AND map each coord
// a -> signal_len - a, so the (unreversed 5'->3') sequence lines up with the time-flipped RNA signal.
static void reverse_seq_to_sig_map(std::vector<uint64_t>& m, size_t signal_len) {
    const size_t n = m.size();
    for (size_t l = 0; l < n / 2; ++l) {
        const size_t r = n - l - 1;
        uint64_t lv = signal_len - m[l];
        uint64_t rv = signal_len - m[r];
        m[l] = rv;
        m[r] = lv;
    }
    if (n % 2 != 0) m[n / 2] = signal_len - m[n / 2];
}

// Per-read core, shared by the batch path (preprocess_modbase_db) and the async pipeline.
// seq is a borrowed pointer that must stay alive until postprocess_modbase has run.
void preprocess_modbase(core_t *core, slow5_rec_t *rec, read_dat_t *read_dat, const char *seq, std::vector<uint8_t> &moves, std::vector<mod_chunk_t> &mod_chunks) {
    uint64_t len_raw_signal = rec->len_raw_signal;
    // double a, b;

    mod_chunks.clear();
    if (len_raw_signal > 0) {
        // read_dat is persistent across batches, clear per-base hit caches to avoid stale work.
        for (auto& hits : read_dat->per_base_hits_seq) {
            hits.clear();
        }
        for (auto& hits : read_dat->per_base_hits_sig) {
            hits.clear();
        }

        char *seq_mut = const_cast<char*>(seq);
        read_dat->seq = seq;

        // RNA models process the signal 3'->5' (reverse_signal=true).
        const bool is_rna_mod = core->modbase_config->context.reverse;

        LOG_TRACE("%s", "tensor_from_record");

        // a = realtime();
        // Trim the raw signal by the same front-trim the basecall scaler applied (RNA adapter / DNA
        // pore-open), matching dorado (its modbase raw_data is the trimmed signal). Without this the
        // modbase signal keeps samples the moves/sequence do not account for, shifting signal vs
        // sequence by the trim length (catastrophic for RNA's large adapter; small but real for DNA).
        at::Tensor raw_full = tensor_from_record(rec);
        const int64_t trim = read_dat->basecall_trim_start;
        read_dat->scaled_signal = (trim > 0 && trim < raw_full.size(0))
            ? raw_full.slice(0, trim, raw_full.size(0)).contiguous()
            : raw_full;
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_tens_from_rec += (b-a);

        LOG_TRACE("%s", "initialise_base_mod_probs");

        // a = realtime();
        initialise_base_mod_probs(core, read_dat, seq_mut);
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_init_base_mod_probs += (b-a);

        // For RNA: Pad signal length to be evenly divisible by the canonical stride so that the
        // sequence to signal mapping is always stride aligned and not offset by any remainder
        // in the last move (which becomes the first move when reversed).
        const int cstride = core->model_config->stride;
        const size_t raw_len = read_dat->scaled_signal.size(0);
        const size_t signal_len = is_rna_mod ? ((raw_len + cstride - 1) / cstride) * cstride : raw_len;

        LOG_TRACE("%s", "populate_hits_seq");

        if (!populate_hits_seq(core, read_dat, seq_mut)) {
            // WARNING("%s", "coud not populate hits sequence, not an error");
            return;
        }

        LOG_TRACE("%s", "get_seq_to_sig_map");

        // moves arrive in signal (3'->5') order for RNA. The map is built in signal order here and
        // reoriented by reverse_seq_to_sig_map below to line up with the 5'->3' sequence.
        if (is_rna_mod) {
            // Defensive check: stitch_chunks bounds moves by the adapter-trimmed length so they span
            // exactly signal_len/cstride blocks with one 1 per base. If that invariant ever breaks
            // (e.g. an untrimmed read), a move block past signal_len would underflow
            // reverse_seq_to_sig_map, so skip modbase (leave default probs) rather than emit a
            // misaligned map.
            size_t move_ones = 0;
            for (uint8_t m : moves) move_ones += (m != 0);
            if (moves.size() > signal_len / cstride || move_ones != strlen(seq)) {
                return;
            }
        }

        // a = realtime();
        std::vector<uint64_t> seq_to_sig_map = get_seq_to_sig_map(moves, signal_len, strlen(seq) + 1, cstride);
        if (is_rna_mod) {
            reverse_seq_to_sig_map(seq_to_sig_map, signal_len);
        }
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_seq_to_sig_map += (b-a);

        LOG_TRACE("%s", "sequence_to_ints");
        // a = realtime();
        std::vector<int> int_seq = sequence_to_ints(seq);
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_seq_to_ints += (b-a);

        auto base_id = core->modbase_config->mods.base_id;

        LOG_TRACE("%s", "populate_hits_sig");

        // a = realtime();
        populate_hits_sig(read_dat->per_base_hits_sig, read_dat->per_base_hits_seq, seq_to_sig_map, base_id);
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_populate_hits_sig += (b-a);

        if (is_rna_mod) {
            // RNA models process the signal 3'->5', so flip it and prepend a short mirrored pad up to
            // the stride-aligned signal_len: sig = [ raw[len-pad:len] , flip(raw) ]
            // (dorado ModBaseChunkCallerNode::populate_signal).
            auto raw = read_dat->scaled_signal;
            const int64_t len = raw.size(0);
            const int64_t pad = (int64_t)signal_len - len;
            at::Tensor sig = at::empty({(int64_t)signal_len}, raw.options());
            sig.slice(0, pad, (int64_t)signal_len) = at::flip(raw, 0);
            if (pad > 0) {
                // pad < cstride, so it only exceeds len for pathologically short reads.
                if (pad <= len) sig.slice(0, 0, pad) = raw.slice(0, len - pad, len);
                else sig.slice(0, 0, pad).zero_();
            }
            read_dat->scaled_signal = sig;
        }

        LOG_TRACE("%s", "populate_signal");

        // a = realtime();
        populate_signal(core, read_dat->scaled_signal, seq_to_sig_map, int_seq);
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_populate_signal += (b-a);

        if (signal_len != static_cast<size_t>(read_dat->scaled_signal.size(0))) {
            ERROR("%s", "modbase signal length is incorrect for read");
            exit(EXIT_FAILURE);
        }

        for (const auto& per_base_hits : read_dat->per_base_hits_seq) {
            for (std::size_t hit : per_base_hits) {
                read_dat->base_mod_simplex_motif_hits.at(hit) = true;
            }
        }

        create_mod_chunks(mod_chunks, core, read_dat);

        // a = realtime();
        const auto base_skips = get_minimal_encoding_skips(core, mod_chunks, seq_to_sig_map, int_seq);
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_get_minimal_encoding_skips += (b-a);

        auto sequence_stride_ratio = core->modbase_config->general.stride_ratio();
        auto kmer_len = core->modbase_config->context.bases_before + core->modbase_config->context.bases_after + 1;

        // a = realtime();
        populate_encoded_kmer(read_dat->encoded_kmers, read_dat->scaled_signal.size(0), int_seq, seq_to_sig_map, base_skips, kmer_len, sequence_stride_ratio);
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_populate_encoded_kmer += (b-a);

        const std::size_t enc_kmer_size = read_dat->encoded_kmers.size();
        const std::size_t expected = (read_dat->scaled_signal.size(0) / sequence_stride_ratio) * kmer_len * NUM_BASES;
        if (enc_kmer_size != expected) {
            ERROR("%s", "Modbase kmer encoding failed");
            exit(1);
        }
    }
}

// Per-read core, shared by the batch path (postprocess_modbase_db) and the async pipeline.
void postprocess_modbase(core_t *core, read_dat_t *read_dat, std::string &mod_string_out, std::vector<uint8_t> &mod_prob_out) {
    const auto threshold_float = 0.05f;
    const auto threshold = static_cast<uint8_t>(std::min(threshold_float * 256.0f, 255.0f));

    const size_t num_channels = core->modbase_info->alphabet.size();
    const std::string cardinal_bases = "ACGT";
    const char *seq = read_dat->seq;
    char *seq_mut = const_cast<char*>(seq);
    const auto seqlen = strlen(seq);

    if (seqlen * num_channels != read_dat->base_mod_probs.size()) {
        ERROR("%s", "Mismatch between base_mod_probs size and sequence length * num channels in modbase_alphabet!");
    }

    std::string modbase_string = "";
    std::vector<uint8_t> modbase_prob;

    // Duplex doesn't retain the mask, and tests may not have it set.
    const bool need_to_generate_mask = read_dat->base_mod_simplex_motif_hits.empty();

    // Create a mask indicating which bases are modified.
    std::bitset<256> base_has_context{};

    ModBaseContext context_handler;
    context_handler.set_context(core->modbase_config->mods.motif, size_t(core->modbase_config->mods.motif_offset));
    std::string context = context_handler.encode();
    // ERROR("%s", context.c_str());
    // exit(1);

    if (!context.empty()) {
        if (!context_handler.decode(context, need_to_generate_mask)) {
            ERROR("%s", "Invalid base modification context string.");
            exit(1);
        }
        for (auto base : cardinal_bases) {
            if (context_handler.motif(base).size() > 1) {
                // If the context is just the single base, then this is equivalent to no context.
                base_has_context[base] = true;
            }
        }
    } else { // this is something i added, not in dorado
        ERROR("%s", "Modbase context not found.");
        exit(1);
    }

    auto modbase_mask = need_to_generate_mask ? context_handler.get_sequence_mask(seq_mut, strlen(seq)) : read_dat->base_mod_simplex_motif_hits;
    context_handler.update_mask(modbase_mask, seq, core->modbase_info->alphabet, read_dat->base_mod_probs, threshold);

    // Iterate over the provided alphabet and find all the channels we need to write out
    char current_cardinal = 0;
    for (size_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
        if (cardinal_bases.find(core->modbase_info->alphabet[channel_idx]) != std::string::npos) {
            // A cardinal base
            current_cardinal = core->modbase_info->alphabet[channel_idx][0];
        } else {
            // A modification on the previous cardinal base
            std::string bam_name = core->modbase_info->alphabet[channel_idx];
            if (!validate_bam_tag_code(bam_name)) {
                return;
            }

            // Write out the results we found
            modbase_string += std::string(1, current_cardinal) + "+" + bam_name;
            modbase_string += base_has_context.test(static_cast<uint8_t>(current_cardinal)) ? "?" : ".";
            int skipped_bases = 0;
            for (size_t base_idx = 0; base_idx < seqlen; base_idx++) {
                if (seq[base_idx] == current_cardinal) {
                    if (modbase_mask[base_idx]) {
                        modbase_string += "," + std::to_string(skipped_bases);
                        skipped_bases = 0;
                        modbase_prob.push_back(read_dat->base_mod_probs[base_idx * num_channels + channel_idx]);
                    } else {
                        // Skip this base
                        skipped_bases++;
                    }
                }
            }
            modbase_string += ";";
        }
    }

    mod_string_out = std::move(modbase_string);
    mod_prob_out = std::move(modbase_prob);
}

void preprocess_modbase_db(core_t *core, db_t *db, int32_t i) {
    preprocess_modbase(core, db->slow5_rec[i], (*db->read_dats)[i], (*db->sequence)[i].c_str(), (*db->moves)[i], (*db->mod_chunks)[i]);
}

void postprocess_modbase_db(core_t *core, db_t *db, int32_t i) {
    if (db->slow5_rec[i]->len_raw_signal <= 0) { return; }
    postprocess_modbase(core, (*db->read_dats)[i], (*db->mod_string)[i], (*db->mod_prob)[i]);
}