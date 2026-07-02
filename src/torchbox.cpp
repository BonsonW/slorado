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
#include "calib.h"
#include "torchbox.h"
#include "dorado/tensor_chunk_utils.h"
#include "dorado/lstm_model.h"
#include "dorado/tx_model.h"
#include "dorado/modbase_model.h"
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

at::Tensor model_forward(runner_t *runner, const at::Tensor &x) {
    if (runner->bc_model) {
        if (runner->bc_family == MODEL_FAMILY_TX) {
            return tx_model_forward((tx_model_t *)runner->bc_model, x);
        }
        if (runner->bc_family == MODEL_FAMILY_FLSTM) {
            return flstm_model_forward((flstm_model_t *)runner->bc_model, x);
        }
        return lstm_model_forward((lstm_model_t *)runner->bc_model, x);
    }
    return runner->module->forward(x);
}

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
        LOG_TRACE("%s", "loading modbase model (procedural)");
        runner->bc_model = load_modbase_model_proc(*core->modbase_config, runner->tensor_opts, batch_size);
    } else {
        if (core->model_config->family == MODEL_FAMILY_TX) {
            LOG_TRACE("%s", "loading tx model (procedural)");
            tx_stats_t *model_stats = init_tx_stats();
            model_stats->calib_stats = core->calib_stats;
            model_stats->quant_config = core->quant_config;
            runner->bc_model = load_tx_model_proc(*core->model_config, runner->tensor_opts, model_stats, (core->opt.flag & SLORADO_FLASH) != 0, core->opt.quant ? core->opt.quant : "", core->opt.num_thread);
            runner->bc_family = MODEL_FAMILY_TX;
            (*core->runner_stats)[runner_idx]->model_stats = model_stats;
        } else if (core->model_config->family == MODEL_FAMILY_FLSTM) {
            LOG_TRACE("%s", "loading flstm model (procedural)");
            lstm_stats_t *model_stats = init_lstm_stats();
            model_stats->calib_stats = core->calib_stats;
            model_stats->quant_config = core->quant_config;
            runner->bc_model = load_flstm_model_proc(*core->model_config, runner->tensor_opts, model_stats);
            runner->bc_family = MODEL_FAMILY_FLSTM;
            (*core->runner_stats)[runner_idx]->model_stats = model_stats;
        } else {
            LOG_TRACE("%s", "loading lstm model (procedural)");
            lstm_stats_t *model_stats = init_lstm_stats();
            model_stats->calib_stats = core->calib_stats;
            model_stats->quant_config = core->quant_config;
            runner->bc_model = load_lstm_model_proc(*core->model_config, runner->tensor_opts, model_stats);
            runner->bc_family = MODEL_FAMILY_LSTM;
            (*core->runner_stats)[runner_idx]->model_stats = model_stats;
        }
    }

    LOG_TRACE("%s", "model populated");

    // Auto GPU batch size: run a dry N=1 forward pass and measure the actual peak activation
    // memory via PyTorch's allocator stats, then scale to fit available GPU memory.
    // This is more robust than hand-counting each model's layer activations.
    if (device != "cpu" && batch_size == 0) {
#ifdef USE_GPU
        if (!modbase) {
            c10::DeviceGuard device_guard(runner->tensor_opts.device());
            const auto device_idx = (c10::DeviceIndex)runner->device_idx;

            // Two dry forward passes (N=1 then N=2) to isolate the truly linear-in-N activation
            // cost via marginal difference. Fixed overhead (MIOpen workspace, first-call algorithm
            // search, per-layer buffers) cancels out: per_chunk = peak_N2 - peak_N1.
            auto run_trial = [&](int n) -> size_t {
                CACHING_ALLOCATOR_NS::resetPeakStats(device_idx);
                {
                    torch::InferenceMode no_grad;
                    auto trial = torch::zeros({n, 1, (int64_t)core->chunk_size},
                        torch::TensorOptions().dtype(runner->tensor_opts.dtype()).device(torch::kCPU));
                    auto out = model_forward(runner, trial.to(runner->tensor_opts.device()));
                    // Replicate call_chunks: transpose(0,1).contiguous() allocates a second
                    // N×T×C copy while the original scores tensor is still alive.
                    // Without this the trial misses half the output tensor cost.
                    out.transpose(0, 1).contiguous();
                    torch::cuda::synchronize(device_idx);
                }
                return (size_t)CACHING_ALLOCATOR_NS::getDeviceStats(device_idx)
                                   .allocated_bytes[0].peak;
            };

            size_t peak_n1 = 0, peak_n2 = 0;
            bool trial_ok = true;
            try {
                peak_n1 = run_trial(1);
                peak_n2 = run_trial(2);
            } catch (const c10::Error &e) {
                WARNING("auto GPU batch size: trial forward pass OOM on %s (%s), falling back to %d",
                        device.c_str(), e.what(), DEFAULT_GPU_BATCH_SIZE);
                trial_ok = false;
            }

            // Sanity check: if peak stats return 0, tracking is not working on this platform.
            if (trial_ok && peak_n1 == 0 && peak_n2 == 0) {
                WARNING("auto GPU batch size: allocator peak stats returned 0 on %s "
                        "(HIP peak tracking may be unavailable), falling back to %d",
                        device.c_str(), DEFAULT_GPU_BATCH_SIZE);
                trial_ok = false;
            }

            if (trial_ok) {
                CACHING_ALLOCATOR_NS::resetPeakStats(device_idx);
                size_t free_mem, total_mem;
#ifdef HAVE_CUDA
                cudaMemGetInfo(&free_mem, &total_mem);
#elif defined(HAVE_ROCM)
                hipMemGetInfo(&free_mem, &total_mem);
#endif
                // On multi-GCD ROCm setups (e.g. MI250X in unified partition mode),
                // hipMemGetInfo reports the combined HBM pool across both GCDs.  Each GCD
                // can only access half of that pool at local bandwidth; cap free_mem at
                // total_mem/2 so we budget for one GCD's share.  On CUDA, free <= total
                // by definition so the guard is just a sanity check.
#ifdef HAVE_ROCM
                if (free_mem > total_mem / 2) free_mem = total_mem / 2;
#else
                if (free_mem > total_mem) free_mem = total_mem;
#endif

                // Marginal cost: the truly linear-in-N component only.
                // Guard against measurement noise flipping the sign.
                const size_t per_n_pytorch = (peak_n2 > peak_n1) ? (peak_n2 - peak_n1) : peak_n1;

                // After the trials, activations are freed back to PyTorch's cache.
                // Available = truly free CUDA memory + the cached (reusable) PyTorch memory.
                auto stats_final = CACHING_ALLOCATOR_NS::getDeviceStats(device_idx);
                const size_t pytorch_cache = (size_t)stats_final.reserved_bytes[0].current
                                           - (size_t)stats_final.allocated_bytes[0].current;
                const size_t available = free_mem + pytorch_cache;

                // openfish gpubuf uses raw CUDA malloc, invisible to the PyTorch allocator
                const int T = (int)(core->chunk_size / core->model_stride);
                const size_t per_n_openfish = openfish_gpubuf_size(T, 1, core->model_config->state_len);
                const size_t per_n_total = per_n_pytorch + per_n_openfish;

                // peak_n1 = fixed overhead (algorithm search, per-layer buffers) regardless of N.
                const size_t budget = (available > peak_n1) ? (size_t)((available - peak_n1) * 0.45) : 0;
                batch_size = (per_n_total > 0) ? (int)(budget / per_n_total) : 1;

                if (batch_size < 1) batch_size = 1;

                const size_t max_input_len = 10000ULL * 6000ULL;
                const int max_batch = (int)(max_input_len / core->chunk_size);
                if (batch_size > max_batch) batch_size = max_batch;

                // Guard against MIOpen/cuDNN RNN int32 overflow.  MIOpen computes
                // sequence descriptor lengths as T × N × hidden_size using 32-bit
                // integers.  When this product exceeds INT_MAX the value wraps negative
                // and miopenRNN* throws "Lengths must be > 0".
                // e.g. DNA fast v5.0 (T=2499, hidden=256): max safe N = INT_MAX/(2499×256) = 3356 → 2048
                {
                    const int lstm_sz = core->model_config->lstm_size;
                    if (lstm_sz > 0 && T > 0) {
                        const int max_rnn = (int)(2147483647LL / ((int64_t)T * lstm_sz));
                        if (batch_size > max_rnn) batch_size = max_rnn;
                    }
                }

                // Round down to nearest power of 2
                if (batch_size > 1) {
                    int p = 1;
                    while (p * 2 <= batch_size) p *= 2;
                    batch_size = p;
                }

                fprintf(stderr, "[%s] %.1f MB free + %.1f MB pytorch cache = %.1f MB available "
                        "(%.1f MB fixed overhead) on %s, "
                        "%zu bytes/chunk (pytorch:%zu openfish:%zu), auto GPU batch size: %d\n",
                        __func__, free_mem / 1e6, pytorch_cache / 1e6, available / 1e6,
                        peak_n1 / 1e6, device.c_str(),
                        per_n_total, per_n_pytorch, per_n_openfish, batch_size);
            } else {
                batch_size = DEFAULT_GPU_BATCH_SIZE;
            }
        } else {
            batch_size = DEFAULT_GPU_BATCH_SIZE;
        }
#endif
    }

    // Allocate openfish GPU buffer and input tensor with the resolved batch size
    if (device != "cpu") {
#ifdef USE_GPU
        c10::DeviceGuard device_guard(runner->tensor_opts.device());
        runner->gpubuf = openfish_gpubuf_init(core->chunk_size / core->model_stride, batch_size, core->model_config->state_len);
#endif
    }

    if (modbase) {
        const int channels = NUM_BASES * core->modbase_config->general.kmer_len;
        runner->input_sigs = torch::zeros({batch_size, 1, (int64_t)core->modbase_config->context.chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
        runner->input_seqs = torch::zeros({batch_size, (int64_t)core->modbase_config->context.chunk_size, channels}, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
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

        std::string device = opt->device;
        core->runner_stats->push_back((runner_stat_t *)malloc(sizeof(runner_stat_t)));
        init_runner_stat((*core->runner_stats).back());

        core->runners->push_back(new runner_t());
        init_runner(core, (*core->runners).back(), model, device, opt->gpu_batch_size, torch::kF32, 0, false);

        if (core->modbase_config != NULL) {
            LOG_DEBUG("adding mod_base runner for device %s", device.c_str());
            core->mod_runners->push_back(new runner_t());
            init_runner(core, (*core->mod_runners).back(), model, device, opt->gpu_batch_size, torch::kF32, 0, true);
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
                init_runner(core, (*core->mod_runners).back(), model, device, opt->gpu_batch_size, torch::kF16, mod_runner_idx++, true);
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
        if (runner->bc_model) {
            if (runner->bc_family == MODEL_FAMILY_TX) free_tx_model((tx_model_t *)runner->bc_model);
            else if (runner->bc_family == MODEL_FAMILY_FLSTM) free_flstm_model((flstm_model_t *)runner->bc_model);
            else free_lstm_model((lstm_model_t *)runner->bc_model);
        }
        delete runner;

        if (core->modbase_config != NULL) {
            runner_t *mod_runner = (*core->mod_runners)[i];
            if (mod_runner->bc_model) free_modbase_model((modbase_model_t *)mod_runner->bc_model);
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

// Scale the raw signal of a single record into read_dat->scaled_signal and split it into
// overlapping basecall chunks. Shared by the batch path (preprocess_signal) and the
// streaming pipeline (preprocess worker). Assumes rec->len_raw_signal > 0.
void preprocess_signal(core_t *core, slow5_rec_t *rec, read_dat_t *read_dat, std::vector<basecall_chunk_t> &chunks) {
    opt_t opt = core->opt;
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

    create_basecall_chunks(chunks, read_dat->scaled_signal.size(0), core->chunk_size, opt.overlap, core->model_stride, read_dat);
}

void preprocess_signal_db(core_t *core, db_t *db, int32_t i) {
    slow5_rec_t *rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;

    (*db->basecall_chunks)[i].clear();
    if (len_raw_signal > 0) {
        read_dat_t *read_dat = (*db->read_dats)[i];
        if (read_dat == NULL) {
            read_dat = new read_dat_t;
            (*db->read_dats)[i] = read_dat;
        }
        preprocess_signal(core, rec, read_dat, (*db->basecall_chunks)[i]);
    }
}

// Per-read modbase preprocessing core. Assumes rec->len_raw_signal > 0. Fills mod_chunks (and the
// modbase state in read_dat). seq is a borrowed pointer that must outlive postprocess_modbase.
// Shared by the batch path (preprocess_modbase_db) and the streaming pipeline.
void preprocess_modbase(core_t *core, slow5_rec_t *rec, read_dat_t *read_dat, const char *seq, std::vector<uint8_t> &moves, std::vector<mod_chunk_t> &mod_chunks) {
    // read_dat is persistent across batches, clear per-base hit caches to avoid stale work.
    for (auto& hits : read_dat->per_base_hits_seq) {
        hits.clear();
    }
    for (auto& hits : read_dat->per_base_hits_sig) {
        hits.clear();
    }

    char *seq_mut = const_cast<char*>(seq);
    read_dat->seq = seq;

    LOG_TRACE("%s", "tensor_from_record");
    read_dat->scaled_signal = tensor_from_record(rec);

    LOG_TRACE("%s", "initialise_base_mod_probs");
    initialise_base_mod_probs(core, read_dat, seq_mut);

    // For RNA: Pad signal length to be evenly divisible by the canonical stride so that the
    // sequence to signal mapping is always stride aligned and not offset by any remainder
    // in the last move (which becomes the first move when reversed).
    // const size_t signal_len =
    //         m_is_rna_model ? utils::pad_to(signal.size(0), m_canonical_stride) : signal.size(0);
    const size_t signal_len = read_dat->scaled_signal.size(0);

    LOG_TRACE("%s", "populate_hits_seq");
    if (!populate_hits_seq(core, read_dat, seq_mut)) {
        // WARNING("%s", "coud not populate hits sequence, not an error");
        return;
    }

    LOG_TRACE("%s", "get_seq_to_sig_map");
    std::vector<uint64_t> seq_to_sig_map = get_seq_to_sig_map(moves, signal_len, strlen(seq) + 1, core->model_config->stride);

    LOG_TRACE("%s", "sequence_to_ints");
    std::vector<int> int_seq = sequence_to_ints(seq);

    auto base_id = core->modbase_config->mods.base_id;

    LOG_TRACE("%s", "populate_hits_sig");
    populate_hits_sig(read_dat->per_base_hits_sig, read_dat->per_base_hits_seq, seq_to_sig_map, base_id);

    LOG_TRACE("%s", "populate_signal");
    populate_signal(core, read_dat->scaled_signal, seq_to_sig_map, int_seq);

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

    const auto base_skips = get_minimal_encoding_skips(core, mod_chunks, seq_to_sig_map, int_seq);

    auto sequence_stride_ratio = general_stride_ratio(core->modbase_config->general);
    auto kmer_len = core->modbase_config->context.bases_before + core->modbase_config->context.bases_after + 1;

    populate_encoded_kmer(read_dat->encoded_kmers, read_dat->scaled_signal.size(0), int_seq, seq_to_sig_map, base_skips, kmer_len, sequence_stride_ratio);

    const std::size_t enc_kmer_size = read_dat->encoded_kmers.size();
    const std::size_t expected = (read_dat->scaled_signal.size(0) / sequence_stride_ratio) * kmer_len * NUM_BASES;
    if (enc_kmer_size != expected) {
        ERROR("%s", "Modbase kmer encoding failed");
        exit(1);
    }
}

void preprocess_modbase_db(core_t *core, db_t *db, int32_t i) {
    slow5_rec_t *rec = db->slow5_rec[i];

    (*db->mod_chunks)[i].clear();
    if (rec->len_raw_signal > 0) {
        preprocess_modbase(core, rec, (*db->read_dats)[i], (*db->sequence)[i].c_str(), (*db->moves)[i], (*db->mod_chunks)[i]);
    }
}

// Per-read modbase postprocessing core: turn read_dat->base_mod_probs into MM/ML output
// (mod_string_out + mod_prob_out). Shared by the batch path and the streaming pipeline.
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

    modbase_context_t context_handler{};
    mb_set_context(context_handler, core->modbase_config->mods.motif, size_t(core->modbase_config->mods.motif_offset));
    std::string context = mb_encode(context_handler);

    if (!context.empty()) {
        if (!mb_decode(context_handler, context)) {
            ERROR("%s", "Invalid base modification context string.");
            exit(1);
        }
        for (auto base : cardinal_bases) {
            if (mb_motif(context_handler, base).size() > 1) {
                // If the context is just the single base, then this is equivalent to no context.
                base_has_context[base] = true;
            }
        }
    } else { // this is something i added, not in dorado
        ERROR("%s", "Modbase context not found.");
        exit(1);
    }

    auto modbase_mask = need_to_generate_mask ? mb_get_sequence_mask(context_handler, seq_mut, strlen(seq)) : read_dat->base_mod_simplex_motif_hits;
    mb_update_mask(context_handler, modbase_mask, seq, core->modbase_info->alphabet, read_dat->base_mod_probs, threshold);

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

void postprocess_modbase_db(core_t *core, db_t *db, int32_t i) {
    if (db->slow5_rec[i]->len_raw_signal <= 0) { return; }
    postprocess_modbase(core, (*db->read_dats)[i], (*db->mod_string)[i], (*db->mod_prob)[i]);
}