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

#include <regex>

#ifdef HAVE_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif

#ifdef HAVE_ROCM
#include <c10/hip/HIPGuard.h>
#endif

void free_read_dat(read_dat_t *read_dat) {
    delete read_dat;
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
    int batch_size,
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

#ifdef HAVE_CUDA
        c10::cuda::CUDAGuard device_guard(device_idx);
#endif
#ifdef HAVE_ROCM
        c10::hip::HIPGuard device_guard(device_idx);
#endif
        runner->gpubuf = openfish_gpubuf_init(core->chunk_size / core->model_stride, batch_size, core->model_config->state_len);
#endif        
    }

    LOG_TRACE("%s", "device str parsed");

    runner->tensor_opts = torch::TensorOptions().dtype(dtype).device(device);
    if (modbase == true) {
        LOG_TRACE("%s", "loading modbase model");
        runner->module = load_modbase_model(*core->modbase_config, runner->tensor_opts, core->opt.gpu_batch_size);
    } else {
        if (core->model_config->tx != NULL) {
            LOG_TRACE("%s", "loading tx model");
            tx_stats_t *model_stats = init_tx_stats();
            runner->module = load_tx_model(*core->model_config, runner->tensor_opts, model_stats, (core->opt.flag & SLORADO_FLASH) != 0, core->opt.num_thread);
            (*core->runner_stats)[runner_idx]->model_stats = model_stats;
        } else {
            LOG_TRACE("%s", "loading lstm model");
            lstm_stats_t *model_stats = init_lstm_stats();
            runner->module = load_lstm_model(*core->model_config, runner->tensor_opts);
            (*core->runner_stats)[runner_idx]->model_stats = model_stats;
        }
    }
    
    LOG_TRACE("%s", "model populated");

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
#ifdef HAVE_CUDA
            c10::cuda::CUDAGuard device_guard(runner->device_idx);
#endif
#ifdef HAVE_ROCM
            c10::hip::HIPGuard device_guard(runner->device_idx);
#endif
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

void preprocess_signal(core_t *core, db_t *db, int32_t i) {
    slow5_rec_t *rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;
    opt_t opt = core->opt;

    (*db->basecall_chunks)[i].clear();
    if (len_raw_signal > 0) {
        read_dat_t *read_dat = (*db->read_dats)[i];
        if (read_dat == NULL) {
            read_dat = new read_dat_t;
            (*db->read_dats)[i] = read_dat;
        }
        auto signal_norm_params = core->model_config->signal_norm_params;

        read_dat->scaled_signal = tensor_from_record(rec);

        scale_signal(core, read_dat->scaled_signal, rec->range / rec->digitisation, rec->offset, signal_norm_params);
        LOG_TRACE("%s", "scaled signal");

        create_basecall_chunks((*db->basecall_chunks)[i], read_dat->scaled_signal.size(0), core->chunk_size, opt.overlap, core->model_stride, read_dat);
    }
}

void preprocess_modbase(core_t *core, db_t *db, int32_t i) {
    slow5_rec_t *rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;
    double a, b;

    (*db->mod_chunks)[i].clear();
    if (len_raw_signal > 0) {
        read_dat_t *read_dat = (*db->read_dats)[i];

        // read_dat is persistent across batches, clear per-base hit caches to avoid stale work.
        for (auto& hits : read_dat->per_base_hits_seq) {
            hits.clear();
        }
        for (auto& hits : read_dat->per_base_hits_sig) {
            hits.clear();
        }

        char *seq = (*db->sequence)[i];
        read_dat->seq = seq;

        std::vector<uint8_t> &moves = (*db->moves)[i];

        LOG_TRACE("%s", "tensor_from_record");

        // a = realtime();
        read_dat->scaled_signal = tensor_from_record(rec);
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_tens_from_rec += (b-a);
        
        LOG_TRACE("%s", "initialise_base_mod_probs");

        // a = realtime();
        initialise_base_mod_probs(core, read_dat, seq);
        // b = realtime();
        // if (core->opt.num_thread == 1) core->time_init_base_mod_probs += (b-a);

        // For RNA: Pad signal length to be evenly divisible by the canonical stride so that the
        // sequence to signal mapping is always stride aligned and not offset by any remainder
        // in the last move (which becomes the first move when reversed).
        // const size_t signal_len =
        //         m_is_rna_model ? utils::pad_to(signal.size(0), m_canonical_stride) : signal.size(0);
        const size_t signal_len = read_dat->scaled_signal.size(0);

        LOG_TRACE("%s", "populate_hits_seq");

        if (!populate_hits_seq(core, read_dat, seq)) {
            // WARNING("%s", "coud not populate hits sequence, not an error");
            return;
        }

        LOG_TRACE("%s", "get_seq_to_sig_map");

        // a = realtime();
        std::vector<uint64_t> seq_to_sig_map = get_seq_to_sig_map(moves, signal_len, strlen(seq) + 1, core->model_config->stride);
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

        create_mod_chunks((*db->mod_chunks)[i], core, read_dat);

        // a = realtime();
        const auto base_skips = get_minimal_encoding_skips(core, (*db->mod_chunks)[i], seq_to_sig_map, int_seq);
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

void postprocess_modbase(core_t *core, db_t *db, int32_t i) {
    slow5_rec_t *rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;

    if (len_raw_signal <= 0) { return; }

    const auto threshold_float = 0.05f;
    const auto threshold = static_cast<uint8_t>(std::min(threshold_float * 256.0f, 255.0f));

    const size_t num_channels = core->modbase_info->alphabet.size();
    const std::string cardinal_bases = "ACGT";
    read_dat_t *read_dat = (*db->read_dats)[i];
    char *seq = read_dat->seq;
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

    auto modbase_mask = need_to_generate_mask ? context_handler.get_sequence_mask(seq, strlen(seq)) : read_dat->base_mod_simplex_motif_hits;
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

    const size_t mod_str_len = modbase_string.size() + 1;
    char* mod_str = static_cast<char*>(realloc((*db->mod_string)[i], mod_str_len));
    MALLOC_CHK(mod_str);
    std::memcpy(mod_str, modbase_string.c_str(), mod_str_len);
    (*db->mod_string)[i] = mod_str;

    (*db->mod_prob)[i] = std::move(modbase_prob);
}