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

#include <regex>

#ifdef HAVE_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif

#ifdef HAVE_ROCM
#include <c10/hip/HIPGuard.h>
#endif

const std::vector<int> BASE_IDS = []() {
    std::vector<int> base_ids(256, -1);
    base_ids['A'] = 0;
    base_ids['C'] = 1;
    base_ids['G'] = 2;
    base_ids['T'] = 3;
    return base_ids;
}();

std::vector<uint64_t> moves_to_map(
    const std::vector<uint8_t>& moves,
    size_t block_stride,
    size_t signal_len,
    size_t reserve
) {
    std::vector<uint64_t> seq_to_sig_map;
    seq_to_sig_map.reserve(reserve);

    LOG_TRACE("seq_to_sig reserved: %zu", reserve);

    for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i] == 1) {
            seq_to_sig_map.push_back(i * block_stride);
        }
    }

    seq_to_sig_map.push_back(signal_len);

    LOG_TRACE("seq_to_sig len: %zu", seq_to_sig_map.size());
    LOG_TRACE("moves len: %zu", moves.size());

    return seq_to_sig_map;
}

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
            runner->module = load_tx_model(*core->model_config, runner->tensor_opts, model_stats, (core->opt.flag & SLORADO_FLS) != 0, core->opt.num_thread);
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

void initialise_base_mod_probs(const core_t *core, read_dat_t *read_dat, char *seq) {
    auto num_states = NUM_BASES + core->modbase_config->mods.count;
    auto seqlen = strlen(seq);
    read_dat->base_mod_probs.resize(seqlen * num_states, 0);
    for (size_t i = 0; i < seqlen; ++i) {
        // init what corresponds to 100% canonical base for each position (one-hot encoding)
        assert(seq[i] >= 0);
        int base_id = BASE_IDS.at(seq[i]);
        if (base_id < 0) {
            ERROR("%s", "invalid char");
        }

        auto offset = core->modbase_info->base_probs_offsets.at(base_id);
        if (offset < 0 || offset >= num_states) {
            ERROR("offset %ld out of range for base_id %d", offset, base_id);
            exit(1);
        }
        read_dat->base_mod_probs[i * num_states + core->modbase_info->base_probs_offsets.at(base_id)] = 1;
    }
    read_dat->base_mod_simplex_motif_hits.resize(seqlen, false);
}

std::vector<size_t> get_motif_hits(char *seq, std::string motif, size_t motif_offset) {
    std::vector<size_t> context_hits;

    MotifMatcher matcher(motif, motif_offset);
    return matcher.get_motif_hits(seq, strlen(seq));
}

bool populate_hits_seq(const core_t *core, read_dat_t *read_dat, char *seq) {
    bool has_hits = false;
    auto modbase_config = core->modbase_config;
    auto motif = modbase_config->mods.motif;
    auto motif_offset = modbase_config->mods.motif_offset;

    const std::vector<size_t> motif_hits = get_motif_hits(seq, motif, motif_offset);
    auto& hits_seq = read_dat->per_base_hits_seq.at(modbase_config->mods.base_id);
    hits_seq.resize(motif_hits.size());

    for (size_t i = 0; i < motif_hits.size(); ++i) {
        hits_seq[i] = static_cast<int64_t>(motif_hits[i]);
    }

    has_hits |= !hits_seq.empty();
    return has_hits;
}

std::vector<uint64_t> get_seq_to_sig_map(
    const std::vector<uint8_t>& moves,
    const size_t signal_len,
    const size_t reserve,
    size_t canonical_stride
) {
    auto seq_to_sig_map = moves_to_map(moves, canonical_stride, signal_len, reserve);
    // if (m_is_rna_model) {
    //     assert(signal_len % canonical_stride == 0);
    //     utils::reverse_seq_to_sig_map(seq_to_sig_map, signal_len);
    // }
    return seq_to_sig_map;
}

inline int base_to_int(char c) { return 0b11 & ((c >> 2) ^ (c >> 1)); }

std::vector<int> sequence_to_ints(const std::string& sequence) {
    std::vector<int> sequence_ints;
    sequence_ints.reserve(sequence.size());
    std::transform(std::begin(sequence), std::end(sequence), std::back_inserter(sequence_ints), &base_to_int);
    return sequence_ints;
}


void populate_hits_sig(
    std::array<std::vector<int64_t>, 4>& per_base_hits_sig,
    std::array<std::vector<int64_t>, 4>& per_base_hits_seq,
    std::vector<uint64_t>& seq_to_sig_map,
    const int base_id
) {
    // todo:
    // for (size_t model_id = 0; model_id < runner->num_models(); ++model_id) {
        
    // }
    const auto& hits_seq = per_base_hits_seq.at(base_id);
    auto& hits_sig = per_base_hits_sig.at(base_id);

    hits_sig.resize(hits_seq.size());
    LOG_TRACE("hits_sig size: %zu, hits_seq size: %zu", hits_sig.size(), hits_seq.size());
    LOG_TRACE("seq_to_sig_map size: %zu", seq_to_sig_map.size());
    for (size_t i = 0; i < hits_seq.size(); ++i) {
        if (hits_seq[i] >= seq_to_sig_map.size()) {
            ERROR("index: %zu, seq pos: %zu", i, hits_seq[i]);
            exit(1);
        }
        hits_sig[i] = seq_to_sig_map.at(hits_seq[i]);
    }
}

// Adapted from https://stackoverflow.com/questions/11964552/finding-quartiles
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value, T>::type>
inline std::vector<T> quantiles(const std::vector<T>& in_data, const std::vector<T>& quants) {
    if (in_data.empty()) {
        return {};
    }

    if (in_data.size() == 1) {
        return {in_data.front()};
    }

    auto data = in_data;
    std::sort(std::begin(data), std::end(data));
    std::vector<T> quantiles;
    quantiles.reserve(quants.size());

    auto linear_interp = [](T v0, T v1, T t) { return (1 - t) * v0 + t * v1; };

    for (size_t i = 0; i < quants.size(); ++i) {
        T pos = linear_interp(0, T(data.size() - 1), quants[i]);

        int64_t left = std::max(int64_t(std::floor(pos)), int64_t(0));
        int64_t right = std::min(int64_t(std::ceil(pos)), int64_t(data.size() - 1));
        T data_left = data.at(left);
        T data_right = data.at(right);

        T quantile = linear_interp(data_left, data_right, pos - left);
        quantiles.push_back(quantile);
    }

    return quantiles;
}

// Perform a least-squares linear regression of the form y = mx + b, solving for m and b.
// Returns a tuple {m, b, r} where r is the regression correlation coefficient
// Adapted from https://stackoverflow.com/questions/5083465/fast-efficient-least-squares-fit-algorithm-in-c
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value, T>::type>
std::tuple<T, T, T> linear_regression(const std::vector<T>& x, const std::vector<T>& y) {
    assert(x.size() == y.size());
    auto sum_square = [](auto s2, auto q) { return s2 + q * q; };

    T sumx2 = std::accumulate(std::begin(x), std::end(x), T(0), sum_square);
    T sumy2 = std::accumulate(std::begin(y), std::end(y), T(0), sum_square);
    T sumx = std::accumulate(std::begin(x), std::end(x), T(0));
    T sumy = std::accumulate(std::begin(y), std::end(y), T(0));

    T sumxy = 0.0;
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        sumxy += x[i] * y[i];
    }

    T denom = (n * sumx2 - (sumx * sumx));
    if (denom == 0) {
        // singular matrix. can't solve the problem, return identity transform
        return std::make_tuple(T(1), T(0), T(0));
    }

    T m = (n * sumxy - sumx * sumy) / denom;
    T b = (sumy * sumx2 - sumx * sumxy) / denom;
    // compute correlation coeff
    T r = (sumxy - sumx * sumy / n) /
          std::sqrt((sumx2 - (sumx * sumx) / n) * (sumy2 - (sumy * sumy) / n));

    return std::make_tuple(m, b, r);
}

std::pair<float, float> calc_offset_scale(
    core_t *core,
    const at::Tensor& samples,
    const std::vector<uint64_t>& seq_to_sig_map,
    const std::vector<float>& levels,
    size_t clip_bases,
    size_t max_bases
) {
    auto kmer_levels = core->modbase_config->mods.kmer_levels;
    if (kmer_levels.empty()) {
        return std::make_pair(0.f, 1.f);
    }

    auto n = std::min({seq_to_sig_map.size() - 1, max_bases});

    std::vector<float> optim_dacs(n, 0.f);
    std::vector<float> new_levels(n, 0.f);

    assert(samples.is_contiguous());
    assert(samples.dtype() == at::kHalf);
    using SignalType = c10::Half;
    SignalType* samples_ptr = samples.data_ptr<SignalType>();
    // get the mid-point of the base
    for (size_t i = 0; i < n; i++) {
        int pos = int((seq_to_sig_map[i] + seq_to_sig_map[i + 1]) / 2);
        optim_dacs[i] = static_cast<float>(samples_ptr[pos]);
        new_levels[i] = levels[i];
    }

    if (clip_bases > 0 && levels.size() > clip_bases * 2) {
        new_levels = {std::begin(new_levels) + clip_bases, std::end(new_levels) - clip_bases};
        optim_dacs = {std::begin(optim_dacs) + clip_bases, std::end(optim_dacs) - clip_bases};
    }

    std::vector<float> quants(19);
    std::generate(std::begin(quants), std::end(quants), [i = 0.f]() mutable { return i += 0.05f; });

    new_levels = quantiles(new_levels, quants);
    optim_dacs = quantiles(optim_dacs, quants);

    const auto result = linear_regression(optim_dacs, new_levels);
    float new_scale  = std::get<0>(result);
    float new_offset = std::get<1>(result);
    float rcoeff     = std::get<2>(result);
    return std::make_pair(new_offset, new_scale);
}

size_t index_from_int_kmer(const int* int_kmer_start, size_t kmer_len) {
    size_t index = 0;
    for (int kmer_pos = 0; kmer_pos < static_cast<int>(kmer_len); ++kmer_pos) {
        index += *(int_kmer_start + kmer_len - kmer_pos - 1) * (1 << (2 * kmer_pos));
    }
    return index;
}

std::vector<float> extract_levels(core_t *core, const std::vector<int>& int_seq) {
    size_t kmer_len = core->modbase_config->context.kmer_len;
    auto center_idx = core->modbase_config->refine.center_idx;
    auto kmer_levels = core->modbase_config->mods.kmer_levels;

    std::vector<float> levels(int_seq.size(), 0.f);

    if (int_seq.size() < kmer_len) {
        return levels;
    }

    auto int_kmer_start_ptr = int_seq.data();
    auto levels_ptr = levels.data() + center_idx;
    for (size_t pos = 0; pos < int_seq.size() - kmer_len;
         ++pos, ++int_kmer_start_ptr, ++levels_ptr) {
        *(levels_ptr) = kmer_levels[index_from_int_kmer(int_kmer_start_ptr, kmer_len)];
    }
    return levels;
}

void scale_signal_modbase(
    core_t *core,
    at::Tensor& signal,
    const std::vector<int>& seq_ints,
    const std::vector<uint64_t>& seq_to_sig_map
) {
    auto levels = extract_levels(core, seq_ints);

    // generate the signal values at the centre of each base, create the nx5% quantiles (sorted)
    // and perform a linear regression against the expected kmer levels to generate a new shift and scale
    const auto result = calc_offset_scale(core, signal, seq_to_sig_map, levels, 10, 1000);
    float offset = result.first;
    float scale = result.second;
    signal = signal * scale + offset;
}

void populate_signal(
    core_t *core,
    at::Tensor& signal,
    std::vector<uint64_t>& seq_to_sig_map,
    // const at::Tensor& raw_data,
    const std::vector<int>& int_seq
) {
    // if (m_is_rna_model) {
    //     // Reverse the RNA signal and prepend a short mirrored slice of padding to ensure moves are
    //     // stride aligned.
    //     const int64_t len = raw_data.size(0);
    //     const int64_t padding = utils::pad_to(len, m_canonical_stride) - len;

    //     at::Tensor sig = at::empty({len + padding}, raw_data.options());
    //     at::Tensor body = sig.slice(0, padding, len + padding);
    //     at::flip_out(body, raw_data, 0);

    //     sig.slice(0, 0, padding) = raw_data.slice(0, len - padding, len);

    //     signal = runner->scale_signal(0, sig, int_seq, seq_to_sig_map);
    //     return;
    // }

    scale_signal_modbase(core, signal, int_seq, seq_to_sig_map);
    return;
}

std_optional<int64_t> get_next_hit(const std::vector<int64_t>& hit_sig_idxs, const int64_t chunk_signal_start) {
    // Check for the first element explicitly
    if (!hit_sig_idxs.empty() && hit_sig_idxs.front() >= chunk_signal_start) {
        return 0;
    }

    // The first context hit signal index at or after `chunk_signal_start`
    const auto next_hit = std::lower_bound(hit_sig_idxs.begin(), hit_sig_idxs.end(), chunk_signal_start);

    if (next_hit != hit_sig_idxs.cend()) {
        return std::distance(hit_sig_idxs.cbegin(), next_hit);
    }

    // Did not find a context hit in this chunk or any remaining chunk
    return STD_NULLOPT;
}

size_t create_mod_chunks(std::vector<mod_chunk_t> &chunks, core_t *core, read_dat_t *read_dat) {
    ASSERT(chunks.size() == 0);
    // todo:
    // for (size_t model_id = 0; model_id < runner->num_models(); ++model_id) {
    const auto model_id = 0;

    ModBaseModelConfig *config = core->modbase_config;
    const int base_id = config->mods.base_id;

    const std::vector<int64_t>& hits_to_sig = read_dat->per_base_hits_sig[base_id];

    const auto num_states = config->mods.count + 1;
    ContextParams ctx = config->context;
    const auto signal_len = read_dat->scaled_signal.size(0);

    // auto end_align_last_chunk = false;

    const int64_t chunk_size = ctx.chunk_size;
    const int64_t context_samples_before = ctx.samples_before;
    const int64_t context_samples_after = ctx.samples_after;

    int64_t chunk_st = 0;
    while (chunk_st < signal_len) {
        std_optional<int64_t> next_hit = get_next_hit(hits_to_sig, chunk_st);

        if (!next_hit) {
            break;
        }

        const int64_t hit_idx = next_hit.value();
        const int64_t hit_sig = hits_to_sig.at(hit_idx);

        // Add context samples as a lead-in
        chunk_st = hit_sig - context_samples_before;
        // If there's no lead-in context start at the first sample
        chunk_st = chunk_st > 0 ? chunk_st : 0;

        chunks.push_back(mod_chunk_t {
            read_dat,
            int(model_id),
            base_id,
            chunk_st,
            hit_idx,
            num_states,
            std::vector<float>()
        });

        // Step chunk forward. Ensure hits with incomplete downstream context are not skipped
        chunk_st += chunk_size - context_samples_after + 1;
        // Always move forward if chunk_size = before+after
        if (chunk_st <= hit_sig) {
            chunk_st = hit_sig + 1;
        }
    }

    // if (chunks.size() > 1 && end_align_last_chunk) {
    //     const int64_t last_hit = hits_to_sig.back();
    //     const int64_t aligned_chunk_st = last_hit + context_samples_after - chunk_size;
    //     if (aligned_chunk_st > 0) {
    //         chunks.back().first = aligned_chunk_st;
    //     }
    // }
    
    return chunks.size();
}

std::vector<std::pair<uint64_t, uint64_t>> merge_chunks(
    const std::vector<mod_chunk_t>& chunks_by_caller,
    const std::vector<uint64_t>& chunk_sizes
) {
    using Item = std::tuple<uint64_t, size_t, size_t>;

    // Sort the minHeap by chunk_size
    auto cmp = [](const Item& a, const Item& b) { return std::get<0>(a) > std::get<0>(b); };
    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> minHeap(cmp);

    size_t max_chunks = 0;
    int model_id = 0;
    // Initialize heap with the first element of each non‐empty list
    // for (size_t model_id = 0; model_id < chunk_sizes.size(); ++model_id) {
    //     if (!chunks_by_caller[model_id].empty()) {
    //         minHeap.emplace(chunks_by_caller[model_id][0]->signal_start, model_id, 0);
    //         max_chunks += chunks_by_caller[model_id].size();
    //     }
    // }

    if (!chunks_by_caller.empty()) {
        minHeap.emplace(chunks_by_caller[0].signal_offset, model_id, 0);
        max_chunks += chunks_by_caller.size();
    }

    // Contiguous intervals from all models and chunks.
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    merged.reserve(max_chunks);

    // Push new interval - if it overlaps with previous interval merge it.
    // Because the minHeap is sorted by the chunk_start we can do this in one pass.
    auto push_interval = [&](const uint64_t start, const uint64_t end) {
        if (merged.empty() || start > merged.back().second) {
            // Add new interval
            merged.emplace_back(start, end);
        } else {
            // Extend the interval
            merged.back().second = std::max(merged.back().second, end);
        }
    };

    // Do all the work in the heap.
    while (!minHeap.empty()) {
        const auto top = minHeap.top();
        const int chunk_start  = std::get<0>(top);
        const int model_id     = std::get<1>(top);
        const int chunk_index  = std::get<2>(top);
        minHeap.pop();

        // Add / merge this interval
        push_interval(chunk_start, chunk_start + chunk_sizes.at(model_id));

        // Add the next chunk from this model if any
        const size_t next_index = chunk_index + 1;
        // if (next_index < chunks_by_caller[model_id].size()) {
        //     minHeap.emplace(chunks_by_caller[model_id][next_index]->signal_start, model_id,
        //                     next_index);
        // }

        if (next_index < chunks_by_caller.size()) {
            minHeap.emplace(chunks_by_caller[next_index].signal_offset, model_id,
                            next_index);
        }
    }

    return merged;
}

std::vector<bool> get_skip_positions(
    const std::vector<uint64_t>& seq_to_sig_map,
    const std::vector<std::pair<uint64_t, uint64_t>>& merged_chunks
) {
    if (seq_to_sig_map.empty() || merged_chunks.empty()) {
        return {};
    }
    std::vector<bool> skips(seq_to_sig_map.size(), true);

    // const size_t N = seq_to_sig_map.size();
    for (const auto& chunk : merged_chunks) {
        auto start = chunk.first;
        auto end   = chunk.second;

        // Find left edge where signal_pos >= start
        auto it_left = std::lower_bound(seq_to_sig_map.begin(), seq_to_sig_map.end(), start);
        if (it_left == seq_to_sig_map.end()) {
            // all positions are less than start → no overlap
            continue;
        }
        size_t left = it_left - seq_to_sig_map.begin();

        // Find right edge where signal_pos <= end
        auto it_right = std::upper_bound(seq_to_sig_map.begin(), seq_to_sig_map.end(), end);
        if (it_right == seq_to_sig_map.begin()) {
            // all positions are greater than end → no overlap
            continue;
        }
        size_t right = (it_right - seq_to_sig_map.begin()) - 1;

        // if the interval is empty, continue:
        if (left > right) {
            continue;
        }

        // Mark all internal positions of chunk
        for (size_t i = left; i <= right; ++i) {
            skips[i] = false;
        }

        // Include the left neighbour - right is ignored because encoding is left -> right
        if (left > 0 && start > seq_to_sig_map[left - 1] && start < seq_to_sig_map[left]) {
            skips[left - 1] = false;
        }
    }

    return skips;
}

std::vector<bool> get_minimal_encoding_skips(
    core_t *core,
    const std::vector<mod_chunk_t>& chunks_by_caller,
    const std::vector<uint64_t>& seq_to_sig_map,
    std::vector<int> &int_seq
) {
    // if (!m_minimal_encode) {
    //     return {};
    // }
    std::vector<uint64_t> chunk_sizes;

    auto num_models = 1; // todo
    chunk_sizes.reserve(num_models);
    for (auto model_id = 0; model_id < num_models; model_id++) {
        chunk_sizes.push_back(static_cast<uint64_t>(core->modbase_config->context.chunk_size));
    }

    // Get contiguous intervals from all models and chunks.
    const auto merged_chunks = merge_chunks(chunks_by_caller, chunk_sizes);
    if (merged_chunks.empty()) {
        ERROR("%s", "Failed to merge modbase chunks");
        exit(1);
    }

    return get_skip_positions(seq_to_sig_map, merged_chunks);
}

// OneHot encoding encodes categorical data (bases) into unique bool-like columns for each category.
// This function returns a u32 representing the 4 i8 base categories and a 5th for N.
// -1(N)[0,0,0,0]; 0(A)[1,0,0,0]; 1(C)[0,1,0,0]; 2(G)[0,0,1,0] 3(T)[0,0,0,1].
// The encoding is done by bit shifting a 1 by the numerical "magnitude" of the base giving:
// Note: Bytes [ABCD] becomes [D,C,B,A] in LE systems which is why T is [0,0,0,1] not [1,0,0,0].
inline uint32_t encode(int base) { return base == -1 ? uint32_t{0} : (uint32_t{1} << (base << 3)); }

// Write the kmer encoding into `output_ptr` whose size must be at least: `kmer_len * 4 * context_samples`
inline void encode_kmer_generic(
    int8_t* output_ptr,
    const std::vector<int>& seq,
    const std::vector<uint64_t>& seq_mappings,
    const std::vector<bool>& base_skips,
    size_t context_seq_len,
    size_t kmer_len
) {
    const size_t seq_len = std::min(seq.size(), context_seq_len);
    for (size_t s = 0; s < seq_len; ++s) {
        const size_t count = seq_mappings[s + 1] - seq_mappings[s];
        if (!base_skips.empty() && base_skips[s]) {
            output_ptr += kmer_len * count * sizeof(uint32_t);  // skip k-mer * 4-bytes * count;
            continue;
        }

        for (size_t b = 0; b < count; ++b) {
            for (size_t k = 0; k < kmer_len; ++k) {
                const size_t seq_idx = s + k;
                assert(seq_idx < seq.size());
                uint32_t base_onehot = encode(seq[seq_idx]);
                // memcpy will be translated to a single 32 bit write.
                std::memcpy(output_ptr, &base_onehot, sizeof(base_onehot));
                output_ptr += sizeof(base_onehot);
            }
        }
    }
}

inline std::vector<int8_t> encode_kmer_chunk_generic(
    const std::vector<int>& seq,
    const std::vector<uint64_t>& seq_mappings,
    const std::vector<bool>& base_skips,
    size_t kmer_len,
    size_t context_samples,
    size_t padding_samples,
    bool kmer_centered
) {
    // Given sequence: ACGTAC
    // Uncentered 7mer: [ACGTACnnnnn] -> ACGTACn CGTACnn GTACnnn TACnnnn ACnnnnn Cnnnnnn
    // Centered 7mer:   [nnnACGTACnnn]-> nnnACGT nnACGTA nACGTAC ACGTACn CGTACnn GTACnnn
    // Extend the sequence with N bases but do not change the mapping so the signal alignment
    // remains unchanged. Offset the copy by start_pos to center the kmer.
    const size_t start_pos = kmer_centered ? kmer_len / 2 : 0;
    std::vector<int> ext_seq(seq.size() + kmer_len - 1, -1);
    std::copy(seq.begin(), seq.end(), ext_seq.begin() + start_pos);
    
    const size_t kmer_bytes = kmer_len * NUM_BASES;
    const size_t total_samples = context_samples + (2 * padding_samples);
    const size_t output_size = kmer_bytes * total_samples;
    const size_t padded_start = kmer_bytes * padding_samples;

    std::vector<int8_t> output(output_size, 0);
    int8_t* output_ptr = &output[padded_start];

    encode_kmer_generic(output_ptr, ext_seq, seq_mappings, base_skips, seq.size(), kmer_len);
    return output;
}

std::vector<int8_t> encode_kmer_chunk(
    const std::vector<int>& seq,
    const std::vector<uint64_t>& seq_mappings,
    const std::vector<bool>& base_skips,
    size_t kmer_len,
    size_t context_samples,
    size_t padding_samples,
    bool kmer_centered
) {
    // if (kmer_len == 9) {
    //     return encode_kmer_chunk_len9(seq, seq_mappings, base_skips, context_samples, padding_samples, kmer_centered);
    // }
    return encode_kmer_chunk_generic(seq, seq_mappings, base_skips, kmer_len, context_samples, padding_samples, kmer_centered);
}

void populate_encoded_kmer(
    std::vector<int8_t>& encoded_kmer,
    const std::size_t signal_len,
    const std::vector<int>& int_seq,
    const std::vector<uint64_t>& seq_to_sig_map,
    const std::vector<bool>& base_skips,
    int kmer_len,
    int sequence_stride_ratio
) {
    // if (m_sequence_stride_ratio == 1) {
    //     encoded_kmer = encode_kmer_chunk(int_seq, seq_to_sig_map, base_skips, m_kmer_len, signal_len, 0, true);
    //     return;
    // }

    // v3 lstm stuff
    // Dividing the signal values by the stride ratio results in a downsampled encoded kmer
    std::vector<std::uint64_t> strided_s2s;
    strided_s2s.reserve(seq_to_sig_map.size());
    std::transform(
            seq_to_sig_map.cbegin(), seq_to_sig_map.cend(), std::back_inserter(strided_s2s),
            [ssr = sequence_stride_ratio](const std::uint64_t value) { return value / ssr; });

    encoded_kmer = encode_kmer_chunk(int_seq, strided_s2s, base_skips, kmer_len, signal_len / sequence_stride_ratio, 0, true);
}

void preprocess_modbase(core_t *core, db_t *db, int32_t i) {
    slow5_rec_t *rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;

    (*db->mod_chunks)[i].clear();
    if (len_raw_signal > 0) {
        read_dat_t *read_dat = (*db->read_dats)[i];

        char *seq = (*db->sequence)[i];
        read_dat->seq = seq;

        std::vector<uint8_t> &moves = (*db->moves)[i];

        LOG_TRACE("%s", "tensor_from_record");

        read_dat->scaled_signal = tensor_from_record(rec).to(torch::kFloat16);
        
        LOG_TRACE("%s", "initialise_base_mod_probs");

        initialise_base_mod_probs(core, read_dat, seq);

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

        create_mod_chunks((*db->mod_chunks)[i], core, read_dat);

        const auto base_skips = get_minimal_encoding_skips(core, (*db->mod_chunks)[i], seq_to_sig_map, int_seq);

        auto sequence_stride_ratio = core->modbase_config->general.stride_ratio();
        auto kmer_len = core->modbase_config->context.bases_before + core->modbase_config->context.bases_after + 1;
        populate_encoded_kmer(read_dat->encoded_kmers, read_dat->scaled_signal.size(0), int_seq, seq_to_sig_map, base_skips, kmer_len, sequence_stride_ratio);

        const std::size_t enc_kmer_size = read_dat->encoded_kmers.size();
        const std::size_t expected = (read_dat->scaled_signal.size(0) / sequence_stride_ratio) * kmer_len * NUM_BASES;
        if (enc_kmer_size != expected) {
            ERROR("%s", "Modbase kmer encoding failed");
            exit(1);
        }
    }
}

static bool validate_bam_tag_code(const std::string& bam_name) {
    // Check the supplied bam_name is a single character
    if (bam_name.size() == 1 && std::isalpha(static_cast<unsigned char>(bam_name[0]))) {
        return true;
    }

    // Check the supplied bam_name is a simple integer and if so, assume it's a CHEBI code.
    if (std::all_of(bam_name.begin(), bam_name.end(),
                    [](const char& c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
        return true;
    }
    return false;
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

    (*db->mod_string)[i] = strdup(modbase_string.c_str());
    assert((*db->mod_string)[i] != NULL);

    (*db->mod_prob)[i] = std::move(modbase_prob);
}