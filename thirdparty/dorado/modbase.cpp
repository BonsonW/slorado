#include "dorado/modbase.h"

#include "dorado/simd.h"
#include "error.h"
#include "misc.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <queue>
#include <tuple>
#include <type_traits>

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

void initialise_base_mod_probs(const core_t *core, read_dat_t *read_dat, char *seq) {
	auto num_states = NUM_BASES + core->modbase_config->mods.count;
	auto seqlen = strlen(seq);
	read_dat->base_mod_probs.assign(seqlen * num_states, 0);
	for (size_t i = 0; i < seqlen; ++i) {
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
	read_dat->base_mod_simplex_motif_hits.assign(seqlen, false);
}

std::vector<size_t> get_motif_hits(char *seq, std::string motif, size_t motif_offset) {
	return modbase_motif_hits(motif, motif_offset, seq, strlen(seq));
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
	const auto& hits_seq = per_base_hits_seq.at(base_id);
	auto& hits_sig = per_base_hits_sig.at(base_id);

	hits_sig.resize(hits_seq.size());
	LOG_TRACE("hits_sig size: %zu, hits_seq size: %zu", hits_sig.size(), hits_seq.size());
	LOG_TRACE("seq_to_sig_map size: %zu", seq_to_sig_map.size());
	for (size_t i = 0; i < hits_seq.size(); ++i) {
		if (hits_seq[i] < 0 || static_cast<size_t>(hits_seq[i]) >= seq_to_sig_map.size()) {
			ERROR("index: %zu, seq pos: %zu", i, hits_seq[i]);
			exit(1);
		}
		hits_sig[i] = seq_to_sig_map.at(hits_seq[i]);
	}
}

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
		return std::make_tuple(T(1), T(0), T(0));
	}

	T m = (n * sumxy - sumx * sumy) / denom;
	T b = (sumy * sumx2 - sumx * sumxy) / denom;
	T r = (sumxy - sumx * sumy / n) /
		  std::sqrt((sumx2 - (sumx * sumx) / n) * (sumy2 - (sumy * sumy) / n));

	return std::make_tuple(m, b, r);
}

std::pair<float, float> calc_offset_scale_i16(
	core_t *core,
	const at::Tensor& samples,
	const std::vector<uint64_t>& seq_to_sig_map,
	const std::vector<float>& levels,
	size_t clip_bases,
	size_t max_bases
) {
	const auto& kmer_levels = core->modbase_config->mods.kmer_levels;
	if (kmer_levels.empty()) {
		return std::make_pair(0.f, 1.f);
	}

	auto n = std::min({seq_to_sig_map.size() - 1, max_bases});
	std::vector<float> optim_dacs(n, 0.f);
	std::vector<float> new_levels(n, 0.f);

	assert(samples.is_contiguous());
	assert(samples.dtype() == at::kShort);
	using SignalType = int16_t;
	SignalType* samples_ptr = samples.data_ptr<SignalType>();

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
	const auto& kmer_levels = core->modbase_config->mods.kmer_levels;

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

	const auto result = calc_offset_scale_i16(core, signal, seq_to_sig_map, levels, 10, 1000);

	float shift = result.first;
	float scale = result.second;

	scale_shift_tensor_i16_to_f16_inplace_impl(signal, shift, scale);
}

void populate_signal(
	core_t *core,
	at::Tensor& signal,
	std::vector<uint64_t>& seq_to_sig_map,
	const std::vector<int>& int_seq
) {
	scale_signal_modbase(core, signal, int_seq, seq_to_sig_map);
}

std_optional<int64_t> get_next_hit(const std::vector<int64_t>& hit_sig_idxs, const int64_t chunk_signal_start) {
	if (!hit_sig_idxs.empty() && hit_sig_idxs.front() >= chunk_signal_start) {
		return 0;
	}

	const auto next_hit = std::lower_bound(hit_sig_idxs.begin(), hit_sig_idxs.end(), chunk_signal_start);

	if (next_hit != hit_sig_idxs.cend()) {
		return std::distance(hit_sig_idxs.cbegin(), next_hit);
	}

	return STD_NULLOPT;
}

size_t create_mod_chunks(std::vector<mod_chunk_t> &chunks, core_t *core, read_dat_t *read_dat) {
	ASSERT(chunks.size() == 0);
	const auto model_id = 0;

	modbase_model_config_t *config = core->modbase_config;
	const int base_id = config->mods.base_id;

	const std::vector<int64_t>& hits_to_sig = read_dat->per_base_hits_sig[base_id];

	const int64_t num_states = static_cast<int64_t>(config->mods.count + 1);
	context_params_t ctx = config->context;
	const size_t signal_len = static_cast<size_t>(read_dat->scaled_signal.size(0));

	const size_t chunk_size = static_cast<size_t>(ctx.chunk_size);
	const size_t context_samples_before = static_cast<size_t>(ctx.samples_before);
	const size_t context_samples_after = static_cast<size_t>(ctx.samples_after);

	size_t chunk_st = 0;
	while (chunk_st < signal_len) {
		std_optional<int64_t> next_hit = get_next_hit(hits_to_sig, static_cast<int64_t>(chunk_st));

		if (!next_hit) {
			break;
		}

		const size_t hit_idx = static_cast<size_t>(next_hit.value());
		const size_t hit_sig = static_cast<size_t>(hits_to_sig.at(hit_idx));

		chunk_st = (hit_sig > context_samples_before) ? (hit_sig - context_samples_before) : 0;
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

		chunk_st += chunk_size - context_samples_after + 1;
		if (chunk_st <= hit_sig) {
			chunk_st = hit_sig + 1;
		}
	}

	return chunks.size();
}

std::vector<std::pair<uint64_t, uint64_t>> merge_chunks(
	const std::vector<mod_chunk_t>& chunks_by_caller,
	const std::vector<uint64_t>& chunk_sizes
) {
	using Item = std::tuple<uint64_t, size_t, size_t>;

	auto cmp = [](const Item& a, const Item& b) { return std::get<0>(a) > std::get<0>(b); };
	std::priority_queue<Item, std::vector<Item>, decltype(cmp)> minHeap(cmp);

	size_t max_chunks = 0;
	int model_id = 0;

	if (!chunks_by_caller.empty()) {
		minHeap.emplace(chunks_by_caller[0].signal_offset, model_id, 0);
		max_chunks += chunks_by_caller.size();
	}

	std::vector<std::pair<uint64_t, uint64_t>> merged;
	merged.reserve(max_chunks);

	auto push_interval = [&](const uint64_t start, const uint64_t end) {
		if (merged.empty() || start > merged.back().second) {
			merged.emplace_back(start, end);
		} else {
			merged.back().second = std::max(merged.back().second, end);
		}
	};

	while (!minHeap.empty()) {
		const auto top = minHeap.top();
		const int chunk_start  = std::get<0>(top);
		const int model_id     = std::get<1>(top);
		const int chunk_index  = std::get<2>(top);
		minHeap.pop();

		push_interval(chunk_start, chunk_start + chunk_sizes.at(model_id));

		const size_t next_index = chunk_index + 1;

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

	for (const auto& chunk : merged_chunks) {
		auto start = chunk.first;
		auto end   = chunk.second;

		auto it_left = std::lower_bound(seq_to_sig_map.begin(), seq_to_sig_map.end(), start);
		if (it_left == seq_to_sig_map.end()) {
			continue;
		}
		size_t left = it_left - seq_to_sig_map.begin();

		auto it_right = std::upper_bound(seq_to_sig_map.begin(), seq_to_sig_map.end(), end);
		if (it_right == seq_to_sig_map.begin()) {
			continue;
		}
		size_t right = (it_right - seq_to_sig_map.begin()) - 1;

		if (left > right) {
			continue;
		}

		for (size_t i = left; i <= right; ++i) {
			skips[i] = false;
		}

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
	std::vector<uint64_t> chunk_sizes;

	auto num_models = 1;
	chunk_sizes.reserve(num_models);
	for (auto model_id = 0; model_id < num_models; model_id++) {
		chunk_sizes.push_back(static_cast<uint64_t>(core->modbase_config->context.chunk_size));
	}

	const auto merged_chunks = merge_chunks(chunks_by_caller, chunk_sizes);
	if (merged_chunks.empty()) {
		ERROR("%s", "Failed to merge modbase chunks");
		exit(1);
	}

	return get_skip_positions(seq_to_sig_map, merged_chunks);
}

inline uint32_t encode(int base) { return base == -1 ? uint32_t{0} : (uint32_t{1} << (base << 3)); }

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
			output_ptr += kmer_len * count * sizeof(uint32_t);
			continue;
		}

		for (size_t b = 0; b < count; ++b) {
			for (size_t k = 0; k < kmer_len; ++k) {
				const size_t seq_idx = s + k;
				assert(seq_idx < seq.size());
				uint32_t base_onehot = encode(seq[seq_idx]);
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

#if ENABLE_AVX2_IMPL
[[maybe_unused]] __attribute__((target("avx2"))) void avx2_encode_kmer_len9(
		uint8_t* output_t_ptr,
		const std::vector<int>& seq,
		const std::vector<uint64_t>& seq_mappings,
		const std::vector<bool>& base_skips,
		size_t seq_len) {
	const __m256i kOnes = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);

	const __m256i kRotate1 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
	const __m256i kRotate2 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
	const __m256i kRotate3 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);

	for (size_t seq_pos = 0; seq_pos < seq_len; ++seq_pos) {
		const int count = seq_mappings[seq_pos + 1] - seq_mappings[seq_pos];

		if (!base_skips.empty() && base_skips[seq_pos]) {
			output_t_ptr += 36 * count;
			continue;
		}

		const __m256i bases_01234567 =
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&seq[seq_pos]));
		const __m256i bases_12345678 =
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&seq[seq_pos + 1]));

		const __m256i shifts_01234567 = _mm256_slli_epi32(bases_01234567, 3);
		const __m256i bases_01234567_oh = _mm256_sllv_epi32(kOnes, shifts_01234567);
		const __m256i shifts_12345678 = _mm256_slli_epi32(bases_12345678, 3);
		const __m256i bases_12345678_oh = _mm256_sllv_epi32(kOnes, shifts_12345678);

		const __m256i bases_70123456_oh = _mm256_permutevar8x32_epi32(bases_01234567_oh, kRotate1);
		const __m256i bases_81234567_oh = _mm256_permutevar8x32_epi32(bases_12345678_oh, kRotate1);
		const __m256i bases_80123456_oh =
				_mm256_blend_epi32(bases_70123456_oh, bases_81234567_oh, 0x1);

		const __m256i bases_67012345_oh = _mm256_permutevar8x32_epi32(bases_01234567_oh, kRotate2);
		const __m256i bases_78123456_oh = _mm256_permutevar8x32_epi32(bases_12345678_oh, kRotate2);
		const __m256i bases_78012345_oh =
				_mm256_blend_epi32(bases_67012345_oh, bases_78123456_oh, 0x3);

		const __m256i bases_56701234_oh = _mm256_permutevar8x32_epi32(bases_01234567_oh, kRotate3);
		const __m256i bases_67812345_oh = _mm256_permutevar8x32_epi32(bases_12345678_oh, kRotate3);
		const __m256i bases_67801234_oh =
				_mm256_blend_epi32(bases_56701234_oh, bases_67812345_oh, 0x7);

		const __m128i bases_5678_oh = _mm256_extracti128_si256(bases_12345678_oh, 1);

		for (int i = 0; i < count / 4; ++i) {
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr + 0), bases_01234567_oh);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr + 32), bases_80123456_oh);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr + 64), bases_78012345_oh);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr + 96), bases_67801234_oh);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(output_t_ptr + 128), bases_5678_oh);
			output_t_ptr += 144;
		}

		const int remaining_count = count % 4;
		const std::uint32_t base8_oh = _mm256_extract_epi32(bases_12345678_oh, 7);
		for (int i = 0; i < remaining_count; ++i) {
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(output_t_ptr), bases_01234567_oh);
			std::memcpy(output_t_ptr + 32, &base8_oh, sizeof(base8_oh));
			output_t_ptr += 36;
		}
	}
}
#endif

#if ENABLE_AVX2_IMPL
[[maybe_unused]] __attribute__((target("avx2"))) std::vector<int8_t> encode_kmer_chunk_len9(
		const std::vector<int>& seq,
		const std::vector<uint64_t>& seq_mappings,
		const std::vector<bool>& base_skips,
		size_t context_samples,
		size_t padding_samples,
		bool kmer_centered) {
	constexpr int kKmerLen = 9;
	constexpr int kNumBases = 4;

	const size_t start_pos = kmer_centered ? kKmerLen / 2 : 0;
	std::vector<int> ext_seq(seq.size() + kKmerLen - 1, -1);
	std::copy(seq.begin(), seq.end(), ext_seq.begin() + start_pos);

	constexpr int kKmerBytes = kKmerLen * kNumBases;
	const size_t total_samples = context_samples + (2 * padding_samples);
	const size_t output_size = kKmerBytes * total_samples;
	const size_t padded_start = kKmerBytes * padding_samples;
	std::vector<int8_t> output_t(output_size);
	uint8_t* output_t_ptr = reinterpret_cast<uint8_t*>(&output_t[padded_start]);

	avx2_encode_kmer_len9(output_t_ptr, ext_seq, seq_mappings, base_skips, seq.size());
	return output_t;
}
#endif

#if ENABLE_AVX2_IMPL
[[maybe_unused]] __attribute__((target("default")))
#endif
std::vector<int8_t>
encode_kmer_chunk_len9(const std::vector<int>& seq,
					   const std::vector<uint64_t>& seq_mappings,
					   const std::vector<bool>& base_skips,
					   size_t context_samples,
					   size_t padding_samples,
					   bool kmer_centered) {
	constexpr size_t kKmerLen = 9;
	return encode_kmer_chunk_generic(seq, seq_mappings, base_skips, kKmerLen, context_samples,
									 padding_samples, kmer_centered);
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
	if (sequence_stride_ratio == 1) {
		encoded_kmer = encode_kmer_chunk(int_seq, seq_to_sig_map, base_skips, kmer_len, signal_len, 0, true);
		return;
	}

	std::vector<std::uint64_t> strided_s2s;
	strided_s2s.reserve(seq_to_sig_map.size());
	std::transform(
			seq_to_sig_map.cbegin(), seq_to_sig_map.cend(), std::back_inserter(strided_s2s),
			[ssr = sequence_stride_ratio](const std::uint64_t value) { return value / ssr; });

	encoded_kmer = encode_kmer_chunk(int_seq, strided_s2s, base_skips, kmer_len, signal_len / sequence_stride_ratio, 0, true);
}

bool validate_bam_tag_code(const std::string& bam_name) {
	if (bam_name.size() == 1 && std::isalpha(static_cast<unsigned char>(bam_name[0]))) {
		return true;
	}

	if (std::all_of(bam_name.begin(), bam_name.end(),
					[](const char& c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
		return true;
	}
	return false;
}

static int64_t resolve_score_index(
    const int64_t hit_sig_abs,
    const int64_t chunk_signal_start,
    const int64_t scores_states,
    const int64_t chunk_size,
    const int64_t context_samples_before,
    const int64_t context_samples_after,
    const int64_t modbase_stride
) {
    if (hit_sig_abs < chunk_signal_start) {
        ERROR("%s", "Modbase hit before chunk start.");
    }

    // Context hit chunk-relative signal index
    const int64_t hit_sig_rel = hit_sig_abs - chunk_signal_start;

    // Skip hits at end of a chunk without enough downstream context
    // It will be processed at the start if the next chunk with complete context
    if (hit_sig_rel > chunk_size - context_samples_after) {
        return -2;
    }

    // Skip hits at the start of a chunk with insufficient context
    // This hit will have been processed in a previous chunk
    // UNLESS it's the start of a read where there's no useful lead-in
    if (hit_sig_abs > context_samples_before && hit_sig_rel < context_samples_before) {
        return -1;
    }

    // We should land on a canonical base
    if (hit_sig_rel % modbase_stride != 0) {
        ERROR("%s", "Modbase score did not align to canonical base.");
    }

    // Convert chunk-relative signal-space score index into sequence-space (/stride)
    // and then into scores-space (*num_states)
    return hit_sig_rel / modbase_stride * scores_states;
}

void extract_mod_probs(
    const core_t *core,
    const mod_chunk_t *chunk,
    const c10::Half *scores_ptr,
    int64_t row_offset,
    int64_t row_size
) {
    read_dat_t *read_dat = chunk->read_dat;

    const std::vector<int64_t>& hits_seq = read_dat->per_base_hits_seq.at(chunk->base_id);
    const std::vector<int64_t>& hits_sig = read_dat->per_base_hits_sig.at(chunk->base_id);

    const auto& cfg = core->modbase_config;
    const char modbase_model_base = cfg->mods.base;
    const int64_t modbase_stride = cfg->general.stride;
    const int64_t chunk_size = cfg->context.chunk_size;
    const int64_t context_samples_before = cfg->context.samples_before;
    const int64_t context_samples_after = cfg->context.samples_after;

    // The number of states predicted by this modbase model `num_mods + 1`
    const int64_t scores_states = chunk->num_states;
    const int64_t scores_size = row_size;

    const int64_t base_offset = static_cast<int64_t>(core->modbase_info->base_probs_offsets.at(cfg->mods.base_id));
    const auto num_states = NUM_BASES + cfg->mods.count;

    for (size_t hit = chunk->hit_offset; hit < hits_sig.size(); ++hit) {
        // Context hit sequence index in the chunk sequence
        const int64_t hit_seq = hits_seq.at(hit);

        char seq = read_dat->seq[hit_seq];

        // The canonical base should be constant for a single model
        if (seq != modbase_model_base) {
            ERROR("Modbase hit base is not correct : %c", seq);
        }

        int64_t hit_score_idx = resolve_score_index(
                hits_sig.at(hit), chunk->signal_offset, scores_states, chunk_size,
                context_samples_before, context_samples_after, modbase_stride);

        if (hit_score_idx <= -2) {
            // No more hits in this chunk
            break;
        } else if (hit_score_idx == -1) {
            // This hit is skipped
            continue;
        }

        // Extract the scores for the canonical base and each of the mods in this model
        for (int64_t mod_offset = 0; mod_offset < scores_states; ++mod_offset) {
            const int64_t score_idx = hit_score_idx + mod_offset;
            if (score_idx >= scores_size) {
                ERROR("%s", "Modbase score index out of bounds.");
            }

            const int64_t row_score_idx = row_offset + score_idx;
            const float score_value = static_cast<float>(scores_ptr[row_score_idx]);
            const uint8_t score = static_cast<uint8_t>(std::min(std::floor(score_value * 256), 255.0f));

            // Index into the probabilities is calculated by
            // sequence_index * num_states := canonical "A" base probs index
            // offset then by the canonical base modification offsets
            const int64_t prob_idx = hit_seq * num_states + base_offset + mod_offset;
            read_dat->base_mod_probs.at(prob_idx) = score;
        }
    }
}
