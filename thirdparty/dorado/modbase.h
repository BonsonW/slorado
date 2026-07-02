#ifndef DORADO_MODBASE_H
#define DORADO_MODBASE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "torchbox.h"

void initialise_base_mod_probs(const core_t *core, read_dat_t *read_dat, char *seq);

bool populate_hits_seq(const core_t *core, read_dat_t *read_dat, char *seq);

std::vector<uint64_t> get_seq_to_sig_map(
	const std::vector<uint8_t>& moves,
	const size_t signal_len,
	const size_t reserve,
	size_t canonical_stride
);

std::vector<int> sequence_to_ints(const std::string& sequence);

void populate_hits_sig(
	std::array<std::vector<int64_t>, 4>& per_base_hits_sig,
	std::array<std::vector<int64_t>, 4>& per_base_hits_seq,
	std::vector<uint64_t>& seq_to_sig_map,
	const int base_id
);

void populate_signal(
	core_t *core,
	at::Tensor& signal,
	std::vector<uint64_t>& seq_to_sig_map,
	const std::vector<int>& int_seq
);

size_t create_mod_chunks(std::vector<mod_chunk_t> &chunks, core_t *core, read_dat_t *read_dat);

std::vector<bool> get_minimal_encoding_skips(
	core_t *core,
	const std::vector<mod_chunk_t>& chunks_by_caller,
	const std::vector<uint64_t>& seq_to_sig_map,
	std::vector<int> &int_seq
);

void populate_encoded_kmer(
	std::vector<int8_t>& encoded_kmer,
	const std::size_t signal_len,
	const std::vector<int>& int_seq,
	const std::vector<uint64_t>& seq_to_sig_map,
	const std::vector<bool>& base_skips,
	int kmer_len,
	int sequence_stride_ratio
);

bool validate_bam_tag_code(const std::string& bam_name);

// Extract per-hit modbase scores for a single chunk into read_dat->base_mod_probs.
// scores_ptr points at the (CPU, fp16, contiguous) model output; row_offset/row_size locate this
// chunk's row within it. Derived from dorado; kept in the dorado tree for licensing.
void extract_mod_probs(
	const core_t *core,
	const mod_chunk_t *chunk,
	const c10::Half *scores_ptr,
	int64_t row_offset,
	int64_t row_size
);

#endif
