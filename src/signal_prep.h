#include <slow5/slow5.h>
#include <torch/torch.h>

struct Chunk {
    Chunk(size_t offset, size_t chunk_in_read_idx, size_t chunk_size) :
        input_offset(offset),
        idx_in_read(chunk_in_read_idx),
        raw_chunk_size(chunk_size){};

    size_t input_offset; // Where does this chunk start in the input raw read data
    size_t idx_in_read; // Just for tracking that the chunks don't go out of order
    size_t raw_chunk_size; // Just for knowing the original chunk size

    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves; // For stitching.
};

slow5_rec_t *read_file_to_record(char *file_path);
torch::Tensor tensor_from_record(slow5_rec_t *rec);
int trim_signal(torch::Tensor signal, int window_size=40, float threshold_factor=2.4, int min_elements=3);
void scale_signal(torch::Tensor &signal);
std::vector<Chunk> chunks_from_tensor(torch::Tensor &tensor, int chunk_size, int overlap);