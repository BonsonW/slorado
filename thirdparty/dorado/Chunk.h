#ifndef CHUNK_H
#define CHUNK_H

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>

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

#endif