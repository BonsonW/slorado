#pragma once

#include "Chunk.h"
#include "slorado.h"
#include <vector>

// Given a read with unstitched chunks, stitch the chunks (accounting for overlap) and assign basecalled read and qstring to Read
// returns {sequence, qstring}
std::pair<std::string, std::string> stitched_chunks(std::vector<Chunk> chunks);
