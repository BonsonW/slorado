#pragma once

#include "Chunk.h"
#include <vector>

// Given a read with unstitched chunks, stitch the chunks (accounting for overlap) and assign basecalled read and qstring to Read
void stitch_chunks(std::vector<Chunk> chunks, std::string &sequence, std::string &qstring);
