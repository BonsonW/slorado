/* @file modcall.h
**
** methods for base modification calling step
** @@
******************************************************************************/

#ifndef MODCALL_H
#define MODCALL_H

#include "slorado.h"

void mod_basecall_db(core_t* core, db_t* db);

// Run modbase inference on a packed batch of mod-chunk pointers using the given runner.
// Exposed for the streaming pipeline (modcall.cpp defines it).
void mod_basecall_chunks(const core_t* core, const int runner_idx, const std::vector<mod_chunk_t *> &chunks);

#endif
