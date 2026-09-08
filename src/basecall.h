/* @file basecall.h
**
** methods for base calling step
** @@
******************************************************************************/

#ifndef BASECALL
#define BASECALL

#include "slorado.h"

void basecall_db(core_t* core, db_t* db);
void mod_basecall_db(core_t* core, db_t* db);

void basecall_chunks(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks);
void mod_basecall_chunks(const core_t* core, const int runner_idx, const std::vector<mod_chunk_t *> &chunks);

#endif