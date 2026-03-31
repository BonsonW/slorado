/* @file writer.h
**
** methods for writing data to files
** @@
******************************************************************************/

#ifndef WRITER_H
#define WRITER_H

#include <cstdint>

void write_to_file_fastq(FILE *out, char *sequence, char *qstring, char *read_id);
void write_to_file_sam(FILE *out, char *sequence, char *qstring, char *read_id, char *mod_string, std::vector<uint8_t> &mod_prob);

#endif