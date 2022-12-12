/* @file writer.h
**
** methods for writing data to files
** @@
******************************************************************************/

#include <string>

#ifndef WRITER_H
#define WRITER_H

void write_to_file(FILE *out, std::string &sequence, std::string &qstring, char *read_id, bool emit_fastq);

#endif