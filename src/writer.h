/* @file writer.h
**
** methods for writing data to files
** @@
******************************************************************************/

#include <string>

#ifndef WRITER_H
#define WRITER_H

void write_to_file(std::string &name, std::string &sequence, std::string &qstring, std::string &read_id, bool emit_fastq);

#endif