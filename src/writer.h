/* @file writer.h
**
** methods for writing data to files
** @@
******************************************************************************/

#ifndef WRITER_H
#define WRITER_H

void write_to_file(FILE *out, char *sequence, char *qstring, char *read_id, bool emit_fastq);

#endif