/* @file writer.h
**
** methods for writing data to files
** @@
******************************************************************************/

#include "slorado.h"
#include <string>

#ifndef WRITER_H
#define WRITER_H

void write_to_file(std::string &sequence, std::string &qstring, char *read_id, opt_t &opt);

#endif