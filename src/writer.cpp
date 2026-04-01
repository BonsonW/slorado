/**
 * @file writer.cpp
 * @brief writes output for decoded DNA sequences
 * @author Bonson Wong (bonson.ym@gmail.com)

MIT License

Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

#include <vector>
#include <string>

#include "error.h"
#include "writer.h"

void write_to_file_fastq(FILE *out, const char *sequence, const char *qstring, const char *read_id) {
    ASSERT(strlen(sequence) == strlen(qstring));

    int ret = fprintf(out, "@%s\n%s\n+\n%s\n", read_id, sequence, qstring);

    if (ret < 0) {
        ERROR("error writing fastq: %s", strerror(ret));
        exit(EXIT_FAILURE);
    }
}

static std::string mod_prob_to_str(const std::vector<uint8_t>& ml) {
    std::string s;
    s.reserve(ml.size() * 4);

    for (auto v : ml) {
        s.push_back(',');
        s += std::to_string(static_cast<int>(v));
    }

    return s;
}

void write_to_file_sam(FILE *out, const char *sequence, const char *qstring, const char *read_id, const char *mod_string, std::vector<uint8_t> &mod_prob) {
    ASSERT(strlen(sequence) == strlen(qstring));

    int flags = 4; // unmapped
    int pos = 0;
    int mapq = 0;
    int pnext = 0;
    int tlen = 0;

    const char *rname = "*";
    const char *cigar = "*";
    const char *rnext = "*";
    std::string mod_prob_string = mod_prob_to_str(mod_prob);

    int ret = fprintf(
        out,
        "%s\t%d\t%s\t%d\t%d\t%s\t%s\t%d\t%d\t%s\t%s\tMM:Z:%s\tML:B:C%s\n",
        read_id, flags, rname, pos, mapq, cigar, rnext, pnext, tlen, sequence, qstring,
        mod_string, mod_prob_string.c_str()
    );
    
    if (ret < 0) {
        ERROR("error writing sam file: %s", strerror(ret));
        exit(EXIT_FAILURE);
    }
}