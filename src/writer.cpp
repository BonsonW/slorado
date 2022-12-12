#include <string>
#include <iostream>

void write_to_file(FILE *out, std::string &sequence, std::string &qstring, char *read_id, bool emit_fastq) {
    // todo:
    // if (!emit_fastq) {
    //     out << "@HD\tVN:1.5\tSO:unknown\n"
    //                 << "@PG\tID:basecaller\tPN:slorado\tVN:" << SLORADO_VERSION << "\tCL:slorado";
        
    //     for (const auto& arg : m_args) {
    //         outdata << " " << arg;
    //     }
    //     out << "\n";
    // }

    if (emit_fastq) {
        fprintf(out, "@%s\n", read_id);
        fprintf(out, "%s\n", sequence.c_str());
        fprintf(out, "+\n");
        fprintf(out, "%s\n", qstring.c_str());
    } else {
        // todo: 
        // try {
        //     for (const auto& sam_line : read->extract_sam_lines()) {
        //         outdata << sam_line << "\n";
        //     }
        // }
        // catch (const std::exception& ex) {
        //     std::cerr << ex.what() << "\n";
        // }
    }
}