#include <string>
#include <iostream>
#include <fstream>

void write_to_file(std::ofstream &out, std::string &sequence, std::string &qstring, char *read_id, bool emit_fastq) {
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
	    out << "@" << read_id << "\n"
                  << sequence << "\n"
                  << "+\n"
                  << qstring << "\n";
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