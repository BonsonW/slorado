#include <string>
#include <iostream>
#include <fstream>

using std::ofstream;
using std::cerr;
using std::endl;


void write_to_file(std::string &name, std::string &sequence, std::string &qstring, std::string &read_id, bool emit_fastq) {
    ofstream out;
    
    out.open(name + ".txt"); // opens the file
       if( !out ) { // file couldn't be opened
          cerr << "Error: file could not be opened" << endl;
          exit(1);
       }
    
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
    
    out.close();
}