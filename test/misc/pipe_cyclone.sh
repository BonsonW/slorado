INPUT="/data/bonwon/cyclone/test_data/wga/wga.blow5"
SAM="wga.sam"
ALIGNED_SAM="aligned_wga.sam"
FILTERED_SAM="filtered_aligned_wga.sam"
REF="/data/bonwon/cyclone/test_data/ref/ecoli.fa"
MODEL="/data/bonwon/slorado/models/cyclone_model_0_exported"

die() {
    echo "$@" >&2
    exit 1
}

[ ! -f "$REF" ] && die "Reference file not found: $REF"

# basecall
./slorado basecaller $MODEL $INPUT --emit-sam -o $SAM -c 10000 -p 500 -C 128 || die "Error in basecalling step"

# align reads to reference genome using minimap2 and sort the output
samtools fastq -TMM,ML $SAM | minimap2 -ax map-ont -Y -y --secondary=no $REF - |
samtools sort - -o $ALIGNED_SAM || die "Error in alignment step"

# filter reads under 200 bp
samtools view -h -F 4 $ALIGNED_SAM | \
awk 'BEGIN{OFS="\t"} /^@/ {print} !/^@/ { if (length($10) >= 200) print }' \
> $FILTERED_SAM || die "Error in filtering step"

paftools.js sam2paf $FILTERED_SAM | awk '{print $10/$11}' | datamash mean 1 sstdev 1 q1 1 median 1 q3 1 count 1
