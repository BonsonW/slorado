#! /usr/bin/env Rscript

#---------------------------------------------------------
# Adapted from f5c/nanopolish plot_methylation.R (Jared Simpson, OICR).
# Original: https://nanopolish.readthedocs.io/en/latest/quickstart_call_methylation.html
#
# 2D-binned scatter of two callers' per-site methylation frequencies with the
# Pearson correlation in the title. Input is the TSV emitted by
# scripts/compare_methylation.py (columns: key depth_1 frequency_1 depth_2 frequency_2).
#
# Generalized here: axis labels and title are configurable (default dorado vs slorado),
# and sites can be filtered by a minimum coverage (min of the two callers' depths).
#
# Usage:
#   Rscript scripts/plot_methylation.R -i cmp.tsv -o cmp.pdf
#   Rscript scripts/plot_methylation.R -i cmp.tsv -o cmp.pdf --xlab dorado --ylab slorado --mincov 5
#---------------------------------------------------------

library(ggplot2)
library(RColorBrewer)
library(optparse)

option_list = list(
    make_option(c("-i", "--input"), type="character", default=NULL,
              help="input methylation comparison tsv (from compare_methylation.py)", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="out.pdf",
              help="output plot file name [default= %default]", metavar="character"),
    make_option(c("-x", "--xlab"), type="character", default="dorado",
              help="x-axis caller name [default= %default]", metavar="character"),
    make_option(c("-y", "--ylab"), type="character", default="slorado",
              help="y-axis caller name [default= %default]", metavar="character"),
    make_option(c("-t", "--title"), type="character", default=NULL,
              help="plot title [default: auto 'N=.. r=..']", metavar="character"),
    make_option(c("-c", "--mincov"), type="integer", default=1,
              help="min per-site coverage (min of the two callers) [default= %default]", metavar="integer")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$input)){
  print_help(opt_parser)
  stop("At least -i input.tsv must be provided.", call.=FALSE)
}

data <- read.table(opt$input, header=T)

# filter by min coverage (min of the two callers' depths at each site)
if (opt$mincov > 1) {
  data <- data[pmin(data$depth_1, data$depth_2) >= opt$mincov, ]
}

r <- cor(data$frequency_1, data$frequency_2)
# N/r always shown; if a descriptive --title (dataset/models) is given it's the main
# heading and the N/r line becomes the subtitle.
stats <- sprintf("N = %d   r = %.6f   (mincov>=%d)", nrow(data), r, opt$mincov)
if (is.null(opt$title)) {
  main <- stats; sub <- NULL
} else {
  main <- opt$title; sub <- stats
}

# Spectral heatmap palette (reversed), log10 count scale
rf <- colorRampPalette(rev(brewer.pal(11, 'Spectral')))

pdf(file = opt$out, width=10, height=8)
print(
  ggplot(data, aes(frequency_1, frequency_2)) +
    geom_bin2d(bins=25) +
    scale_fill_gradientn(colors=rf(32), trans="log10") +
    xlab(sprintf("%s methylation frequency", opt$xlab)) +
    ylab(sprintf("%s methylation frequency", opt$ylab)) +
    theme_bw(base_size=20) +
    ggtitle(main, subtitle=sub)
)
dev.off()

cat(sprintf("wrote %s  (N=%d, r=%.6f, mincov>=%d)\n", opt$out, nrow(data), r, opt$mincov))
