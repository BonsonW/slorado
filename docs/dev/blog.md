# Slorado v0.2.0

Nanopore technology in the context of genomics provides a cost-effective and fully scalable method for long-read sequencing. With the new 5.0.0 models, and tools such as [ReadFish](https://github.com/LooseLab/readfish), we are continuously seeing improvements in the accuracy and application DNA basecalling. The basecalling technology itself however, has historically been limited to [ONT’s Dorado Basecaller](https://github.com/nanoporetech/dorado). If one were to improve on this software, they would face challenges in profiling and extending the basecaller. Here we provide an early snapshot of Slorado, a simplified and **completely open-source** version of Dorado that enables GPU acceleration on both **NVIDIA** and **AMD** devices.

## Background

// developing a simple framework for benchmarking and optimizations

// experimenting with different architectures

## The KOI library

// explain what the KOI library is

## Slorado's Execution Breakdown

// show breakdown of time, simplify plot (separate AMD and NVIDIA parts)

// show that implementing on GPU would eliminate copy time (the bottleneck)

// show benchmarks between original CPU decoding and openfish

## Results

// 16 million read accuracies

// AMD accelerated version vs CPU

// NVIDIA accelerated version vs CPU vs KOI

## Installation and Usage