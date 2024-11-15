# Basecalling on Pawsey's AMD GPUs

With slorado, now you can do some basecalling of your nanopore data on [Australia's Pawsey supercomputer](https://pawsey.org.au/). The [Sentonix cluster](https://pawsey.org.au/systems/setonix/) in Pawsey has many AMD Instinct MI250X GPUs.
For those who have access to Pawsey, this post will show how you can do this. The binaries have been already installed on a shared location, so you do not need to compile. 

Following are the directory paths to get started:

- slorado binaries: `/scratch/references/slorado/slorado-06-11-2024/`.
- test dataset: `/scratch/references/slorado/slow5-testdata/hg2_prom_lsk114_5khz_chr22/PGXXXX230339_reads_chr22.blow5`
- example slurm script: `/scratch/references/slorado/slorado-06-11-2024/scripts/slurm.sh`

First, copy the example slurm script and change the account in the header from `pawsey0001-gpu` to your account code. Now you simply call `sbatch slurm.sh` to submit the test job. 
This script will basecall the above test dataset using the `dna_r10.4.1_e8.2_400bps_hac@v4.2.0` model and generate a fastq file in the current directory called giga.fastq.

To run on your data, below are the steps.

1. Convert your POD5 files to BLOW5 using [blue-crab](https://github.com/Psy-Fer/blue-crab). I could install and use blue-crab as follows:
    ```
    # First install
    python3.11 -m venv ./blue-crab-venv
    source blue-crab-venv/bin/activate
    python3 -m pip install --upgrade pip
    pip install blue-crab

    T# hen convert a POD5 directory to BLOW5
    blue-crab p2s pod5_dir/ -o merged.blow5
    ```
   
2. Now change the example slurm script to point to this BLOW5 file you just created.
3. Edit other options such as the model, output file path, the max wall-clock time, number of GPUs in the script as you wish (see the comments in the script)
4. Now submit the job
5. Once the job is finished, you can do some sanity checks, for instance, mapping the reads to the reference using minimap2 and see checking the identity scores.

