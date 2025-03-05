# Basecalling on Pawsey's AMD GPUs


With slorado, now you can do some basecalling of your nanopore data on [Australia's Pawsey supercomputer](https://pawsey.org.au/). The [Setonix cluster](https://pawsey.org.au/systems/setonix/) in Pawsey has several hundred AMD Instinct MI250X GPUs.
For those who have access to Pawsey, this post will show how you can do this. 

## Getting started

### Installing
First, download and extract the slorado rocm Linux binaries tarball.

```
VERSION=v0.2.0-beta
wget "https://github.com/BonsonW/slorado/releases/download/$VERSION/slorado-$VERSION-x86_64-rocm-linux-binaries.tar.gz"
tar xvf slorado-$VERSION-x86_64-rocm-linux-binaries.tar.gz
cd slorado-$VERSION
bin/slorado --help
```
Detailed instructions are found [here](rocm-bin.md)

### Example Datasets

Path to example dataset on Pawsey: `/scratch/references/slorado/slorado-v0.2.0-beta/slow5-testdata/hg2_prom_lsk114_5khz_chr22/PGXXXX230339_reads_chr22.blow5`

Or you may download a 20k dataset with: `wget -O PGXXXX230339_reads_20k.blow5 https://slow5.bioinf.science/hg2_prom_5khz_subsubsample`

### Example Slurm Script
Copy the example slurm script in [Note 1](#note-1) in to a file called `example.sh`, and simply call `sbatch --account=${PAWSEY_PROJECT}-gpu example.sh` to submit the test job. 
This script will basecall the above test dataset using the `dna_r10.4.1_e8.2_400bps_hac@v4.2.0` model and generate a fastq file in the current directory called `reads.fastq`.

## Running on your own data

To run on your own data, below are the steps. If you run into a problem, feel free to open a [GitHub issue](https://github.com/BonsonW/slorado/issues).

1. Convert your POD5 files to BLOW5 using [blue-crab](https://github.com/Psy-Fer/blue-crab) or FAST5 files to BLOW5 using [slow5tools](https://github.com/hasindu2008/slow5tools). See [Note 2](#note-2) for some example commands.
2. Now change the example slurm script to point to this BLOW5 file you just created.
3. Edit other options such as the model, output file path, the max wall-clock time, and number of GPUs in the script as you wish (see the comments in the script)
4. Now submit the job
5. Once the job is finished, you can do some sanity checks, for instance, mapping the reads to the reference using minimap2 and checking the identity scores.

## Tests and benchmarks

We tested slorado on [Pawsey using a complete PromethION dataset (~20X coverage HG002)](https://gentechgp.github.io/gtgseq/docs/data.html#na24385-hg002-promethion-data-20x). We used all 4 MI250X GPUs on the node (8 graphics compute dies in total). The execution times were as follows for the three different basecalling models:

| Basecalling model | Execution time (hh:mm:ss) |
|---|---|
| super accuracy (dna_r10.4.1_e8.2_400bps_sup@v4.2.0)    | 21:03:59       |
| high accuracy  (dna_r10.4.1_e8.2_400bps_hac@v4.2.0)    | 07:30:45        |
| fast (dna_r10.4.1_e8.2_400bps_fast@v4.2.0)             | 04:46:31        |

After basecalling, we aligned the reads to the hg38 genome using minimap2 and calculated the statistics (e.g., mean, median) for the identity scores. The values were as expected to what we see in the same model versions in the original Dorado:

| Basecalling model | mean (slorado) | median (slorado) | mean (Dorado) | median (Dorado) |
|---|---|---|---|---|
| super accuracy (dna_r10.4.1_e8.2_400bps_sup@v4.2.0)    | 0.950829    | 0.985673    | 0.946401 | 0.985461 |
| high accuracy  (dna_r10.4.1_e8.2_400bps_hac@v4.2.0)    | 0.941249    | 0.977212    | 0.938371 | 0.977654 |
| fast (dna_r10.4.1_e8.2_400bps_fast@v4.2.0)             | 0.912168    | 0.938976    | 0.906703 | 0.937500 |

---

### Note 1

Pawsey example slurm script:
```
#!/bin/bash --login
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00

# usage: sbatch --account=${PAWSEY_PROJECT}-gpu example.sh

# you can get the test BLOW5 as: wget -O PGXXXX230339_reads_chr22.blow5 https://slow5.bioinf.science/hg2_prom_5khz_chr22
BLOW5=/scratch/references/slorado/slorado-v0.2.0-beta/slow5-testdata/hg2_prom_lsk114_5khz_chr22/PGXXXX230339_reads_chr22.blow5
FASTQ_OUT=reads.fastq

# you may change the model one of the available:  dna_r10.4.1_e8.2_400bps_fast@v4.2.0  dna_r10.4.1_e8.2_400bps_hac@v4.2.0  dna_r10.4.1_e8.2_400bps_sup@v4.2.0
MODEL=dna_r10.4.1_e8.2_400bps_hac@v4.2.0
# batch size must be adjusted so that we do not overflow the GPU memory: tested to work:  2000 for fast, 500 for hac and 200 for sup
BATCH_SIZE=500

########################################################################

SLORADO_DIR=/scratch/references/slorado/slorado-v0.2.0-beta
SLORADO=${SLORADO_DIR}/bin/slorado

srun /usr/bin/time -v ${SLORADO} basecaller ${SLORADO_DIR}/models/${MODEL} ${BLOW5} -o ${FASTQ_OUT} -t64 -C ${BATCH_SIZE}
```

See the [Pawsey GPU documentation](https://pawsey.atlassian.net/wiki/spaces/US/pages/51928618/Setonix+GPU+Partition+Quick+Start) for best practices on using GPUs effectively. 

### Note 2

On Pawsey, I could install and use blue-crab using a virtual environment as follows:
```
# First install
python3.11 -m venv ./blue-crab-venv
source blue-crab-venv/bin/activate
python3 -m pip install --upgrade pip
pip install blue-crab

# Then convert a POD5 directory to BLOW5
blue-crab p2s pod5_dir/ -o merged.blow5
```
Alternatively, [@gbouras13](https://github.com/gbouras13) has created a docker image, so you can use it through singularity as well:
```
module  load pawseyenv/2023.08
module load singularity/3.11.4-slurm
 
singularity pull --dir $PWD docker://quay.io/gbouras13/blue_crab:0.2.0
singularity exec blue_crab_0.2.0.sif blue-crab p2s pod5_dir/ -o merged.blow5
```


On Pawsey, you can simply download the slow5tools binaries, extract it and use:
```
VERSION=v1.3.0
wget "https://github.com/hasindu2008/slow5tools/releases/download/$VERSION/slow5tools-$VERSION-x86_64-linux-binaries.tar.gz" && tar xvf slow5tools-$VERSION-x86_64-linux-binaries.tar.gz && cd slow5tools-$VERSION/
./slow5tools
```


   
