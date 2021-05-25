# numbat

A transformer based approach to learning in the context of offworld terrain images. The approach leverages self-distillation with no supervision; it learns largely from unsupervised images.

## environment

This repository has been tested with CUDA 11.1 and the library versions listed in ```env.yml```. Recreate the environment using conda with the following command:

```console
conda env create --name numbat --file=env.yml
```

The ```env.yml``` file was written using the following command:

```console
conda env export | grep -v "^prefix: " > env.yml
```

## data

The datasets used here are publically available offworld terrain image datasets.

### mars32k

The mars32k dataset is an unlabelled set of terrain images available here https://dominikschmidt.xyz/mars32k/. It contains images taken with Curiosity's Mastcam camera.

It can be downloaded and unzipped as required using the following commands:

```console
cd data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yeLkE1p5oeCqa5pA7tc0tI4eoyvjZc5X' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yeLkE1p5oeCqa5pA7tc0tI4eoyvjZc5X" -O mars32k.zip && rm -rf /tmp/cookies.txt
unzip mars32k.zip
```

### msl

The msl dataset is a labelled classification dataset available here https://zenodo.org/record/1049137#.YKLKXahKiPq. It contains train, test, and validation sets according to their Martian day of acquisition.

### HiRISE

The Mars Orbital Image HiRISE dataset is a labelled classification dataset available here https://zenodo.org/record/4002935#.YKwVMahKiPo.

## improvements

* Multi-GPU parallelisation: https://pytorch.org/tutorials/beginner/dist_overview.html.