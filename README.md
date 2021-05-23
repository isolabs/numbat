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

### mars32k

The mars32k dataset is an unlabelled set of terrain images available here https://dominikschmidt.xyz/mars32k/. It contains images taken with Curiosity's Mastcam camera.

It can be downloaded and unzipped as required using the following commands:

```console
cd data
curl -O http://download1979.mediafire.com/tnzs5gujohyg/r49yc8jvl26tjo8/mars32k.zip
unzip mars32k.zip
```

### msl

The msl dataset is a labelled classification dataset available here https://zenodo.org/record/1049137#.YKLKXahKiPq. It contains train, test, and validation sets according to their Martian day of acquisition.

