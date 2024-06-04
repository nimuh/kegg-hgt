# kegg-hgt

## Heterogeneous Graph Transformer for learning Compound-Ortholog links from the Kyoto Encyclopedia of Genes and Genomes (KEGG)

This repository contains code for building and running a HGT model on a constructed knowledge graph from the KEGG database containing features for Kegg Orthologs and compounds.

Create your conda environment:

```
conda env create --name YOUR_ENV_NAME --file environment.yml
```

How to set up:

```
python setup.py
```

To run the code as is:

```
cd scripts
bash run
```

`scripts/run` contains how to use the run.py to run with your parameters.

Note: Make sure to have git-lfs (git large file system) installed. The file `data/embeddings/prok_esm650_ko_emb.csv` is larger than 100MB and is necessary to build the KG.
