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
