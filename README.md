# kegg-hgt

## Heterogeneous Graph Transformer for learning Compound-Ortholog links from the Kyoto Encyclopedia of Genes and Genomes (KEGG)

This repository contains code for building and running a HGT model on a constructed knowledge graph from the KEGG database containing features for Kegg Orthologs and compounds.

Create your conda environment:

```
conda env create --name YOUR_ENV_NAME --file environment.yml
conda activate YOUR_ENV_NAME
```

How to set up:

```
pip install -e .
```

To run the code as is:

```
cd scripts
bash run
```

`scripts/run` contains how to use the run.py to run with your parameters.

Note: Make sure to have git-lfs (git large file system) installed. The file `data/embeddings/prok_esm650_ko_emb.csv` is larger than 100MB and is necessary to build the KG.

## About the project

The idea for this project is build a knowledge graph (KG) from the KEGG database from a subset of the data types contained in the database and evaluate a heterogeneous GNN on a link prediction task between 2 different types. Here we focus on learning a link prediction task by optimizing the node embeddings for each type to accurately capture compound-ortholog relationships.

### Node features and types

We have two types of nodes in this graph: compounds, which are represented by a MACCS structural key from the PubChem database, and KEGG Orthologs (KO) which are represented as dense vectors that are mean-pooled protein embeddings from the ESM-2 transformer-based protein language model. A KO represents a family (or set) of proteins that related by function. In order to represent each KO in a compact way, we just take the average of the embeddings for each protein within the KO. This produces a single vector for each KO. Below is an subgraph from our graph which illustrates this idea.

![alt text](https://github.com/nimuh/kegg-hgt/blob/main/figures/kegg_het_net_K18983.png)

### The HGT Model

The model is based on the Heterogeneous Graph Transformer model. We use the same algorithm as defined in the paper from the implementation in `pytorch geometric`. We add a link prediction head on top of the GNN to predict if a compound and a KO link or not.

### Experiments

We ran a series of experiments to evaluate the HGT link model. We tested a series of attention heads and hidden layer size values. Each experiment was ran for 10 epochs due to time and resource constraints.
