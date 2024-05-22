import logging
import argparse
import warnings
import torch
import pandas as pd
from torch_geometric import seed_everything
from torch_geometric.nn import GAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils.convert import from_networkx

import networkx as nx
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from kgt.models.gcn import GCNEncoder


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

seed_everything(42)

# TODO
# add second data type to the graph
# add compounds connected to KOs by reaction
# set KOs to be average of protein embeddings from ESM
# implement HGT model
# measure link prediction performance?


def main():
    args = parse_cmd_args()
    graph_filename = args.graph_file

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    geo_g, geo_g_train, geo_g_val, _, _ = process_kegg_graph(
        graph_filename,
        val_frac=0.20,
        test_frac=0.20,
        feat_name=args.feature,
        # add_node_cid=5312377,
    )

    out_channels = args.outdim
    num_features = geo_g_train.num_features
    epochs = args.epochs
    encoder = GCNEncoder(num_features, out_channels)
    model = GAE(encoder)

    # training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        loss = train(model, geo_g_train, optimizer)
        z = model.encode(geo_g.x, geo_g.edge_index)
        model.eval()
        auc, ap = model.test(
            z, geo_g_val.pos_edge_label_index, geo_g_val.neg_edge_label_index
        )
        logging.info(
            "EPOCH: {:03d} TRAIN LOSS: {:.4f} AUC: {:.4f} AP: {:.4f}".format(
                epoch, loss, auc, ap
            )
        )


"""
CMD line parser
"""


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_file")
    parser.add_argument("--feature")
    parser.add_argument("--met_data")
    parser.add_argument("--metadata")
    parser.add_argument("--data_info")
    parser.add_argument("--hiddendim", type=int)
    parser.add_argument("--outdim", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


"""
Training logic for GNN
"""


def train(model, g_train, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(g_train.x, g_train.edge_index)
    loss = model.recon_loss(
        z,
        g_train.pos_edge_label_index,
        neg_edge_index=g_train.neg_edge_label_index,
    )
    loss.backward()
    optimizer.step()
    return float(loss)


"""
To predict edges of newly added node(s) to graph
"""


def predict(model, g, inf_g):
    model.eval()
    z = model.encode(g.x, g.edge_index)
    h = model.encode(inf_g.x, inf_g.edge_index)
    probs = torch.sigmoid(torch.matmul(z, h.T))[:, 0]
    best_matches = torch.argwhere(torch.where(probs > 0.90, probs, 0))[:, 0]
    best_probs = [probs[j].detach().numpy().item() for j in best_matches]
    best_comps = [g.kegg_cpd[best_match] for best_match in best_matches]
    best_matches = dict(zip(best_comps, best_probs))
    best_matches = {
        k: v
        for k, v in sorted(best_matches.items(), reverse=True, key=lambda item: item[1])
    }
    for k, v in best_matches.items():
        print(k, v)


"""
General function to query pubchem with given CID
"""


def query_pubchem(cid):
    compound_info = pcp.Compound.from_cid(cid)
    cid = int(compound_info.cid)
    s = compound_info.canonical_smiles
    print(f"got cid: {cid}")
    rdk_mol = Chem.MolFromSmiles(s)
    fpt = MACCSkeys.GenMACCSKeys(rdk_mol).ToBitString()
    print(f"got fpt: {fpt}")
    return fpt, str(cid)


"""
For adding any new nodes to the graph for inference
"""


def add_node_to_graph(g, pc_cid, feat_name):
    fpt, cid = query_pubchem(pc_cid)
    new_g = nx.Graph()
    new_g.add_edges_from(g.edges())
    new_g.add_nodes_from(g.nodes(data=True))
    new_g.add_node(cid)
    new_g.nodes[cid]["cid"] = cid
    new_g.nodes[cid]["maccs"] = fpt
    new_g.nodes[cid]["kegg_cpd"] = "N/A"
    new_g = from_networkx(new_g)
    node_features_str = list(map(list, new_g[feat_name]))
    new_node_features = []
    for node_feature in node_features_str:
        new_node_features.append(torch.Tensor(list(map(int, node_feature))))
    new_g.x = torch.stack(new_node_features)
    return new_g


# TODO
# map each KO in the graph to a vector that is the average over the embeddings
# produced by ESM-2-650M
def add_ko_embeddings(embeddings):

    # for each KO in embeddings
    # for each node in graph G
    # if node is a KO node AND matches to current KO
    # set feature to be embedding associated with KO

    pass


# TODO
# add KO data type so that there are 2 node types in the graph
# T(V) = <CPD, KO>
# PHI(E) = <CPD, REACTS, CPD> or <> <CPD, INTERACTS, KO>
# This way we also have 2 edges types representing compound-compound
# relationships or compound-protein relationships
# HGT should capture this
def add_kegg_data_to_graph(g, type="ko"):

    # grab relations from KEGG DB

    # read text file so that each reaction maps to a list of KOs

    # read text file so that each compound maps to a list of reactions

    # iterate through nodes in graph
    # for each compound
    # find node with CPD ID
    # iterate through reactions of CPD
    # for each reaction, get associated KO
    # add KO node
    # add edge between CPD and KO

    pass


"""
For processing and loading in KG for training and validation
"""


def process_kegg_graph(filename, val_frac, test_frac, feat_name, add_node_cid=None):
    nx_file = filename
    split_transform = RandomLinkSplit(
        num_val=val_frac,
        num_test=test_frac,
        add_negative_train_samples=True,
        split_labels=True,
        is_undirected=True,
    )

    print(f"reading in file {nx_file}")
    g = nx.read_graphml(nx_file)
    new_g = None
    if add_node_cid:
        new_g = add_node_to_graph(g, add_node_cid, feat_name)

    print(f"# of Nodes: {len(g.nodes)}")
    print(f"of Edges: {len(g.edges)}")
    g.remove_edges_from(nx.selfloop_edges(g))
    print(f"Removing self edges ... # of edges left: {len(g.edges)}")
    print()
    geo_g = nx.Graph()
    geo_g.add_edges_from(g.edges())
    geo_g.add_nodes_from(g.nodes(data=True))
    geo_g = from_networkx(geo_g)
    print("converting node features to tensors...")
    node_features_str = list(map(list, geo_g[feat_name]))
    new_node_features = []
    for node_feature in node_features_str:
        new_node_features.append(torch.Tensor(list(map(int, node_feature))))
    geo_g.x = torch.stack(new_node_features)
    g_train, g_val, g_test = split_transform(geo_g)
    return geo_g, g_train, g_val, g_test, new_g


if __name__ == "__main__":
    main()
