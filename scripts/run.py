import logging
import argparse
import warnings
import torch
import pandas as pd
from torch_geometric import seed_everything
from collections import defaultdict
from torch_geometric.nn import GAE, to_hetero
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import numpy as np
import networkx as nx
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from kgt.models.hgt import HGTLink, HGTLConfig
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import tqdm
import torch.nn.functional as F


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

seed_everything(42)


def main():
    args = parse_cmd_args()
    graph_filename = args.graph_file

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train_data, val_data, test_data = process_kegg_graph_het(graph_filename)
    # hidden = args.hiddendim
    epochs = args.epochs
    # output_dim = args.outdim
    # heads = args.attn_heads

    config = HGTLConfig(
        hidden_channels=args.hiddendim,
        out_channels=args.outdim,
        num_heads=args.attn_heads,
        num_layers=args.num_layers,
    )

    train_model(
        train_data,
        val_data,
        config=config,
        # attn_heads=heads,
        # hidden_channels=hidden,
        epochs=epochs,
        # output_dims=output_dim,
    )

    # TODO
    # run on test data too
    # clean up parser code


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_file")
    parser.add_argument("--hiddendim", type=int)
    parser.add_argument("--outdim", type=int)
    parser.add_argument("--attn_heads", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num_layers", type=int)
    args = parser.parse_args()
    return args


def query_pubchem(cid):
    """
    For querying pubchem database with given CID

    Arguments:
        cid: compound id to query pubchem for.

    Returns:
        MACCSKey: fingerprint representing molecule
        cid: pubchem ID as a string.
    """
    compound_info = pcp.Compound.from_cid(cid)
    cid = int(compound_info.cid)
    s = compound_info.canonical_smiles
    print(f"got cid: {cid}")
    rdk_mol = Chem.MolFromSmiles(s)
    fpt = MACCSkeys.GenMACCSKeys(rdk_mol).ToBitString()
    print(f"got fpt: {fpt}")
    return fpt, str(cid)


def add_node_to_graph(g, pc_cid, feat_name):
    """
    Add a compound to the KEGG KG for the compound that maps to the pc_id.

    Arguments:
        g: networkx graph or torch data
        pc_cid: pubchem compound id used to query the pubchem database
        feat_name: the name of the features for representing the compound

    Returns:
        networkx Graph with added compound
    """

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


def add_embeddings(embeddings_path):
    """
    Given a set of embeddings with their associated labels, we add them to the
    graph.

    Arguments:
        embeddings: file containing embeddings for node type
        node_type: string defining the type of node we want to add the
                   embeddings for.

    Returns:
        Graph: New graph, new_g, with added embeddings.
    """

    emb_df = pd.read_csv(embeddings_path)
    emb_df = emb_df.set_index("KO")
    return emb_df


def add_kegg_data_to_graph(
    g, relations_txt_file_c_r, relations_txt_file_r_ko
) -> HeteroData:
    """
    Given a graph g and the relations_txt_file, this function adds the new
    relations and nodes to the graph

    Arguments:
        g: Graph, a networkx graph
        relations_txt_file: text file containing edges in (entity1, entity2)
                            format

    Returns:
        A new graph, new_g, that contains the new information from
        relations_txt_file.
    """

    cpd_to_r = defaultdict(list)
    r_to_ko = defaultdict(list)
    cpd_to_ko = defaultdict(list)
    # read text file so that each reaction maps to a list of KOs
    # use hashtable to map compounds to list of reactions
    with open(relations_txt_file_c_r) as f:
        for line in f.readlines():
            cpd, rn = line.split()
            cpd_to_r[cpd].append(rn)

    # use hashtable to map reactions to list of KOs
    with open(relations_txt_file_r_ko) as f:
        for line in f.readlines():
            ko, rn = line.split()
            r_to_ko[rn].append(ko.split(":")[1])

    # construct cpd to ko relationships
    for cpd in cpd_to_r:
        for rn in cpd_to_r[cpd]:
            for ko in r_to_ko[rn]:
                cpd_to_ko[cpd].append(ko)

    v_cpd = list(g.nodes())
    for cpd in v_cpd:
        kegg_id = g.nodes()[cpd]["kegg_cpd"]
        for ko in cpd_to_ko[kegg_id]:
            emb = [0 for _ in range(100)]
            g.add_node(ko, emb=emb)
            g.add_edge(cpd, ko)

    cpd_to_idx = defaultdict(int)
    ko_to_idx = defaultdict(int)
    cpd_to_maccs = defaultdict(list)
    ko_to_emb = defaultdict(np.array)
    ko_emb_df = add_embeddings("../data/embeddings/prok_esm650_ko_emb.csv")
    ko_emb_we_have = list(ko_emb_df.index)
    i, j = 0, 0
    V = list(g.nodes())
    for i, v_name in enumerate(V):
        if "pubchem" in v_name:
            cpd_to_idx[v_name] = i
            cpd_to_maccs[v_name] = list(map(int, g.nodes[v_name]["maccs"]))
            i += 1
        elif "K" in v_name:
            ko_to_idx[v_name] = j
            if v_name in ko_emb_we_have:
                ko_to_emb[v_name] = ko_emb_df.loc[v_name].values
                j += 1
            else:
                g.remove_node(v_name)

    g.remove_edges_from(nx.selfloop_edges(g))

    # Remove this graph plotting code

    #######################################
    """
    nodes_sorted = sorted(g.degree, key=lambda x: x[1], reverse=True)
    print(nodes_sorted)
    g.remove_edges_from(nx.selfloop_edges(g))
    eg = nx.ego_graph(g, "K18983", radius=2)
    color_map = []
    for node in eg:
        if "pubchem" in node:
            color_map.append("blue")
        else:
            color_map.append("red")

    nx.draw(eg, pos=graphviz_layout(eg), node_color=color_map, with_labels=True)
    plt.savefig("../figures/kegg_het_net_K18983.png", format="PNG")
    """
    ##################################

    cpd_ko_edges = [
        (cpd_to_idx[c], ko_to_idx[k])
        for (c, k) in g.edges()
        if "pubchem" in c and "K" in k
    ]

    cpd_cpd_edges = [
        (cpd_to_idx[c], cpd_to_idx[k])
        for (c, k) in g.edges()
        if "pubchem" in c and "pubchem" in k
    ]

    maccs = torch.stack(list(map(torch.Tensor, cpd_to_maccs.values())))
    kos = torch.stack(list(map(torch.Tensor, ko_to_emb.values())))
    cpd_ko_edges = torch.Tensor(cpd_ko_edges).type(torch.int64).T
    cpd_cpd_edges = torch.Tensor(cpd_cpd_edges).type(torch.int64).T
    pyg_data = construct_het_graph(g, maccs, kos, cpd_cpd_edges, cpd_ko_edges)
    print(pyg_data)

    transform = T.RandomLinkSplit(
        is_undirected=True,
        num_val=0.2,
        num_test=0.2,
        # disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=True,
        edge_types=("cpd", "interacts", "ko"),
        rev_edge_types=("ko", "interacts", "cpd"),
    )
    train_data, val_data, test_data = transform(pyg_data)
    return train_data, val_data, test_data


def train_model(train_data, val_data, config, epochs=100):
    model = HGTLink(
        config=config,
        data=train_data,
    )

    # TRAINING
    training_epoch_losses = []
    val_epoch_losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(epochs):

        # TRAIN
        total_loss = total_examples = 0
        optimizer.zero_grad()
        model.train()
        pred = model(train_data)
        ground_truth = train_data["cpd", "interacts", "ko"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        training_epoch_losses.append(total_loss / total_examples)
        print(f"Epoch: {epoch:03d}, Train Loss: {total_loss / total_examples:.4f}")

        # Validation
        model.eval()
        total_loss = total_examples = 0
        pred = model(val_data)
        ground_truth = val_data["cpd", "interacts", "ko"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
        val_epoch_losses.append(total_loss / total_examples)
        print(f"Epoch: {epoch:03d}, Val Loss: {total_loss / total_examples:.4f}")

    plt.figure()
    plt.title(
        f"BCE Loss CPD-KO LP AH={config.num_heads} HD={config.hidden_channels} OD={config.out_channels} L={config.num_layers}"
    )
    plt.plot(training_epoch_losses, label="train BCE loss")
    plt.plot(val_epoch_losses, label="val BCE loss")
    plt.ylabel("BCE Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig(
        f"../figures/train_losses_heads={config.num_heads}_hidden={config.hidden_channels}_outdims={config.out_channels}_L={config.num_layers}_lin.png"
    )

    return model


# function should construct a HeteroData type for torch_geo from
# networkx graph
def construct_het_graph(
    g: nx.Graph, feat1, feat2, cpd_cpd_edges, cpd_ko_edges
) -> HeteroData:
    print(f"construcing Het graph from {g}...")

    het_graph = HeteroData()
    het_graph["cpd"].x = feat1
    het_graph["ko"].x = feat2
    het_graph["cpd", "reacts", "cpd"].edge_index = cpd_cpd_edges
    het_graph["cpd", "interacts", "ko"].edge_index = cpd_ko_edges
    het_graph = ToUndirected()(het_graph)
    return het_graph


def process_kegg_graph_het(filename):
    g = nx.read_graphml(filename)
    return add_kegg_data_to_graph(
        g, "../data/kegg_files/compound", "../data/kegg_files/ko.1"
    )


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
