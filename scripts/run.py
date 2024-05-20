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


def main():
    args = parse_cmd_args()
    print(f"SAVE ?: {args.save}")
    #print(f"metabolite dataset: {args.met_data}")
    #print(f"metabolite metadata: {args.metadata}")
    #print(f"metabolite dataset info: {args.data_info}")

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


def process_kegg_graph(filename, val_frac, test_frac, feat_name):
    nx_file = filename
    split_transform = RandomLinkSplit(
        num_val=val_frac,
        num_test=test_frac,
        add_negative_train_samples=True,
        split_labels=True,
        is_undirected=True,
    )

    print("reading in file {}".format(nx_file))
    g = nx.read_graphml(nx_file)
    print(g.nodes["pubchem:3303"])

    compound_info = pcp.Compound.from_cid(5312377)
    cid = int(compound_info.cid)
    s = compound_info.canonical_smiles
    print("got cid: ", cid)
    rdk_mol = Chem.MolFromSmiles(s)
    fpt = MACCSkeys.GenMACCSKeys(rdk_mol).ToBitString()
    print("got fpt: ", fpt)
    print("# of Nodes: {}".format(len(g.nodes)))
    print("adding node for 5D...")

    g_5d = nx.Graph()
    g_5d.add_node("5312377")
    g_5d.nodes["5312377"]["cid"] = "5312377"
    g_5d.nodes["5312377"]["maccs"] = fpt
    # G.nodes['5312377']['smiles'] = s
    g_5d.nodes["5312377"]["kegg_cpd"] = "N/A"
    print(g_5d.nodes["5312377"])
    g_5d = from_networkx(g_5d)
    print("converting node features to tensors...")
    node_features_str = list(map(list, g_5d[feat_name]))
    new_node_features = []
    for node_feature in node_features_str:
        new_node_features.append(torch.Tensor(list(map(int, node_feature))))
    g_5d.x = torch.stack(new_node_features)

    print("# of Nodes: {}".format(len(g.nodes)))
    print("# of Edges: {}".format(len(g.edges)))
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
    return geo_g, g_train, g_val, g_test, g_5d


if __name__ == "__main__":
    main()
