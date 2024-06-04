
import torch
from torch_geometric.nn import HGTConv, Linear

"""
Sources 
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py
https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
"""
class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['cpd']), self.lin(x_dict['ko'])
    

class LinkPredictionModule(torch.nn.Module):
    def forward(self, x_cpd, x_ko, edge_label_index):
        edge_feat_user = x_cpd[edge_label_index[0]]
        edge_feat_movie = x_ko[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class HGTLink(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, data, num_heads=3, num_layers=3):
        super().__init__()
        self.gnn = HGT(hidden_channels, out_channels, num_heads, num_layers, data)
        self.classifier = LinkPredictionModule()

    def forward(self, data):
        embeddings = self.gnn(data.x_dict, data.edge_index_dict)
        x_dict = {'cpd': embeddings[0], 'ko': embeddings[1]}
        pred = self.classifier(
            x_dict["cpd"],
            x_dict["ko"],
            data["cpd", "interacts", "ko"].edge_label_index,
        )

        return pred
