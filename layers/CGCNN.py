import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, CGConv, Set2Set
from torch_scatter import scatter_sum


class CGCNN(nn.Module):
    
    def __init__(self, layers, n_atom_feats, n_bond_feats, 
                hidden_dim, device):
        super().__init__()
        
        self.num_layers = layers
        
        self.convs = nn.ModuleList()

        self.convs.append(CGConv(n_atom_feats, n_bond_feats))
        for i in range(1, layers):
            self.convs.append(CGConv(n_atom_feats, n_bond_feats))

        self.lin = nn.Linear(n_atom_feats * (layers + 1), n_atom_feats)        

    def forward(self, data):

        h_list = [data.x]

        for layer in range(self.num_layers):
            h = F.elu(self.convs[layer](h_list[layer], data.edge_index, data.edge_attr))
            h_list.append(h)
        
        # Jumping Knowledge with Concatenation
        node_representation = torch.cat(h_list, dim = 1)
        node_representation = self.lin(node_representation)
        out = scatter_sum(node_representation, data.batch, dim = 0)

        return out, node_representation
        
    def get_embeddings(self, data):

        h_list = [data.x]

        for layer in range(self.num_layers):
            h = F.elu(self.convs[layer](h_list[layer], data.edge_index, data.edge_attr))
            h_list.append(h)
        
        # Jumping Knowledge with Concatenation
        node_representation = torch.cat(h_list, dim = 1)
        node_representation = self.lin(node_representation)
        crystal = self.set2set(node_representation, data.batch)

        return node_representation, crystal