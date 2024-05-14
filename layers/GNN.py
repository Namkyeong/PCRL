import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, GATConv, GINConv, GINEConv


class GCN(nn.Module):
    
    def __init__(self, node_input_dim = 200, node_hidden_dim = 200, 
                out_dim = 200, num_step_message_passing = 3):
        super().__init__()
        
        self.num_layers = num_step_message_passing
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_step_message_passing):
            self.convs.append(GCNConv(node_hidden_dim, node_hidden_dim))
            self.bns.append(nn.BatchNorm1d(node_hidden_dim))
        
        self.lin_ih = nn.Linear(node_input_dim, node_hidden_dim)
        self.lin = nn.Linear(node_hidden_dim * (num_step_message_passing + 1), out_dim)

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim

    def forward(self, x, edge_index):

        if self.node_input_dim != self.node_hidden_dim:
            h = self.lin_ih(x)
            h_list = [h]
        else:
            h_list = [x]

        for layer in range(self.num_layers):
            h = F.elu(self.convs[layer](h_list[layer], edge_index))
            h += h_list[layer]
            # h = self.bns[layer](h)
            h_list.append(h)
        
        # Jumping Knowledge with Concatenation
        node_representation = torch.cat(h_list, dim = 1)
        
        output = self.lin(node_representation)

        return output


class GAT(nn.Module):
    
    def __init__(self, node_input_dim = 200, node_hidden_dim = 200, 
                out_dim = 200, num_step_message_passing = 3):
        super().__init__()

        self.num_layers = num_step_message_passing
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_step_message_passing):
            self.convs.append(GATConv(node_hidden_dim, node_hidden_dim))
            self.bns.append(nn.BatchNorm1d(node_hidden_dim))
        
        self.lin_ih = nn.Linear(node_input_dim, node_hidden_dim)
        self.lin = nn.Linear(node_hidden_dim * (num_step_message_passing + 1), out_dim)

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim

    def forward(self, x, edge_index):

        if self.node_input_dim != self.node_hidden_dim:
            h = self.lin_ih(x)
            h_list = [h]
        else:
            h_list = [x]

        for layer in range(self.num_layers):
            h = F.elu(self.convs[layer](h_list[layer], edge_index))
            h += h_list[layer]
            # h = self.bns[layer](h)
            h_list.append(h)
        
        # Jumping Knowledge with Concatenation
        node_representation = torch.cat(h_list, dim = 1)
        
        output = self.lin(node_representation)

        return output


class GIN(nn.Module):
    
    def __init__(self, node_input_dim = 200, node_hidden_dim = 200, 
                out_dim = 200, num_step_message_passing = 3):

        super(GIN, self).__init__()
        
        self.num_layers = num_step_message_passing
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_step_message_passing):
            mlp = nn.Sequential(
                nn.Linear(node_hidden_dim, 2 * node_hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * node_hidden_dim, node_hidden_dim))
            
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(node_hidden_dim))

        self.lin_ih = nn.Linear(node_input_dim, node_hidden_dim)
        self.lin = nn.Linear(node_hidden_dim * (num_step_message_passing + 1), out_dim)

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim

    def forward(self, x, edge_index):

        if self.node_input_dim != self.node_hidden_dim:
            h = self.lin_ih(x)
            h_list = [h]
        else:
            h_list = [x]

        for layer in range(self.num_layers):
            h = F.elu(self.convs[layer](h_list[layer], edge_index))
            h += h_list[layer]
            # h = self.bns[layer](h)
            h_list.append(h)
        
        # Jumping Knowledge with Concatenation
        node_representation = torch.cat(h_list, dim = 1)
        
        output = self.lin(node_representation)

        return output 