import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import Set2Set

from layers import GCN, GAT, GIN
from torch_scatter import scatter_mean, scatter_sum


class Sto_Enc(nn.Module):
    """
    This the main class for our model
    """

    def __init__(self, gcn, layers, input_dim, hidden_dim, pooling):
        super(Sto_Enc, self).__init__()

        if gcn == "GCN":
            self.gnn = GCN(input_dim, hidden_dim, hidden_dim, layers)
        
        elif gcn == "GAT":
            self.gnn = GAT(input_dim, hidden_dim, hidden_dim, layers)
        
        elif gcn == "GIN":
            self.gnn = GIN(input_dim, hidden_dim, hidden_dim, layers)

        self.num_step_set2set = 2
        self.num_layer_set2set = 1

        # Mean Module
        self.mean_set2set = Set2Set(hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.mean_lin = nn.Linear(hidden_dim * 2, hidden_dim)

        # Uncertainty Module
        self.var_set2set = Set2Set(hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.var_lin = nn.Linear(hidden_dim * 2, hidden_dim)

        self.num_layers = layers
        self.pooling = pooling

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
            )

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_mean_var(self, data, weight = None):
        
        x = weight[data.sto_x - 1]
        x = self.gnn(x, data.sto_edge_index)
        
        weighted_x = x * data.sto_weight.reshape(-1, 1)
        sto = scatter_sum(weighted_x, data.sto_batch, dim = 0)

        mean = self.mean_set2set(weighted_x, data.sto_batch)
        mean = self.mean_lin(mean) + sto
        mean = F.normalize(mean, dim = 1)

        logsigma = self.var_set2set(weighted_x, data.sto_batch)
        logsigma = self.var_lin(logsigma) + sto

        return mean, logsigma

    def forward(self, data, weight = None, test = False):

        if weight != None:
            x = weight[data.sto_x - 1]
        
        x = self.gnn(x, data.sto_edge_index)
        
        weighted_x = x * data.sto_weight.reshape(-1, 1)
        sto = scatter_sum(weighted_x, data.sto_batch, dim = 0)

        mean = self.mean_set2set(weighted_x, data.sto_batch)
        mean = self.mean_lin(mean) + sto
        mean = F.normalize(mean, dim = 1)

        return self.predictor(mean)