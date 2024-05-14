import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm

from torch_geometric.nn import Set2Set

from embedder import embedder
from layers import GCN, GAT, GIN, GraphNetwork, MCSoftContrastiveLoss
from torch_scatter import scatter_sum

class PCRL_Trainer(embedder):
    
    def __init__(self, args, dataset):
        embedder.__init__(self, args, dataset)

        self.sto_model = Sto_Enc(args.sto_gnns, args.sto_layers, args.input_dim, args.hidden_dim, args.pooling, args.n_samples, args.reparameterize, args.clip).to(self.device)
        self.str_model = Structure(args.str_layers, args.num_atom_feats, args.num_bond_feats,
                                args.hidden_dim, args.pooling, args.no_node_feature, self.device).to(self.device)
        self.criterion = MCSoftContrastiveLoss(args).to(self.device)
        self.optimizer = optim.Adam(params = list(self.str_model.parameters())+list(self.criterion.parameters())+list(self.sto_model.parameters()), 
                                        lr = self.args.lr, weight_decay = self.args.weight_decay)

    def train(self):

        self.str_model.train()

        for epoch in range(1, self.args.epochs + 1):
            
            self.sto_model.train()
            epoch_loss = 0
            log_sigmas =[]

            for bc, samples in enumerate(tqdm(self.train_loader, leave=False)):
                
                self.optimizer.zero_grad()

                stoichiometry, logsigma = self.sto_model(samples.to(self.device), self.elem_attrs.to(self.device))
                structure = self.str_model(samples.to(self.device))
                loss = self.criterion(stoichiometry, structure, logsigma, samples.formula)

                epoch_loss += loss
                
                loss.backward()
                if self.args.grad_clip:
                    nn.utils.clip_grad_norm_(list(self.str_model.parameters())+list(self.criterion.parameters())+list(self.sto_model.parameters()), self.args.max_grad)
                else:
                    pass
                self.optimizer.step()
                
                # Get Variance stat
                log_sigmas.append(logsigma.detach().cpu().numpy())

            self.sto_model.eval()
            variance = np.vstack(log_sigmas).mean()
            max_var = np.vstack(log_sigmas).max()
            
            # Write Statistics
            self.writer.add_scalar("loss/train loss", epoch_loss/bc, epoch)
            self.writer.add_scalar("stats/variance", variance, epoch)
            self.writer.add_scalar("stats/max variance", max_var, epoch)

            shift_neg = []
            for n, p in self.criterion.named_parameters():
                shift_neg.append(p.data.item())

            self.writer.add_scalar("stats/Shift", shift_neg[0], epoch)
            self.writer.add_scalar("stats/Negative Scale", shift_neg[1], epoch)
            self.writer.add_scalar("stats/Distance", self.criterion.distance, epoch)
            self.writer.add_scalar("stats/Right pos ratio", self.criterion.right_pos, epoch)
            self.writer.add_scalar("stats/Right neg ratio", self.criterion.right_neg, epoch)
            
            st = '[Epoch {}/{}] Loss: {:.4f}'.format(epoch, self.args.epochs, epoch_loss.item())
            print(st)

        self.writer.close()
        torch.save(self.sto_model.state_dict(), self.check_dir)


class Sto_Enc(nn.Module):
    """
    This the main class for our model
    """

    def __init__(self, gcn, layers, input_dim, hidden_dim, pooling, n_samples, 
                 reparameterize, clip = False):
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

        self.n_samples = n_samples
        self.reparameterize = reparameterize
        self.clip = clip

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        
        eps = torch.normal(0, self.reparameterize, size = (mu.size(0), num_samples, mu.size(1)), dtype=mu.dtype, device=mu.device)
        samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(mu.unsqueeze(1))
        
        return samples

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

        if test:
            return mean

        else:

            logsigma = self.var_set2set(weighted_x, data.sto_batch)
            logsigma = self.var_lin(logsigma) + sto
            if self.clip:
                logsigma = torch.clip(logsigma, max = 40)
            samples = self.sample_gaussian_tensors(mean, logsigma, self.n_samples)

            return samples, logsigma


class Structure(nn.Module):
    """
    This the main class for our model
    """

    def __init__(self, num_layers, num_atom_feats, num_bond_feats, hidden_dim, pooling, no_node_feature, device):
        super(Structure, self).__init__()

        self.graphnetwork = GraphNetwork(num_layers, num_atom_feats, num_bond_feats, hidden_dim, device = device)
        self.pooling = pooling

        self.no_node_feature = no_node_feature
        if no_node_feature:            
            self.node_embedding = nn.Parameter(torch.empty((hidden_dim,)))
            nn.init.normal_(self.node_embedding)
        else:
            pass

        self.init_model()
    
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, data):

        if self.no_node_feature:
            data.x = self.node_embedding.expand(data.x.shape[0], -1)
            
        crystal, nodes = self.graphnetwork(data)
        
        return F.normalize(crystal, dim = 1)