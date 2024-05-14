import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torch_geometric.nn import Set2Set
from torch_geometric.utils import dropout_adj

from embedder import embedder
from layers import GCN, GAT, GIN

from tqdm import tqdm

class GraphCL_Trainer(embedder):
    
    def __init__(self, args, dataset):
        embedder.__init__(self, args, dataset)

        self.sto_model = Sto_Enc(args.sto_gnns, args.sto_layers, args.input_dim, args.hidden_dim, args.pooling).to(self.device)
        self.optimizer = optim.Adam(params = self.sto_model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
    
    def augmentation(self, data):

        edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, p=self.args.p_e1)
        edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, p=self.args.p_e2)

        new_data1, new_data2 = data.clone(), data.clone()
        new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
        new_data1.edge_attr, new_data2.edge_attr = edge_attr1, edge_attr2

        return new_data1, new_data2

    def contrastive_loss(self, crystal, elements):

        batch_size, _ = crystal.size()
        crystal_abs = crystal.norm(dim = 1)
        elements_abs = elements.norm(dim = 1) + 1e-6

        sim_matrix = torch.einsum('ik,jk->ij', crystal, elements) / torch.einsum('i,j->ij', crystal_abs, elements_abs)
        sim_matrix = torch.exp(sim_matrix / self.args.tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

    def train(self):

        for epoch in range(1, self.args.epochs + 1):
            
            self.sto_model.train()

            epoch_loss = 0

            for bc, samples in enumerate(tqdm(self.train_loader, leave=False)):
                
                self.optimizer.zero_grad()
                samples1, samples2 = self.augmentation(samples)

                stoichiometry1 = self.sto_model(samples1.to(self.device), self.elem_attrs.to(self.device))
                stoichiometry2 = self.sto_model(samples2.to(self.device), self.elem_attrs.to(self.device))
                loss = self.contrastive_loss(stoichiometry1, stoichiometry2)

                epoch_loss += loss
                
                loss.backward()
                if self.args.grad_clip:
                    nn.utils.clip_grad_norm_(self.sto_model.parameters(), self.args.max_grad)
                else:
                    pass
                self.optimizer.step()
                
            self.sto_model.eval()
            
            # Write Statistics
            self.writer.add_scalar("loss/train loss", epoch_loss/bc, epoch)            
            st = '[Epoch {}/{}] Loss: {:.4f}'.format(epoch, self.args.epochs, epoch_loss.item())
            print(st)

        self.writer.close()
        torch.save(self.sto_model.state_dict(), self.check_dir)
        

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

        self.num_layers = layers
        self.pooling = pooling

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, weight = None, test = False):

        if weight != None:
            x = weight[data.sto_x - 1]
        
        x = self.gnn(x, data.sto_edge_index)
        
        weighted_x = x * data.sto_weight.reshape(-1, 1)
        mean = self.mean_set2set(weighted_x, data.sto_batch)
        mean = self.mean_lin(mean)

        return mean