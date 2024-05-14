import torch
import torch.nn.functional as F
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

import datetime
import os

from argument import config2string

from util.chem import *

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

os.environ['OMP_NUM_THREADS'] = "2"


class embedder:

    def __init__(self, args, dataset):
        self.args = args
        
        d = datetime.datetime.now()
        date = d.strftime("%x")[-2:] + d.strftime("%x")[0:2] + d.strftime("%x")[3:5]
        
        self.config_str = "{}_".format(date) + config2string(args)
        print("\n[Config] {}\n".format(self.config_str))

        # Define Tensorboard Writer
        self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))

        # Model Checkpoint Path
        CHECKPOINT_PATH = "model_checkpoints/{}/".format(args.embedder)
        os.makedirs(CHECKPOINT_PATH, exist_ok=True) # Create directory if it does not exist
        self.check_dir = CHECKPOINT_PATH + self.config_str + ".pt"

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        # Load initial points
        self.elem_attrs = load_elem_attrs('res/matscholar-embedding.json')

        # Set Arguments
        args.num_atom_feats = dataset[0].x.shape[1]
        args.num_bond_feats = dataset[0].edge_attr.shape[1]
        args.input_dim = args.hidden_dim
        
        self.train_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True)