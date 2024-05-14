import argparse
import numpy as np
import os

from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from layers import Sto_Enc
from util.chem import *
from util.data import get_k_folds
from argument import config2string

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sto_gnns", type=str, default="GCN")
    parser.add_argument("--sto_layers", type=int, default=3)
    parser.add_argument("--predictor", type=int, default=3)
    parser.add_argument("--pooling", type = str, default = 'set2set')

    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type = str, default = 'ebg')
    parser.add_argument("--model", type = str, default = 'PCRL')

    parser.add_argument("--save_checkpoints", action = "store_true", default=False, help = 'Save model predictions?')

    return parser.parse_known_args()

args, unknown = parse_args()

config = config2string(args)
print(config)
writer = SummaryWriter(log_dir="runs_finetune/single_{}".format(config))

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
random_seed = args.seed
n_folds = 5
n_epochs = args.epochs
dataset = list()
eval_r2 = list()
eval_mae = list()
fold_ys = list()
fold_preds = list()

args.rand_init = False
if args.model == "PCRL":
    check_dir = "./model_checkpoints/PCRL_vib_1e-8.pt"
elif args.model == "Infomax":
    check_dir = "./model_checkpoints/Infomax.pt"
elif args.model == "MPBandG":
    check_dir = "./model_checkpoints/MPBandG.pt"
elif args.model == "MPFormE":
    check_dir = "./model_checkpoints/MPFormE.pt"
elif args.model == "GraphCL":
    check_dir = "./model_checkpoints/GraphCL.pt"
else:
    args.rand_init = True


if args.dataset == "ebg":
    dataset = torch.load("./dataset/processed/ebg_test_data.pt")    
    batchsize = 128
elif args.dataset == "efe":
    dataset = torch.load("./dataset/processed/efe_test_data.pt")
    batchsize = 128
elif args.dataset == "metallic":
    dataset = torch.load("./dataset/processed/metallic_test_data.pt")
    batchsize = 128
elif args.dataset == "lec300k":
    dataset = torch.load("./dataset/processed/lec300k_test_data.pt")
    batchsize = 32
elif args.dataset == "lec600k":
    dataset = torch.load("./dataset/processed/lec600k_test_data.pt")
    batchsize = 32
elif args.dataset == "ltc300k":
    dataset = torch.load("./dataset/processed/ltc300k_test_data.pt")
    batchsize = 32
elif args.dataset == "ltc600k":
    dataset = torch.load("./dataset/processed/ltc600k_test_data.pt")
    batchsize = 32
elif args.dataset == "seebeck300k":
    dataset = torch.load("./dataset/processed/seebeck300k_test_data.pt")
    batchsize = 32
elif args.dataset == "seebeck600k":
    dataset = torch.load("./dataset/processed/seebeck600k_test_data.pt")
    batchsize = 32



k_folds = get_k_folds(dataset, k=n_folds, random_seed=random_seed)
for k in range(0, n_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]

    train_loader = DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batchsize)

    model = Sto_Enc(gcn = args.sto_gnns, layers = args.sto_layers, input_dim = 200, hidden_dim = 200, pooling = args.pooling)        

    if not args.rand_init: # Call pretrained weights
        model.load_state_dict(torch.load(check_dir, map_location = device), strict=False)
    else: # Random Initialze
        pass

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    criterion = torch.nn.L1Loss()
    for epoch in tqdm(range(0, n_epochs)):

        model.train()

        for bc, samples in enumerate(train_loader):
            
            optimizer.zero_grad()            
            
            preds = model(samples.to(device), elem_attrs.to(device))
            loss = criterion(samples.y, preds.reshape(-1))

            loss.backward()
            optimizer.step()

        if epoch % args.eval_freq == 0:
            
            model.eval()
            ys, preds =[], []
            for bc, samples in enumerate(test_loader):
                pred = model(samples.to(device), elem_attrs.to(device))
                ys.append(samples.y.detach().cpu().numpy())
                preds.append(pred.detach().cpu().numpy())
            ys = np.hstack(ys)
            preds = np.vstack(preds).reshape(-1)
            writer.add_scalar("loss/test R2 Fold {}".format(k), r2_score(ys, preds), epoch)
            writer.add_scalar("loss/test MSE Fold {}".format(k), mean_absolute_error(ys, preds), epoch)
    
    model.eval()
    ys, preds =[], []
    for bc, samples in enumerate(test_loader):
        pred = model(samples.to(device), elem_attrs.to(device))
        ys.append(samples.y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
    ys = np.hstack(ys)
    preds = np.vstack(preds).reshape(-1)

    eval_r2.append(r2_score(ys, preds))
    eval_mae.append(mean_absolute_error(ys, preds))

    fold_ys.append(ys)
    fold_preds.append(preds)

if args.save_checkpoints:
    SAVE_PATH = "pred_saves/{}/".format(args.dataset)
    os.makedirs(SAVE_PATH, exist_ok=True) # Create directory if it does not exist
    save_dir = SAVE_PATH + config + ".pt"
    torch.save([fold_ys, fold_preds], save_dir)

# Write experimental results
WRITE_PATH = "results_rep/"
os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
f = open("results_rep/{}.txt".format(args.dataset), "a")
f.write("--------------------------------------------------------------------------------- \n")
f.write("{}".format(config))
f.write("\n")
f.write("[lr_{}] R2: {:.4f} ({:.4f}) / MAE: {:.4f} ({:.4f})".format(args.lr, numpy.mean(eval_r2), numpy.std(eval_r2), numpy.mean(eval_mae), numpy.std(eval_mae)))
f.write("\n")
f.write("--------------------------------------------------------------------------------- \n")
f.close()