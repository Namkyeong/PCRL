import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--embedder", type=str, default="PCRL")
    
    parser.add_argument("--sto_gnns", type=str, default="GCN")
    parser.add_argument("--sto_layers", type=int, default=3)
    parser.add_argument("--str_layers", type=int, default=3)
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default = 0.0)
    parser.add_argument("--dropout", type=float, default = 0.0)

    parser.add_argument("--es", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=256)
    
    parser.add_argument("--save_checkpoints", action = "store_true", default=False, help = 'save model parameters when True')
    parser.add_argument("--writer", action = "store_true", default=False, help = 'Tensorboard writer')
    
    parser.add_argument("--no_node_feature", action = "store_true", default=True, help = 'Do not use node feature?')
    
    if 'PCRL' in parser.parse_known_args()[0].embedder:
        parser.add_argument("--n_samples", type=int, default=8)
        parser.add_argument("--negative_scale", type=float, default = 20.0)
        parser.add_argument("--shift", type=float, default = 20.0)
        parser.add_argument("--uniform", type=float, default = 0.0)
        parser.add_argument("--vib", type=float, default = 1e-8)
        parser.add_argument("--reparameterize", type=float, default = 1.0)
    
    if 'Infomax' in parser.parse_known_args()[0].embedder:
        parser.add_argument("--tau", type=float, default = 0.2)
    
    if 'GraphCL' in parser.parse_known_args()[0].embedder:
        parser.add_argument("--p_e1", type=float, default = 0.5)
        parser.add_argument("--p_e2", type=float, default = 0.5)
        parser.add_argument("--tau", type=float, default = 0.2)

    parser.add_argument("--pooling", type = str, default = 'set2set')
    
    parser.add_argument("--normalize", action = "store_true", default=False, help = 'Use normalized vector for evaluation?')
    parser.add_argument("--clip", action = "store_true", default=False, help = 'Clip distance?')

    parser.add_argument("--grad_clip", action = "store_true", default=True, help = 'Clip gradient?')
    if parser.parse_known_args()[0].grad_clip:
        parser.add_argument("--max_grad", type=float, default = 2.0)

    # test dataset
    parser.add_argument("--dataset", type = str, default = 'ebg')

    return parser.parse_known_args()


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['eval_freq', 'patience', 'device', 'writer', "batch_size", 'str_layers',
                        'es', 'save_checkpoints', "message_passing", "no_node_feature", "pretrain", "clip",
                        'num_atom_feats', 'num_bond_feats', 'hidden_dim', "pooling", "normalize"]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


