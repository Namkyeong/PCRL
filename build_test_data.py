import numpy
import pandas as pd

import torch
import json
from chemparse import parse_formula
from data import Custom_Test_Data
import os

atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
             'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
             'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
             'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
             'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
             'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
             'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
             'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
             'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
             'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}
atom_syms = {v: k for k, v in atom_nums.items()}
elem_feat_names = ['atomic_number', 'period', 'en_pauling', 'covalent_radius_bragg',
                   'electron_affinity', 'atomic_volume', 'atomic_weight', 'fusion_heat']
n_elem_feats = len(elem_feat_names) + 1
n_bond_feats = 32


def load_json_dataset(meta_data, target):

    if target == "ebg":
        target = "exp_band_gap"
    elif target == "efe":
        target = "exp_form_eng"

    dataset = []

    for idx, d in enumerate(meta_data.keys()):

        parsed_formula = parse_formula(d)

        fully_connected = []
        for j in range(len(parsed_formula.keys())):
            for k in range(len(parsed_formula.keys())):
                fully_connected.append(numpy.asarray([j, k]))

        sto_x = torch.tensor([atom_nums[i] for i in list(parsed_formula.keys())], dtype = torch.long)
        sto_batch = torch.zeros_like(sto_x)
        sto_bonds = torch.tensor(numpy.vstack(fully_connected).T)

        sto_weight = []
        for i in range(len(sto_x)):
            sto_weight.append(parsed_formula[list(atom_nums.keys())[list(atom_nums.values()).index(sto_x[i])]])
        sto_weight = torch.tensor(sto_weight, dtype=torch.float)

        y = numpy.mean(meta_data[d][target])
        data = Custom_Test_Data(sto_x = sto_x, sto_edge_index = sto_bonds, sto_batch = sto_batch, formula = d, sto_weight = sto_weight, y = y, idx = idx)
        dataset.append(data)
    
    return dataset


def load_pandas_dataset(meta_data, target, scaler = None):

    if target == "metallic":
        col_name = "Material compositions"
        target = "reduced_glass_transition_temperature"
    
    elif target == "lec":
        col_name = "Formula"
        target = "log_electrical_conductivity"
    
    elif target == "ltc":
        col_name = "Formula"
        target = "log_thermal_conductivity"
    
    elif target == "seebeck":
        col_name = "Formula"
        target = "seebeck_coefficient(μV/K)"
    
    dataset = []
    error = 0

    max_val = meta_data[target].max()
    min_val = meta_data[target].min()

    for idx, d in enumerate(meta_data[col_name]):

        try:
            parsed_formula = parse_formula(d)

            fully_connected = []
            for j in range(len(parsed_formula.keys())):
                for k in range(len(parsed_formula.keys())):
                    fully_connected.append(numpy.asarray([j, k]))

            sto_x = torch.tensor([atom_nums[i] for i in list(parsed_formula.keys())], dtype = torch.long)
            sto_batch = torch.zeros_like(sto_x)
            sto_bonds = torch.tensor(numpy.vstack(fully_connected).T)

            sto_weight = []
            for i in range(len(sto_x)):
                sto_weight.append(parsed_formula[list(atom_nums.keys())[list(atom_nums.values()).index(sto_x[i])]])
            sto_weight = torch.tensor(sto_weight, dtype=torch.float)

            if scaler == "log":
                y = numpy.log(meta_data.iloc[idx][target]) # For log scaling
            elif scaler == "min-max":
                y = (meta_data.iloc[idx][target] - min_val) / (max_val - min_val) # For min-max scaling
            else:
                y = meta_data.iloc[idx][target]
                
            data = Custom_Test_Data(sto_x = sto_x, sto_edge_index = sto_bonds, sto_batch = sto_batch, formula = d, sto_weight = sto_weight, y = y, idx = idx)
            dataset.append(data)
        except:
            error += 1
    
    print("Total Error: {}".format(error))

    return dataset


def load_matbench_dataset(meta_data):
    
    dataset = []

    for idx, d in enumerate(meta_data["formula"]):

        parsed_formula = parse_formula(d)

        fully_connected = []
        for j in range(len(parsed_formula.keys())):
            for k in range(len(parsed_formula.keys())):
                fully_connected.append(numpy.asarray([j, k]))

        sto_x = torch.tensor([atom_nums[i] for i in list(parsed_formula.keys())], dtype = torch.long)
        sto_batch = torch.zeros_like(sto_x)
        sto_bonds = torch.tensor(numpy.vstack(fully_connected).T)

        sto_weight = []
        for i in range(len(sto_x)):
            sto_weight.append(parsed_formula[list(atom_nums.keys())[list(atom_nums.values()).index(sto_x[i])]])
        sto_weight = torch.tensor(sto_weight, dtype=torch.float)

        y = numpy.log(meta_data.iloc[idx]["target"])
        data = Custom_Test_Data(sto_x = sto_x, sto_edge_index = sto_bonds, sto_batch = sto_batch, formula = d, sto_weight = sto_weight, y = y)
        dataset.append(data)
    
    return dataset



if __name__ == "__main__":
    
    ## For band gap dataset
    target = "ebg"
    with open("./dataset/{}/metadata.json".format(target), "r") as f:
        meta_data = json.load(f)
    
    dataset = load_json_dataset(meta_data, target)

    SAVE_PATH = "./dataset/processed/"
    os.makedirs(SAVE_PATH, exist_ok=True)
    torch.save(dataset, SAVE_PATH + "ebg_test_data.pt")

    ## For Metallic dataset
    # target = "metallic"
    # df = pd.read_excel("./dataset/raw/metallic_glass_temp.xlsx")
    # dataset = load_pandas_dataset(df, target)

    ## For ESTM 300K Electrical Conductivity dataset
    # target = "lec"
    # df = pd.read_excel("./dataset/raw/estm_300k.xlsx")
    # dataset = load_pandas_dataset(df, target, "log")

    ## For ESTM 300K Thermal Conductivity dataset
    # target = "ltc"
    # df = pd.read_excel("./dataset/raw/estm_300k.xlsx")
    # dataset = load_pandas_dataset(df, target, "log")

    ## For ESTM 300K Seebeck Coefficient dataset
    # target = "seebeck"
    # df = pd.read_excel("./dataset/raw/estm_300k.xlsx")
    # dataset = load_pandas_dataset(df, target, "min-max")