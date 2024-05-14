import torch
import torch_geometric
from torch_geometric.data import Data


class Custom_Data(Data):
    """A plain old python object modeling a single graph with various
    (optional) attributes:
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, 
                 elem_nums = None, elem_batch = None, 
                 sto_x = None, sto_edge_index = None,
                 y=None, pos=None, normal=None, face=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.sto_x = sto_x
        self.sto_edge_index = sto_edge_index
        self.elem_nums = elem_nums
        self.elem_batch = elem_batch
        self.unique_elems = len(sto_x)
        self.y = y
        self.pos = pos
        self.normal = normal
        self.face = face
        
        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and edge_index.dtype != torch.long:
            raise ValueError(
                (f'Argument `edge_index` needs to be of type `torch.long` but '
                 f'found type `{edge_index.dtype}`.'))

        if face is not None and face.dtype != torch.long:
            raise ValueError(
                (f'Argument `face` needs to be of type `torch.long` but found '
                 f'type `{face.dtype}`.'))

        if torch_geometric.is_debug_enabled():
            self.debug()

    def __inc__(self, key, value, *args, **kwargs):

        if (key == 'elem_batch') or (key == 'sto_batch'):
            return 1
        if (key == 'sto_edge_index') or (key == 'sto_elem_index') or (key =='elem_sto_index') :
            return self.unique_elems
        else:
            return super().__inc__(key, value, *args, **kwargs)
        

class Custom_Test_Data(Data):
    """A plain old python object modeling a single graph with various
    (optional) attributes:
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, 
                 sto_x = None, sto_edge_index = None,
                 y=None, pos=None, normal=None, face=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.sto_x = sto_x
        self.sto_edge_index = sto_edge_index
        self.unique_elems = len(sto_x)
        self.y = y
        self.pos = pos
        self.normal = normal
        self.face = face
        
        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and edge_index.dtype != torch.long:
            raise ValueError(
                (f'Argument `edge_index` needs to be of type `torch.long` but '
                 f'found type `{edge_index.dtype}`.'))

        if face is not None and face.dtype != torch.long:
            raise ValueError(
                (f'Argument `face` needs to be of type `torch.long` but found '
                 f'type `{face.dtype}`.'))

        if torch_geometric.is_debug_enabled():
            self.debug()

    def __inc__(self, key, value, *args, **kwargs):

        if (key == 'elem_batch') or (key == 'sto_batch'):
            return 1
        if key == 'sto_edge_index':
            return self.unique_elems
        else:
            return super().__inc__(key, value, *args, **kwargs)