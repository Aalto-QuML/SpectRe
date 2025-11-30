import os.path as osp

import torch
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import ZINC
from torch_geometric.utils import degree


class FilterConstant(object):
  def __init__(self, dim):
    self.dim = dim

  def __call__(self, data):
    data.x = torch.ones(data.num_nodes, self.dim)
    return data


def get_tudataset(name, pre_transform=None, cleaned=False):
  path = osp.join(osp.dirname(osp.realpath(__file__)), '')
  dataset = TUDataset(path, name, pre_transform=pre_transform, cleaned=cleaned)

  if not hasattr(dataset, 'x'):
    max_degree = 0
    degs = []
    for data in dataset:
      degs += [degree(data.edge_index[0], dtype=torch.long)]
      max_degree = max(max_degree, degs[-1].max().item())
    dataset.transform = T.OneHotDegree(max_degree)
  return dataset


def data_split(dataset, seed=42):
  skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
  train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.y))[0]
  skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
  val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.y[val_test_idx]))[0]
  train_data = dataset[train_idx]
  val_data = dataset[val_test_idx[val_idx]]
  test_data = dataset[val_test_idx[test_idx]]
  return train_data, val_data, test_data

def add_attributes(dataset, features):
  data_list = []
  for i, data in enumerate(dataset):
    data.graph_features = features[i, :].unsqueeze(0)
    data_list.append(data)
  dataset.data, dataset.slices = dataset.collate(data_list)
  return dataset

def get_data(name, pre_transform=None, seed=42):
  if name == 'ZINC':
    train_data, val_data, test_data = get_zinc(pre_transform=pre_transform)
    num_classes = 1
  elif name == 'ogbg-molhiv':
    train_data, val_data, test_data = get_molhiv(pre_transform=pre_transform)
    num_classes = 2
  else:
    data = get_tudataset(name, pre_transform=pre_transform)
    num_classes = data.num_classes
    train_data, val_data, test_data = data_split(data, seed=seed)

  stats = dict()
  stats['num_features'] = train_data.num_node_features
  stats['num_classes'] = num_classes

  return train_data, val_data, test_data, stats


def get_zinc(pre_transform=None):
  path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ZINC')
  train_data = ZINC(path, subset=True, split='train', pre_transform=pre_transform)
  data_val = ZINC(path, subset=True, split='val', pre_transform=pre_transform)
  data_test = ZINC(path, subset=True, split='test', pre_transform=pre_transform)

  return train_data, data_val, data_test

def get_molhiv(pre_transform=None):
  path = osp.dirname(osp.realpath(__file__))
  dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=path, pre_transform=pre_transform)
  split_idx = dataset.get_idx_split()
  return dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]

