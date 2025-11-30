import torch.nn as nn
import torch.nn.functional as F

from layers.gps_layer import GPS
from layers.graph_convolution_layer import GCNLayer
from models.gnn_factory.gnn_factory_interface import GNNFactoryInterface


class GpsCreator(GNNFactoryInterface):
    def __init__(self, hidden_dim, batch_norm, attention_type):
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm
        self.attention_type = attention_type

    def return_gnn_instance(self, is_last=False):
        return GPS(
            in_channels=self.hidden_dim,
            # num_node_features if is_first else hidden_dim,
            out_channels=self.hidden_dim,
            # num_classes if is_last else hidden_dim,
            pe_dim=8,
            num_layers=2,
            attn_type=self.attention_type
        )
