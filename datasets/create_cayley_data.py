import argparse

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from reproducibility.utils import set_seeds

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, choices=["minCayleyGraphs12Vertices", "minCayleyGraphs16Vertices",
                                    "minCayleyGraphs20Vertices", "minCayleyGraphs24Vertices",
                                    "minCayleyGraphs32Vertices", "minCayleyGraphs36Vertices",
                                    "minCayleyGraphs60Vertices", "minCayleyGraphs63Vertices"],
    default="minCayleyGraphs24Vertices"
)

parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

set_seeds(args.seed)
graphs = nx.read_graph6(f"./cayley/{args.dataset}.g6")
dataset = []
for i, g in enumerate(graphs):
    pyg_graph = from_networkx(g)
    x = torch.ones(len(g.nodes))
    new_g = Data(x=x.unsqueeze(1), edge_index=pyg_graph.edge_index)
    dataset.append(new_g)
print('Number of graphs:', len(graphs))
print('Number of nodes:', len(graphs[0].nodes))
torch.save(dataset, f"./cayley/{args.dataset}_seed-{args.seed}.dat")
