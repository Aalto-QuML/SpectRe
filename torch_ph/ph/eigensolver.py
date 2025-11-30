import torch
from torch_geometric.utils import get_laplacian, to_dense_adj
class EigenSolver(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.jit.export
    def forward(self, L, n):
        if n > 2:
            # Use PyTorch's built-in lobpcg
            k_val = int(n // 3)  # Explicitly cast to int
#            k_val = 2
            eigvals = torch.lobpcg(L, k=k_val, largest=False)[0]  # Get smallest eigenvalues
            return eigvals
        else:
            # For small matrices, use eigh
            return torch.linalg.eigh(L)[0]  # Just the eigenvalues

# Create an instance
solver = EigenSolver()

#edge_index = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6], [1, 2, 0, 2, 6, 0, 1, 4, 5, 3, 5, 3, 4, 6, 1, 5]])
#lap_sparse = get_laplacian(edge_index)
#lap = to_dense_adj(edge_index=lap_sparse[0], edge_attr=lap_sparse[1]).squeeze()
#print(lap)
#solver(lap, lap.shape[0])

# Script the model
scripted_solver = torch.jit.script(solver)

# Save for C++ usage
scripted_solver.save("eigensolver.pt")