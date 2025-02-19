import torch
import scanpy as sc
from src.model import GRNODE
from src.utils import plot_network

# Load trained model
model = GRNODE(input_dim=1000, hidden_dim=128)
model.load_state_dict(torch.load("models/grn_ode.pth"))

# Generate adjacency matrix (example)
with torch.no_grad():
    dummy_input = torch.randn(1, 1000)  # 1 sample × 1000 genes
    interaction_weights = model.net[0].weight.detach().numpy()  # First layer weights

# Plot
adata = sc.read("data/processed/GSE65938_processed.h5ad")
plot_network(interaction_weights, gene_names=adata.var_names)
