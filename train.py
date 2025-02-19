import yaml
import torch
from src.data_loader import RNADataset
from src.model import GRNODE, train
from torch.utils.data import DataLoader

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize dataset and model
dataset = RNADataset(config["data"]["path"])
dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
model = GRNODE(config["model"]["input_dim"], config["model"]["hidden_dim"])

# Train and save
train(model, dataloader, epochs=config["training"]["epochs"], lr=config["training"]["lr"])
torch.save(model.state_dict(), "models/grn_ode.pth")
