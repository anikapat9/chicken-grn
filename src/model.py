import torch
import torch.nn as nn
from torchdiffeq import odeint

class GRNODE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, t, x):
        return self.net(x)  # dx/dt = f(x)

def train(model, dataloader, epochs=1000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            pred = odeint(model, batch[0], torch.tensor([0.0, 1.0]))  # Time points
            loss = torch.mean((pred[-1] - batch[1])**2)  # MSE between pred and target
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
