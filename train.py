import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from xrd_model import XRDNet

# Simulated Dataset
class XRDDataset(Dataset):
    def __init__(self, n_samples=100, n_points=200):
        np.random.seed(42)
        self.X = np.random.rand(n_samples, n_points)
        self.y_phase = np.random.randint(0, 2, (n_samples, 5))
        self.y_abundance = np.random.rand(n_samples, 5)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y_phase[idx], dtype=torch.float32), \
               torch.tensor(self.y_abundance[idx], dtype=torch.float32)

# Training
dataset = XRDDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = XRDNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_phase = nn.BCELoss()
criterion_abundance = nn.MSELoss()

epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_phase, y_abundance in loader:
        optimizer.zero_grad()
        phase_pred, abundance_pred = model(X_batch)
        loss = criterion_phase(phase_pred, y_phase) + criterion_abundance(abundance_pred, y_abundance)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/xrdnet.pth")
print("Model saved to models/xrdnet.pth")
