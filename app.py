"""
AutoXRD Prototype
AI-powered Phase & Composition Identification from XRD
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1. Simulated Dataset
# --------------------------
class XRDDataset(Dataset):
    def __init__(self, n_samples=1000, n_points=200):
        np.random.seed(42)
        self.X = np.random.rand(n_samples, n_points)  # Simulated XRD intensities
        self.y = np.random.randint(0, 2, size=(n_samples, 5))  # 5 possible phases (multi-label)
        self.y_abundance = np.random.rand(n_samples, 5)  # Abundance of each phase

        # Optional: normalize input
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32), \
               torch.tensor(self.y_abundance[idx], dtype=torch.float32)


# --------------------------
# 2. Simple 1D CNN Model
# --------------------------
class XRDNet(nn.Module):
    def __init__(self, n_points=200, n_phases=5):
        super(XRDNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * n_points, 128)
        self.fc_phase = nn.Linear(128, n_phases)       # Multi-label phase classification
        self.fc_abundance = nn.Linear(128, n_phases)   # Abundance regression

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        phase_out = torch.sigmoid(self.fc_phase(x))  # Multi-label
        abundance_out = torch.relu(self.fc_abundance(x))  # Non-negative
        return phase_out, abundance_out


# --------------------------
# 3. Training Loop
# --------------------------
def train_model():
    dataset = XRDDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = XRDNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_phase = nn.BCELoss()          # Multi-label classification
    criterion_abundance = nn.MSELoss()      # Regression for abundance

    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_phase, y_abundance in dataloader:
            optimizer.zero_grad()
            phase_pred, abundance_pred = model(X_batch)
            loss_phase = criterion_phase(phase_pred, y_phase)
            loss_abundance = criterion_abundance(abundance_pred, y_abundance)
            loss = loss_phase + loss_abundance
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model

# --------------------------
# 4. Inference + Visualization
# --------------------------
def predict_and_plot(model, sample_idx=0):
    dataset = XRDDataset()
    X_sample, y_phase, y_abundance = dataset[sample_idx]
    with torch.no_grad():
        phase_pred, abundance_pred = model(X_sample.unsqueeze(0))
    
    x_axis = np.arange(len(X_sample))
    plt.figure(figsize=(10,4))
    plt.plot(x_axis, X_sample.numpy(), label="XRD pattern")
    plt.title("Simulated XRD Pattern")
    plt.xlabel("2θ index")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()
    
    print("Predicted phases (probabilities):", phase_pred.numpy()[0])
    print("Predicted abundances:", abundance_pred.numpy()[0])

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    model = train_model()
    predict_and_plot(model)
