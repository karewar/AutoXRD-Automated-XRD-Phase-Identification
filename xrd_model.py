import torch
import torch.nn as nn

class XRDNet(nn.Module):
    def __init__(self, n_points=200, n_phases=5):
        super(XRDNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * n_points, 128)
        self.fc_phase = nn.Linear(128, n_phases)       # Phase classification
        self.fc_abundance = nn.Linear(128, n_phases)   # Abundance regression

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        phase_out = torch.sigmoid(self.fc_phase(x))
        abundance_out = torch.relu(self.fc_abundance(x))
        return phase_out, abundance_out
