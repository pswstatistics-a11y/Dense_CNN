import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


class DenseCNN(nn.Module):
    def __init__(self, dims = 64):
        super(DenseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(8 * 8 * 256, dims)
        self.fc_classifier = nn.Linear(dims, 10)
        
    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        dense_bridge = torch.cat([h1, h2], dim=1)
        out = self.pool(F.relu(self.conv3(dense_bridge)))
        out = self.pool(F.relu(self.conv4(out)))

        flattened = out.view(out.size(0), -1)

        latent_features = self.fc1(flattened)
        latent_activated = F.relu(latent_features)
        logits = self.fc_classifier(latent_activated)
        
        return logits, latent_features
    
