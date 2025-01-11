# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DistrictNoirNN(nn.Module):
    def __init__(self, input_size=76):
        super().__init__()
        # Input: 196 features (see state representation)
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        # Two heads:
        # 1. Value head (estimates expected final score difference)
        self.value_head = nn.Linear(128, 1)

        # 2. Policy head (outputs probabilities for each possible action)
        # Actions: play any card from hand (5) or collect
        self.policy_head = nn.Linear(128, 7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = torch.tanh(self.value_head(x))
        policy = F.softmax(self.policy_head(x), dim=-1)

        return value, policy
