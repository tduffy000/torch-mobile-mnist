import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout

class MNISTModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        n_classes = 10
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.3),
            torch.nn.Flatten(),
            torch.nn.Linear(4608, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.network(x)
        return F.log_softmax(x, dim=1)