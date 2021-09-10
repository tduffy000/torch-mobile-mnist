import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout

class MNISTModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        n_classes = 10
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 5, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.network(x)
        return F.log_softmax(x, dim=1)