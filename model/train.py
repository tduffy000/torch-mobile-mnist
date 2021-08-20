import argparse
import os
import time

import torch
from torch._C import dtype
import torchvision
import torch.nn.functional as F
from torchvision.datasets import mnist

from model import MNISTModel

from torch.utils.mobile_optimizer import optimize_for_mobile

DEFAULT_TARGET_PATH = './data/mnist'
ARTIFACT_PATH = './artifacts'

torch.manual_seed(42)

def get_loaders(target_path):
    mnist_mean, mnist_std = 0.1307, 0.3081
    mnist_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((mnist_mean,), (mnist_std,))
    ])

    training_set = torchvision.datasets.MNIST(
        root=target_path,
        download=True,
        train=True,
        transform=mnist_transform
    )

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)

    test_set = torchvision.datasets.MNIST(
        root=target_path,
        download=True,
        train=True,
        transform=mnist_transform
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)
    return train_loader, test_loader

def train_epoch(model, optim, loader):
    
    log_batch_interval = 500

    n = len(loader.dataset)
    model.train()

    for i, (X, y) in enumerate(loader, start=1):
        optim.zero_grad()
        output = model(X)
        loss = F.nll_loss(output, y)
        loss.backward()
        optim.step()

        if i % log_batch_interval == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i * len(X), n,
                100. * i / len(loader), loss.item()))

def evaluate(model, loader):
    model.eval()
    n = len(loader.dataset)
    n_correct, total_loss = 0, 0

    with torch.no_grad():
        for X, y in loader:
            output = F.log_softmax(model(X), dim=1)
            loss = F.nll_loss(output, y, reduction='sum')
            _, y_pred = torch.max(output, dim=1)

            total_loss += loss.item()
            n_correct += y_pred.eq(y).sum()
    
    print('Accuracy: ' + '{:4.2f}'.format(100. * n_correct / n) + "%\n")
    
def store_models(model):
    ts = int(time.time())
    models_path = os.path.join(ARTIFACT_PATH, str(ts))
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    full_model_path = os.path.join(models_path, 'full_model.pt')
    torch.save(model, full_model_path)
    model = torch.load(full_model_path)
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8
    )
    input = torch.zeros(1, 1, 28, 28)
    ts_model = torch.jit.trace(quantized_model, input)
    optim_model = optimize_for_mobile(ts_model)
    optim_model_path = os.path.join(models_path, 'mobile_model.pt')
    optim_model.save(optim_model_path)

def run(args):
    
    model = MNISTModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(1, args.epochs + 1):
        print(f'Epoch: {i}...\n')
        train_loader, test_loader = get_loaders(args.download_path)
        train_epoch(model, optimizer, train_loader)
        evaluate(model, test_loader)

    if args.freeze:
        store_models(model)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='The number of training epochs.')
    parser.add_argument('--lr', type=float, help='The learning rate.')
    parser.add_argument('--download_path', type=str, default=DEFAULT_TARGET_PATH)
    parser.add_argument('--freeze', default=False, action='store_true')
    args = parser.parse_args()

    run(args)

