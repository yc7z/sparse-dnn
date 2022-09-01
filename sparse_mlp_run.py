from cProfile import label
from cmath import exp
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import torchvision.transforms as transforms
import sparse_mlp
from experiment import *
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--lr", default=4e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--sparsity_level", default=0.5, type=float)
    
    args = parser.parse_args()
    
    transform = transforms.Compose(
    [
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='./datasets', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.MNIST(root='./datasets', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=True)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f'training device: {device}')
    print(f'training set size: {len(trainset)}')
    
    model = sparse_mlp.SimpleSparseMLP(args.sparsity_level)
    # model = sparse_mlp.SimpleMLP()
    
    # optimizer = torch.optim.SparseAdam(params=model.parameters(), lr=args.lr)
    optimizer = None
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    experiment = SparseTimingBenchmark(model)
    experiment.set_up_exp(args.epochs, trainloader, testloader, args.lr, criterion, optimizer)
    experiment.run()
