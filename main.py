from cProfile import label
from cmath import exp
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import torchvision.transforms as transforms
import models
from experiment import *
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")

    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--lr", default=4e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--sparsity_level", default=0.5, type=float)
    
    args = parser.parse_args()
    
    transform = transforms.Compose(
    [
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f'training device: {device}')
    print(f'training set size: {len(trainset)}')
    
    model = models.LeNet5(num_classes=10)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    experiment = SparseBenchmarkClassifier(model)
    experiment.set_up_exp(args, trainloader, testloader, optimizer, criterion)
    # sparse_accuracies = experiment.run_sparse()
    dense_accuracies = experiment.run_dense()
    print(dense_accuracies)
    
    # epochs_lst = [_ for _ in range(1, args.epochs + 1)]
    # plt.legend(loc="lower center", bbox_to_anchor=(0.8, 0.25))
    # plt.savefig('./sparse_acc_plot(1)')
