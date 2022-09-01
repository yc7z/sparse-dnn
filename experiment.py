from math import ceil
import torch
import torch.nn as nn
import time


class Experiment():
    def __init__(self, model):
        self.model = model
    
    def set_up_exp(self):
        pass

    def run(self):
        pass


class SparseBenchmarkClassifier(Experiment):
    def __init__(self, model):
        super().__init__(model)
    
    
    def set_up_exp(self, args, trainloader, testloader, optimizer, criterion):
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.criterion = criterion
        self.sparsity_level = args.sparsity_level
    
    
    def train_epoch(self):
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
    
    def test_accuracy(self):
        self.model.eval()
        correct_pred = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            outputs = self.model(inputs)
            predictions = outputs.argmax(
                dim=1, keepdim=True
            )
            correct_pred += predictions.eq(targets.view_as(predictions)).sum().item()
        
        return correct_pred / len(self.testloader.dataset)            
            
    
    def sparsify(self, sparsity_level):
        """
        sparse_level is a positive real number between 0 to 1.
        components whose norm fall into the top sparse_level percentile will be kept,
        and the rest components will be set to zero.
        """
        for param in self.model.parameters():
            p_flat = param.flatten()
            k = ceil(len(p_flat) * sparsity_level)
            topk_vals, topk_inds = torch.topk(input=torch.abs(p_flat), k=k)
            mask = torch.zeros(size=p_flat.shape, device=topk_inds.device)
            mask.scatter_(0, topk_inds, 1, reduce='add')
            param *= mask.reshape(param.size())
    
    
    def reset_model(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.model.apply(weight_reset)

    
    def run_sparse(self):
        interm_sparsity_level = self.sparsity_level ** (1/self.epochs)
        sparse_accuracies = []
        # dense_accuracies = []
        # for _ in range(1, self.epochs + 1):
        #     self.train_epoch()
        #     dense_accuracies.append(self.test_accuracy())
        
        # with torch.no_grad():
        #     self.reset_model()
            
        for _ in range(1, self.epochs + 1):
            self.train_epoch()
            sparse_accuracies.append(self.test_accuracy())
            with torch.no_grad():
                self.sparsify(interm_sparsity_level)
                interm_sparsity_level *= self.sparsity_level
        
        # with torch.no_grad():
        #     self.reset_model()
        
        # for _ in range(1, self.epochs + 1):
        #     self.train_epoch()
        #     dense_accuracies.append(self.test_accuracy())
        return sparse_accuracies

    
    def run_dense(self):
        dense_accuracies = []
        for _ in range(1, self.epochs + 1):
            self.train_epoch()
            dense_accuracies.append(self.test_accuracy())
        return dense_accuracies



class SparseTimingBenchmark(Experiment):
    def __init__(self, model):
        super().__init__(model)
        
    
    def set_up_exp(self, epochs, trainloader, testloader, lr, criterion, optimizer):
        self.trainloader = trainloader
        self.testloader = testloader
        self.lr = lr
        self.epochs = epochs
        self.criterion = criterion  
        self.optimizer = optimizer  
        
    
    def train_epoch(self):
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            # self.optimizer.step()
            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.lr * param.grad
    
    
    def test_accuracy(self):
        self.model.eval()
        correct_pred = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            outputs = self.model(inputs)
            predictions = outputs.argmax(
                dim=1, keepdim=True
            )
            correct_pred += predictions.eq(targets.view_as(predictions)).sum().item()
        
        return correct_pred / len(self.testloader.dataset)
    
    
    def run(self):
        start = time.perf_counter()
        for epoch in range(self.epochs):
            self.train_epoch()
            print(self.test_accuracy())
        end = time.perf_counter()
        print(f'total time of experiment: {end - start}s')
        
    
                
    