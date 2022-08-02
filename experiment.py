from math import ceil
import torch


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
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    
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
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    
    def run(self):
        interm_sparsity_level = self.sparsity_level ** (1/self.epochs)
        sparse_accuracies = []
        dense_accuracies = []
        for _ in range(1, self.epochs + 1):
            self.train_epoch()
            sparse_accuracies.append(self.test_accuracy(self.testloader))
            with torch.no_grad():
                self.sparsify(interm_sparsity_level)
                interm_sparsity_level *= self.sparsity_level
        
        self.reset_model()
        
        for _ in range(1, self.epochs + 1):
            self.train_epoch()
            dense_accuracies.append(self.test_accuracy(self.testloader))
        
        return sparse_accuracies, dense_accuracies
