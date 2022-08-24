import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLinear(nn.Module):
    def __init__(self, shape):
        super(SparseLinear, self).__init__()
        self.sparse_linear = nn.Parameter(torch.randn(shape).to_sparse().requires_grad_(True))

    def forward(self, x):
        return torch.sparse.mm(self.sparse_linear, x)


class SimpleSparseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = SparseLinear((512, 28 * 28))
        self.fc2 = SparseLinear((512, 512))
        self.fc3 = SparseLinear((10, 512))
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.transpose(0, 1).reshape(-1, batch_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.T
        return x


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

    

