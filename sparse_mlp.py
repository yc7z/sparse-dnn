import torch
import math
from math import ceil
import torch.nn as nn
import torch.nn.functional as F

class SparseLinear(nn.Module):
    def __init__(self, shape, sparsity_level):
        super(SparseLinear, self).__init__()
        p = torch.rand(shape)
        p_flat = p.flatten()
        k = ceil(len(p_flat) * sparsity_level)
        topk_vals, topk_inds = torch.topk(input=torch.abs(p_flat), k=k)
        mask = torch.zeros(size=p_flat.shape, device=topk_inds.device)
        ones = torch.ones(size=p_flat.shape, device=topk_inds.device)
        mask.scatter_(dim=0, index=topk_inds, src=ones, reduce='add')
        p *= mask.reshape(p.size())
        # self.sparse_linear = nn.Parameter(p.to_sparse().requires_grad_(True))
        self.sparse_linear = nn.Parameter(p.to_sparse())
        # nn.init.kaiming_uniform_(self.sparse_linear, a=math.sqrt(5))
    

    def forward(self, x):
        return torch.sparse.mm(self.sparse_linear, x)
        # return torch.sparse.mm(x, self.sparse_linear)



class SimpleSparseMLP(nn.Module):
    def __init__(self, sparsity_level):
        super().__init__()
        self.fc1 = SparseLinear((512, 28 * 28), sparsity_level)
        self.fc2 = SparseLinear((512, 512), sparsity_level)
        self.fc3 = SparseLinear((10, 512), sparsity_level)
        self.bias1 = nn.Parameter(torch.rand(size=(512, 100)))
        self.bias2 = nn.Parameter(torch.rand(size=(512, 100)))
        # self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.transpose(0, 1).reshape(-1, batch_size)
        x = x.reshape(-1, batch_size)
        # print(x)
        x = F.relu(torch.add(self.fc1(x), self.bias1))
        # x = torch.sigmoid(self.fc1(x))
        # print(x)
        # x = self.dropout(x)
        x = F.relu(torch.add(self.fc2(x), self.bias2))
        # x = torch.sigmoid(self.fc2(x))
        # x = self.dropout(x)
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
    

    

