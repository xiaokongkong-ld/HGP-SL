import torch
import torch.nn as nn
from ultils import edge_2_adj, adj_2_edge, edge_2_adj_tf

identity = torch.eye(400).float()
print(identity)

x = torch.squeeze(identity)
fc = nn.Linear(400, 400)
mask = fc(identity)
mask = torch.sigmoid(mask)
mask = torch.triu(mask)
mask += mask.T - torch.diag(torch.diag(mask, 0), 0)
print(mask)

# print('weight: ', weight)
adjacency = torch.round(mask)

print(adjacency)

