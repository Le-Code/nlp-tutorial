import torch
print(torch.empty(3, dtype=torch.long).random_(5))