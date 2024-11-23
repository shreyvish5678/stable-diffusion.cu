import numpy as np
import torch
tensor = np.random.rand(4, 64, 64, 3).astype(np.float32)
tensor.tofile('ye.bin')
tensor = torch.from_numpy(tensor)
print(tensor[2, 3, 1, 0])
tensor = tensor.transpose(1, 2)
print(tensor[2, 3, 1, 0])