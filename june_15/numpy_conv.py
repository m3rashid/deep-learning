import numpy as np
import torch

# convert numpy array back to torch tensor
numpy_arr = np.zeros((5, 5))
tensor = torch.from_numpy(numpy_arr)
print(tensor)

# convert back to numpy array
arr = tensor.numpy()
