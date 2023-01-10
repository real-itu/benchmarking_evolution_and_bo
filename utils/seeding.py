import random
import numpy as np
import os
import torch

def seed_python_numpy_torch_cuda(seed: int):
    seed = np.random.randint(0, 2**32 - 1) if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'\nUsing seed for search algorithm {seed}\n')