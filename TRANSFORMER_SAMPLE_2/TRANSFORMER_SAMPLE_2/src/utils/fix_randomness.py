import os
import numpy as np
import random
import torch

def seed_libs(seed=2020):
    """
       Fix the seeds for the random generators of torch and other libraries
       :param seed: Seed to pass to the random seed generators
       """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def seed_torch(seed=2020):
    """
    Fix the seeds for the random generators of torch and other libraries
    :param seed: Seed to pass to the random seed generators
    """

    seed_libs(2020)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
