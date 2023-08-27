import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import data

batch_size = 8  # The number of dialogues in a batch


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
