from collections import defaultdict

import dnnlpy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2
from torch import Tensor

torch.manual_seed(42)
    #固定PyTorch全局随机数生成器的种子为42
dnnlpy.set_matplotlib_format('svg')
    #设置Matplotlib输出图像的格式为 SVG（矢量图）
print('PyTorch version:', torch.__version__)