import os 
import torch
import random
import numpy as np
from typing import Optional, Tuple
import torch
from torch import Tensor
from typing import Optional, Tuple

def topk_mrr_hr(scores, targets,k):
    hit,mrr = [],[]

    pre_scores = scores.topk(k)[1].numpy()
    for score,label in zip(pre_scores,targets):
        hit.append(np.isin(label, score))
        if len(np.where(score == label)[0]) == 0:
            mrr.append(0)
        else:
            mrr.append(1 / (np.where(score == label)[0][0] + 1))

    hr = np.sum(hit) 
    mrr = np.sum(mrr) 

    return hr,mrr

def fix_seed(seed=42):
    """
    固定随机数种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
