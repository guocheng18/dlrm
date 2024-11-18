import torch
import numpy as np
from tqdm import tqdm

from dlrm_data_pytorch import CriteoDataset, collate_wrapper_criteo_offset

# Read dataset
train_data = CriteoDataset(
    "kaggle",
    -1,
    0,
    "total",
    "train",
    "./input/train.txt",
    "./input/kaggleAdDisplayChallenge_processed.npz",
    False,
    False,
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=128,
    shuffle=False,
    num_workers=16,
    collate_fn=collate_wrapper_criteo_offset,
    pin_memory=False,
    drop_last=False,
)

emb_table_sizes = [1460,  583, 10131227,  2202608,      305,       24,
                   12517,      633,        3,    93145,     5683,  8351593,
                   3194,       27,    14992,  5461306,       10,     5652,
                   2173,        4,  7046547,       18,       15,   286181,
                   105,   142572]

# Count the frequency of each feature

res = [np.zeros(size) for size in emb_table_sizes]

for batch in tqdm(train_loader):
    batch_emb_indices = batch[2]  # [26, 128]
    for i in range(26):
        emb_indices = batch_emb_indices[i]
        np.add.at(res[i], emb_indices, 1)

np.savez("itemfreq.npz", *res)